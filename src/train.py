import copy
import pickle
import torch
import argparse
import configparser
from tqdm import tqdm
import os.path as osp
from tsc.tsc import TemporalSemanticContextualization
from tsc.moe_messg_pass import MoEMessage
from tsc.attn_agg import MultiHeadAttentionAggregator
from tsc.gat_attn_agg import GATAttentionAggregator
from tsc.cross_attn_agg import CrossAttentionAggregator
from tsc.simple_attn_agg import SimpleAttentionAggregator
from utils.utils import save_model


from typing import Callable, Dict, Tuple
from torch.nn import Linear
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TransformerConv
from tsc.tsc import (
    ConcatMessage,
    MeanAggregator,
    LastAggregator,
    LastNeighborLoader,
)
import torch
from torch import Tensor
from torch.nn import GRUCell, Linear, Dropout
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import scatter
from torch_geometric.utils._scatter import scatter_argmax
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score


### CREDIT: Code borrows from Torch-Geometric: https://github.com/pyg-team/pytorch_geometric


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add argument parser
parser = argparse.ArgumentParser(description="ConceptDrift")

# ConceptDrift uses the default hyperparameters:
parser.add_argument("--dataset", type=str, required=True, help="Dataset to use ('immunology', 'virology', 'neurology')")
parser.add_argument("--aggregator", type=str, required=False, default="last", help="Batch processing logic to use ('last', 'mean', 'attn')")  ### This is specific for batch processing logic borrowed from a TGN. ConceptDrift uses 'last'.
parser.add_argument("--messenger", type=str, required=False, default="concat", help="Message Module to use ('concat', 'moe')") ### Concatonate for Context Aggregation Function. 
parser.add_argument("--dropout", type=float, required=False, default=0.1, help="Float value for dropout (e.g. 0.3)")
parser.add_argument("--batch_size", type=int, required=False, default=200, help="Batch Size")
parser.add_argument("--max_epochs", type=int, required=False, default=2, help="Batch Size")
parser.add_argument("--lr", type=float, required=False, default=0.0001, help="Learning rate")
args = parser.parse_args()

# add config parser
config = configparser.ConfigParser()
config.read('config.ini')


# Load the dataset based on the argument

base_data_path = config.get('DEFAULT','data_dir')

# Assuming dataset names and paths
graph_paths = {
    'immunology': f'{base_data_path}/immunology/immunology_event_temp_graph.pkl',
    'virology': f'{base_data_path}/virology/virology_event_temp_graph.pkl',
    'neurology': f'{base_data_path}/neurology/neurology_event_temp_graph.pkl'
}

# Assuming dataset names and paths
node_feat_paths = {
    'immunology': f'{base_data_path}/immunology/biobert_immunology.pt',
    'virology': f'{base_data_path}/virology/biobert_virology.pt',
    'neurology': f'{base_data_path}/neurology/biobert_neurology.pt'
}


# Verify that the dataset argument is valid
if args.dataset not in graph_paths:
    raise ValueError(f"Invalid dataset name: {args.dataset}. Available datasets: {list(graph_paths.keys())}")

graph_path = graph_paths[args.dataset]

print(f'loaded graph dataset from {graph_path}')


with open(graph_path, 'rb') as f:
    data = pickle.load(f)


node_feat_path = node_feat_paths[args.dataset]
# Assuming `node_features` is a tensor of size [num_nodes, 768]
# node_features = torch.load(f'/scratch/ahs5ce/PubTator3/temporal_cooc_graphs/mesh_node_embeddings/biobert_{args.dataset}.pt').to(device)
node_features = torch.load(node_feat_path).to(device)

print(f'loaded node features from {node_feat_path}')


data.msg = torch.full([data.src.shape[0],25], 1) # making dim smaller because it cannot fit in memory
print('created data.msg')
data.msg = data.msg.to(torch.float)
print('casted data.msg to float')
data = data.to(device)
print('data put on device')


# This datasplit ensures that 2000-2022 is for training, 2023 is for validation and 2024 is for testing.
train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=0.2, test_ratio=0.1)

train_loader = TemporalDataLoader(
    train_data,
    batch_size=args.batch_size,
    neg_sampling_ratio=1.0,
)
val_loader = TemporalDataLoader(
    val_data,
    batch_size=args.batch_size,
    neg_sampling_ratio=1.0,
)
test_loader = TemporalDataLoader(
    test_data,
    batch_size=args.batch_size,
    neg_sampling_ratio=1.0,
)
print('created loaders')


neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)


class TemporalCrossAttn(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)
        self.dropout = Dropout(p=0.1)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        x = self.conv(x, edge_index, edge_attr)
        return self.dropout(x)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)


memory_dim = time_dim = embedding_dim = 768 # BioBERT embedding dimension
attention_dim = embedding_dim // 2
print("Data message size: ", data.msg.shape )

# Choose message_module 
messengers = {
    "concat": ConcatMessage(data.msg.size(-1), memory_dim, time_dim),
    "moe": MoEMessage(data.msg.size(-1), memory_dim, time_dim, dropout=args.dropout),
}
message_module=messengers[args.messenger]
print("ID MESSAGE OUT CHANNELS: ", message_module.out_channels)

# Batch processing logic. ConceptDrift uses 'last'. 
aggregators = {
    "last": LastAggregator(),
    "mean": MeanAggregator(),
    "gat-attn": GATAttentionAggregator(message_dim=message_module.out_channels, attention_dim=attention_dim, num_heads=2, dropout=args.dropout),
    "smpl-attn": SimpleAttentionAggregator(message_dim=message_module.out_channels, attention_dim=attention_dim, num_heads=2, dropout=args.dropout),
    "attn": MultiHeadAttentionAggregator(message_dim=message_module.out_channels, attention_dim=attention_dim, num_heads=2, dropout=args.dropout),
    "cross_attn": CrossAttentionAggregator(input_dim=memory_dim, hidden_dim=memory_dim, out_dim=message_module.out_channels, heads=8, device=device, dropout=args.dropout)
}
aggregator = aggregators[args.aggregator]


memory = TemporalSemanticContextualization(
    data.num_nodes,
    data.msg.size(-1),
    memory_dim,
    time_dim,
    node_features,
    message_module=message_module,
    aggregator_module=aggregator,
).to(device)

gnn = TemporalCrossAttn(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters())
    | set(link_pred.parameters()), lr=args.lr)
criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

def train():
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
    # for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        # print(batch)
        # print(batch.n_id)
        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))
    
        pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events
        # print("one batch")
    return total_loss / train_data.num_events



@torch.no_grad()
def test(loader):
    memory.eval()
    gnn.eval()
    link_pred.eval()

    if isinstance(aggregator, MultiHeadAttentionAggregator):
        print("setting attention aggregator to eval mode")
        aggregator.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    aps, aucs, f1s, recalls = [], [], [], []
    for batch in tqdm(loader, desc="Evaluation"):
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                    data.msg[e_id].to(device))
        
        pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)),
             torch.zeros(neg_out.size(0))], dim=0)

        # Calculate metrics
        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))


        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)

    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())


best_val_ap = 0.0  # Initialize the best validation AP
best_model_path = None  # To keep track of the best model saved
patience = 3  # Number of epochs to wait for improvement before stopping
patience_counter = 0  # Counter to track how long validation AP hasn't improved

max_epochs = args.max_epochs  
for epoch in range(1, max_epochs + 1):
    print(f'Epoch: {epoch:02d}')
    
    # Train the model for the current epoch
    loss = train()
    print(f'Loss: {loss:.4f}')
    
    # Evaluate on the validation set
    val_ap, val_auc = test(val_loader)
    test_ap, test_auc = test(test_loader)

    print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')
    print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')

    # Check if the current validation AP is higher than the best one so far
    if val_ap > best_val_ap:
        best_val_ap = val_ap  # Update the best validation AP
        best_model_path = f'saved_models/best_model_{args.dataset}_agg-{args.aggregator}_messenger-{args.messenger}_batchsize-{args.batch_size}_epoch_{epoch}.pth'  # Define the new best model path
        
        # Save the model if the validation AP has improved
        save_model(best_model_path, epoch, memory, gnn, link_pred, optimizer, loss, model_dir="saved_models")
        print(f"New best model saved with Val AP: {val_ap:.4f} at epoch {epoch}")
        
        patience_counter = 0  # Reset the patience counter since we had improvement
    else:
        patience_counter += 1  # Increment patience counter if no improvement
        print(f"No improvement in Val AP. Patience counter: {patience_counter}/{patience}")
    
    # Early stopping check
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}. Best Val AP: {best_val_ap:.4f}")
        break  # Stop training if patience is exhausted

print(f'Training completed. Best model saved at: {best_model_path} with Val AP: {best_val_ap:.4f}')

