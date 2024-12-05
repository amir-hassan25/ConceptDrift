import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.utils import scatter

class CrossAttentionAggregator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, heads: int = 1, device=None, dropout: float = 0.3):
        super(CrossAttentionAggregator, self).__init__()
        self.out_dim = out_dim  # for debugging purposes
        self.device = device
        self.heads = heads
        self.hidden_dim = hidden_dim // heads

        # Linear layers for query, key, and value projections
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(out_dim, hidden_dim)
        self.value_proj = nn.Linear(out_dim, hidden_dim)

        # Output projection layer to combine attention results from multiple heads
        self.out_proj = nn.Linear(hidden_dim, out_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, node: Tensor, messages: Tensor, index: Tensor, t: Tensor, dim_size: int):
        """
        Args:
            node (Tensor): The node embeddings before message passing, shape [num_nodes, input_dim].
            messages (Tensor): The messages after message passing, shape [num_messages, input_dim].
            index (Tensor): The index of the nodes to which messages are aggregated, shape [num_messages].
            t (Tensor): Time information, not used here but passed for consistency.
            dim_size (int): The total number of nodes in the batch.
        """
        # Check if there are any messages to process
        if messages.size(0) == 0:
            # If no messages, return zero tensor
            return torch.zeros(node.shape[0], self.out_dim).to(self.device)
        
        # Compute the Query, Key, and Value projections
        Q = self.query_proj(node)  # Shape: [num_nodes, hidden_dim]
        K = self.key_proj(messages)  # Shape: [num_messages, hidden_dim]
        V = self.value_proj(messages)  # Shape: [num_messages, hidden_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(-1, self.heads, self.hidden_dim)  # Shape: [num_nodes, heads, hidden_dim]
        K = K.view(-1, self.heads, self.hidden_dim)  # Shape: [num_messages, heads, hidden_dim]
        V = V.view(-1, self.heads, self.hidden_dim)  # Shape: [num_messages, heads, hidden_dim]

        # Scatter the keys and values to ensure node-wise aggregation
        K = scatter(K, index, dim=0, dim_size=dim_size, reduce='mean')  # Shape: [num_nodes, heads, hidden_dim]
        V = scatter(V, index, dim=0, dim_size=dim_size, reduce='mean')  # Shape: [num_nodes, heads, hidden_dim]

        # Compute attention scores (scaled dot-product attention)
        scores = torch.einsum("nhd,nmd->nhm", Q, K)  # Shape: [num_nodes, heads, num_nodes]
        scores = scores / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)  # Shape: [num_nodes, heads, num_nodes]

        # Apply dropout to attention weights (optional)
        attention_weights = self.dropout(attention_weights)

        # Compute the weighted sum of the values
        attention_output = torch.einsum("nhm,nmd->nhd", attention_weights, V)  # Shape: [num_nodes, heads, hidden_dim]

        # Reshape and project the output to the desired out_dim
        attention_output = attention_output.view(-1, self.hidden_dim * self.heads)  # Shape: [num_nodes, hidden_dim * heads]

        # Apply dropout after attention output
        attention_output = self.dropout(attention_output)

        # Final projection to out_dim
        output = self.out_proj(attention_output)  # Shape: [num_nodes, out_dim]

        return output

