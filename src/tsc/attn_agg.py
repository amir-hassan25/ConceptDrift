import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter, scatter_softmax

class MultiHeadAttentionAggregator(nn.Module):
    def __init__(self, message_dim: int, attention_dim: int, num_heads: int, concat: bool = True, dropout: float = 0.1):
        super().__init__()
        self.message_dim = message_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.concat = concat
        self.negative_slope = 0.2  # For LeakyReLU
        self.dropout = nn.Dropout(dropout)

        # Linear transformations for multi-head attention
        self.query_weights = nn.ModuleList([nn.Linear(message_dim, attention_dim, bias=False) for _ in range(num_heads)])
        self.key_weights = nn.ModuleList([nn.Linear(message_dim, attention_dim, bias=False) for _ in range(num_heads)])
        self.value_weights = nn.ModuleList([nn.Linear(message_dim, attention_dim, bias=False) for _ in range(num_heads)])

        # Final linear layer to combine attended messages after concatenation/averaging
        output_dim = num_heads * attention_dim if concat else attention_dim
        self.out_layer = nn.Linear(output_dim, message_dim)

    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        """
        Args:
            msg: Tensor of shape [m, message_dim] representing messages for each interaction.
            index: Tensor of shape [m], mapping each message to a node.
            t: Tensor of shape [m], representing the time of each message (optional, for possible future use).
            dim_size: The total number of nodes.

        Returns:
            out: Tensor of shape [dim_size, message_dim], where each row corresponds to the aggregated message for a node.
        """
        if msg.size(0) == 0:
            return msg.new_zeros((dim_size, self.message_dim))

        head_outputs = []

        # Iterate over each attention head
        for i in range(self.num_heads):
            # Apply value transformation for each head
            values = self.value_weights[i](msg)  # Shape: [m, attention_dim]
            values = self.dropout(values)  # Apply dropout

            # Compute query and key projections for each head
            queries = self.query_weights[i](msg)  # Shape: [m, attention_dim]
            queries = self.dropout(queries)  # Apply dropout

            keys = self.key_weights[i](msg)  # Shape: [m, attention_dim]
            keys = self.dropout(keys)  # Apply dropout

            # Compute query * key attention scores for each message (dot product)
            attention_scores = (queries * keys).sum(dim=-1, keepdim=True)  # [m, 1]

            # Apply LeakyReLU activation for attention scores
            attention_scores = nn.functional.leaky_relu(attention_scores, self.negative_slope)  # [m, 1]

            # Normalize attention scores with scatter_softmax
            attention_scores = scatter_softmax(attention_scores, index, dim=0)  # Shape: [m, 1]

            # Weight the values by the attention scores
            weighted_values = values * attention_scores  # Shape: [m, attention_dim]

            # Aggregate the weighted values by summing them for each node
            out_per_head = scatter(weighted_values, index, dim=0, dim_size=dim_size, reduce='sum')  # Shape: [dim_size, attention_dim]
            head_outputs.append(out_per_head)

        # Concatenate or average the outputs from each head
        if self.concat:
            out = torch.cat(head_outputs, dim=-1)  # Concatenate heads along the feature dimension
        else:
            out = torch.stack(head_outputs).mean(dim=0)  # Average the heads

        # Final linear transformation
        out = self.out_layer(out)
        out = self.dropout(out)  # Apply dropout before the final output

        return out

# import torch
# import torch.nn as nn
# from torch import Tensor
# from torch_scatter import scatter, scatter_softmax

# class AttentionAggregator(nn.Module):
#     def __init__(self, message_dim: int, attention_dim: int):
#         super().__init__()
#         self.message_dim = message_dim
#         self.attention_dim = attention_dim

#         # Linear transformations for query, key, and value
#         self.query = nn.Linear(message_dim, attention_dim)
#         self.key = nn.Linear(message_dim, attention_dim)
#         self.value = nn.Linear(message_dim, attention_dim)

#         # Final linear layer to combine attended messages
#         self.out_layer = nn.Linear(attention_dim, message_dim)

#     def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
#         """
#         Args:
#             msg: Tensor of shape [m, message_dim] representing messages for each interaction.
#             index: Tensor of shape [m], mapping each message to a node.
#             t: Tensor of shape [m], representing the time of each message (optional, for possible future use).
#             dim_size: The total number of nodes.

#         Returns:
#             out: Tensor of shape [dim_size, message_dim], where each row corresponds to the aggregated message for a node.
#         """
#         # Early exit if no messages are present
#         if msg.size(0) == 0:
#             return msg.new_zeros((dim_size, self.message_dim))

#         # Compute query, key, and value projections for the messages
#         queries = self.query(msg)  # Shape: [m, attention_dim]
#         keys = self.key(msg)       # Shape: [m, attention_dim]
#         values = self.value(msg)   # Shape: [m, attention_dim]

#         print(f"msg shape: {msg.shape}")
#         print(f"index shape: {index.shape}")        
#         print("queries shape: ", queries.shape)
#         print("keys shape: ", keys.shape)
#         print("values shape: ", values.shape)
#         # Normalize the keys and queries for stability in attention score calculation
#         queries = queries / torch.sqrt(torch.tensor(self.attention_dim, dtype=torch.float32))
#         print("queries shape: ", queries.shape)


#         # Compute attention scores for messages belonging to the same node
#         # attention_scores = scatter(queries * keys, index, dim=0, dim_size=dim_size, reduce='sum')
#         attention_scores = scatter(queries * keys, index, dim=0, dim_size=index.shape[0], reduce='sum')
#         print(f'dim_size', dim_size)
#         print(f"attention_scores shape: {attention_scores.shape}")
#         print(f"attention_scores.sum(dim=-1, keepdim=True) shape: {attention_scores.sum(dim=-1, keepdim=True).shape}")
#         # Use scatter_softmax to normalize attention scores within each node
#         attention_scores = scatter_softmax(attention_scores.sum(dim=-1, keepdim=True), index, dim=0)

#         # Weight the values by the attention scores
#         weighted_values = values * attention_scores

#         # Aggregate the weighted values by summing them for each node
#         out = scatter(weighted_values, index, dim=0, dim_size=dim_size, reduce='sum')
#         print("out ", out.shape)
#         # Apply the final linear layer to combine the attended messages
#         out = self.out_layer(out)

#         return out