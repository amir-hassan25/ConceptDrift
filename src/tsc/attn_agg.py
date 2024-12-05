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