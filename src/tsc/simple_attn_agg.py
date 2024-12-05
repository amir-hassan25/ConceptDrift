import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter, scatter_softmax

class SimpleAttentionAggregator(nn.Module):
    def __init__(self, message_dim: int, attention_dim: int, num_heads: int, concat: bool = True, dropout: float = 0.1):
        super().__init__()
        self.message_dim = message_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.concat = concat
        self.negative_slope = 0.2  # For LeakyReLU
        self.dropout = nn.Dropout(dropout)

        # Attention weights
        self.attn = nn.ModuleList([nn.Linear(message_dim, 1, bias=False) for _ in range(num_heads)])
        
        # # Final linear layer to combine attended messages after concatenation/averaging
        # output_dim = num_heads * attention_dim if concat else attention_dim
        # self.out_layer = nn.Linear(output_dim, message_dim)

    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        """
        Simple attention based message aggregation approach: 



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

            # Create attention scores using the linear layer in self.attn
            attn_scores = self.attn[i](msg)  # Shape: [m, 1]
            # print(f"attn_scores shape: {attn_scores.shape}")

            # Normalize attention scores with softmax
            attn_scores = scatter_softmax(src=attn_scores.squeeze(), index=index, dim=0).unsqueeze(1)  # Shape: [m, 1]
            # print(f"attn_scores (after softmax) shape: {attn_scores.shape}")
            # print(attn_scores)
            # Weight original messages with normalized attention weights
            weighted_msg = attn_scores * msg  # Shape: [m, message_dim]
            # print(f"weighted_msg shape: {weighted_msg.shape}")

            # Average weighted messages
            avg_weighted_msg = scatter(weighted_msg, index, dim=0, dim_size=dim_size, reduce='mean')  # Shape: [n, message_dim]
            # print(f"avg_weighted_msg shape: {avg_weighted_msg.shape}")

            # Append to head outputs
            head_outputs.append(avg_weighted_msg)
            

        # Average head outputs
        out = torch.mean(torch.stack(head_outputs, dim=0), dim=0)  # Shape [dim_size, message_dim]

        return out
