import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter, scatter_softmax

class GATAttentionAggregator(nn.Module):
    def __init__(self, message_dim: int, attention_dim: int, num_heads: int, concat: bool = False, dropout: float = 0.1):
        super().__init__()
        self.message_dim = message_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.concat = concat
        self.negative_slope = 0.2  # For LeakyReLU
        self.dropout = nn.Dropout(dropout)

        # Attention weights
        self.attn = nn.ModuleList([nn.Linear(message_dim, 1, bias=False) for _ in range(num_heads)])
        self.weight = nn.ModuleList([nn.Linear(message_dim, message_dim, bias=False) for _ in range(num_heads)])
        self.leaky_relu = nn.LeakyReLU(self.negative_slope)

    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        if msg.size(0) == 0:
            return msg.new_zeros((dim_size, self.message_dim))

        head_outputs = []

        # Iterate over each attention head
        for i in range(self.num_heads):
            # Create attention scores using the linear layer in self.attn
            attn_scores = self.attn[i](msg)  # Shape: [m, 1]

            # Apply LeakyReLU before softmax
            attn_scores = self.leaky_relu(attn_scores)

            # Normalize attention scores with softmax
            attn_scores = scatter_softmax(src=attn_scores.squeeze(), index=index, dim=0).unsqueeze(1)  # Shape: [m, 1]

            # Apply dropout to attention scores for regularization
            attn_scores = self.dropout(attn_scores)

            # Weight original messages with normalized attention weights
            weighted_msg = attn_scores * msg  # Shape: [m, message_dim]

            # Apply dropout to weighted messages as well
            weighted_msg = self.dropout(weighted_msg)

            # Transform weighted_msg with weight matrix
            weighted_msg = self.weight[i](weighted_msg)  # Shape: [m, message_dim]

            # Sum weighted messages
            sum_weighted_msg = scatter(weighted_msg, index, dim=0, dim_size=dim_size, reduce='sum')  # Shape: [n, message_dim]

            # Append to head outputs
            head_outputs.append(sum_weighted_msg)

        # Concatenate or average the head outputs based on concat flag
        if self.concat:
            out = torch.cat(head_outputs, dim=1)  # Concatenate along the feature dimension
        else:
            out = torch.mean(torch.stack(head_outputs, dim=0), dim=0)  # Average over heads

        return out

