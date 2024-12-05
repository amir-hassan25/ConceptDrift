import torch
from torch import nn, Tensor

class Router(nn.Module):
    """ A smaller router to assign weights to experts based on the input """
    def __init__(self, n_embd: int, num_experts: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_embd, n_embd * 2), nn.ReLU(),  # Reduced from n_embd * 4 to n_embd * 2
            nn.Linear(n_embd * 2, num_experts),        # Removed one intermediate layer
            nn.Softmax(dim=-1)
        )

    def forward(self, x: Tensor):
        return self.net(x)

class FeedForward(nn.Module):
    """ A smaller feed forward network used as an expert """
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size * 2), nn.ReLU(),  # Reduced from input_size * 4 to input_size * 2
            nn.Linear(input_size * 2, output_size),            # Reduced the depth
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor):
        return self.net(x)

class MoEMessage(nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int, num_experts: int = 3, dropout: float = 0.1):
        super().__init__()
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim
        self.num_experts = num_experts
        
        # Define router to determine the expert weights
        self.router = Router(self.out_channels, num_experts, dropout=dropout)
        
        # Define the experts as smaller feedforward networks
        self.experts = nn.ModuleList([
            FeedForward(self.out_channels, self.out_channels, dropout=dropout) for _ in range(num_experts)
        ])
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, z_src: Tensor, z_dst: Tensor, raw_msg: Tensor, t_enc: Tensor):
        # Concatenate to get the identity message
        identity_msg = torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)
        
        # Apply dropout to the identity message
        identity_msg = self.dropout(identity_msg)
        
        # Get the routing probabilities for the experts
        expert_weights = self.router(identity_msg)  # Shape: [batch_size, num_experts]
        
        # Compute the output of each expert
        expert_outputs = torch.stack([expert(identity_msg) for expert in self.experts], dim=-2)  # Shape: [batch_size, ..., num_experts, output_dim]
        
        # Reshape expert_weights to match expert outputs for weighting
        expert_weights = expert_weights.unsqueeze(-1)  # Shape: [batch_size, num_experts, 1]
        
        # Compute the weighted sum of expert outputs
        weighted_sum = torch.sum(expert_outputs * expert_weights, dim=-2)  # Sum over experts: [batch_size, output_dim]
        
        return weighted_sum


# class MoEMessage(torch.nn.Module):
#     def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int, num_experts: int = 3, dropout: float = 0.1):
#         super().__init__()
#         self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim
#         self.num_experts = num_experts
        
#         # Define the experts as a list of linear layers
#         self.experts = torch.nn.ModuleList([
#             torch.nn.Linear(self.out_channels, self.out_channels) for _ in range(num_experts)
#         ])
        
#         # Gating network to decide how to weight the experts' outputs
#         self.gating_network = torch.nn.Linear(self.out_channels, num_experts)
        
#         # Dropout layer
#         self.dropout = nn.Dropout(p=dropout)
    
#     def forward(self, z_src: Tensor, z_dst: Tensor, raw_msg: Tensor, t_enc: Tensor):
#         # Concatenate to get the identity message
#         identity_msg = torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)
        
#         # Apply dropout to the identity message
#         identity_msg = self.dropout(identity_msg)
        
#         # Pass the identity message through the gating network to get the weights for each expert
#         expert_weights = torch.softmax(self.gating_network(identity_msg), dim=-1)
        
#         # Compute the output of each expert
#         expert_outputs = torch.stack([expert(identity_msg) for expert in self.experts], dim=-1)
        
#         # Optionally, apply dropout to expert outputs (can be skipped if not needed)
#         expert_outputs = self.dropout(expert_outputs)
        
#         # Reshape expert_weights to broadcast over the out_channels dimension
#         expert_weights = expert_weights.unsqueeze(1)  # Shape becomes [batch_size, 1, num_experts]

#         # print("Id message shape: ", identity_msg.shape)
#         # print("expert weights shape: ", expert_weights.shape)
#         # print("expert outputs shape: ", expert_outputs.shape)

#         # Combine the expert outputs using the weights
#         final_message = torch.sum(expert_outputs * expert_weights, dim=-1)
        
#         return final_message

