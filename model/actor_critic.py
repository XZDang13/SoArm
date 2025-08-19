import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import MLPLayer, make_mlp_layers, DeterministicHead, CriticHead
from RLAlg.nn.steps import DeterministicContinuousPolicyStep, ValueStep

class EncoderNet(nn.Module):
    def __init__(self, state_dim:int, hidden_dims:list[int]):
        super().__init__()
        
        self.layers = nn.ModuleList(self.init_layers(state_dim, hidden_dims))

    def init_layers(self, in_dim:int, hidden_dims:list[int]):
        layers = []
        dim = in_dim
        
        for hidden_dim in hidden_dims:
            mlp = MLPLayer(dim, hidden_dim, nn.Identity(), True)
            dim = hidden_dim

            layers.append(mlp)

        self.dim = dim
        return layers
    
    def get_features(self, x:torch.Tensor) -> list[torch.Tensor]:
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
            x = F.silu(x)

        return features

    def forward(self, x:torch.Tensor, aug:bool=False) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            x = F.silu(x)
            x = F.dropout(x, p=0.1, training=aug)

        return x
    
class StochasticDDPGActor(nn.Module):
    def __init__(self, feature_dim, hidden_dims, action_dim):
        super().__init__()

        self.layers, in_dim = make_mlp_layers(feature_dim, hidden_dims, nn.SiLU(), True)
        self.policy_layer = DeterministicHead(in_dim, action_dim, max_action=1.0)

    def forward(self, feature, std):
        x = self.layers(feature)
        step:DeterministicContinuousPolicyStep = self.policy_layer(x, std)

        return step
    
class QNet(nn.Module):
    def __init__(self, feature_dim, hidden_dims, action_dim):
        super().__init__()
        self.layers, in_dim = make_mlp_layers(feature_dim+action_dim, hidden_dims, nn.SiLU(), True)
        self.critic_layer = CriticHead(in_dim)

    def forward(self, feature, action):
        x = torch.cat([feature, action], 1)
        x = self.layers(x)
        step:ValueStep = self.critic_layer(x)

        return step
    
class Critic(nn.Module):
    def __init__(self, feature_dim, hidden_dims, action_dim):
        super().__init__()
        self.qnet_1 = QNet(feature_dim, hidden_dims, action_dim)
        self.qnet_2 = QNet(feature_dim, hidden_dims, action_dim)

    def forward(self, feature, action):
        q1 = self.qnet_1(feature, action)
        q2 = self.qnet_2(feature, action)

        return q1, q2