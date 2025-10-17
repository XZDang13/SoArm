import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import MLPLayer, make_mlp_layers, DeterministicHead, CriticHead, Conv2DLayer
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

        self.dim = dim * 2
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
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.25, training=aug)
        return x
    
class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class FrameObservationEncoderNet(nn.Module):
    def __init__(self,feature_dim:int):
        super().__init__()
        self.aug = RandomShiftsAug(4)
        
        self.dim = feature_dim

        self.visual_encoder = nn.Sequential(
            Conv2DLayer(3, 64, 3, 2, 1, F.silu, True),
            Conv2DLayer(64, 128, 3, 2, 1, F.silu, True),
            Conv2DLayer(128, 256, 3, 2, 1, F.silu, True),
            Conv2DLayer(256, 512, 3, 2, 1, F.silu, True)
        )
        
        self.mlp_layer = nn.Sequential(
            MLPLayer(512*7*7, feature_dim, nn.Identity(), True),
            #MLPLayer(1024, feature_dim, F.silu, True),
        )

    def get_features(self, x:torch.Tensor) -> torch.Tensor:
        x = self.aug(x)
        x = self.visual_encoder(x)
        x = x.flatten(1)
        x = self.mlp_layer(x)

        return x
        
    def forward(self, x:torch.Tensor, with_act_func:bool=True, aug:bool=False) -> torch.Tensor:
        if aug:
            x = self.aug(x)
        x = self.visual_encoder(x)
        #x = F.avg_pool2d(x, 7)
        x = x.flatten(1)
        x = self.mlp_layer(x)
        if with_act_func:
            x = F.silu(x)
        return x
    
class MobileFrameObservationEncoderNet(nn.Module):
    def __init__(self, feature_dim:int):
        super().__init__()

        self.dim = feature_dim
        
        self.visual_encoder = torch.hub.load('pytorch/vision', 'mobilenet_v3_large', pretrained=True)
        self.visual_encoder.classifier = nn.Identity()
        self.mlp_layer = nn.Sequential(
            MLPLayer(960, feature_dim, nn.Identity(), True),
        )

    def get_features(self, x:torch.Tensor) -> torch.Tensor:
        x = self.visual_encoder(x)
        x = self.mlp_layer(x)

        return x
        
    def forward(self, x:torch.Tensor, with_act_func:bool=True, aug:bool=False) -> torch.Tensor:

        x = self.visual_encoder(x)
        x = self.mlp_layer(x)
        if with_act_func:
            x = F.silu(x)
        return x
    
class EfficientNetFrameObservationEncoderNet(nn.Module):
    def __init__(self, feature_dim:int):
        super().__init__()

        self.dim = feature_dim
        
        self.visual_encoder = torch.hub.load("pytorch/vision", "efficientnet_v2_s", weights="IMAGENET1K_V1")
        self.visual_encoder.classifier = nn.Identity()
        self.mlp_layer = nn.Sequential(
            MLPLayer(1280, feature_dim, nn.Identity(), True),
        )

    def get_features(self, x:torch.Tensor) -> torch.Tensor:
        x = self.visual_encoder(x)
        x = self.mlp_layer(x)

        return x
        
    def forward(self, x:torch.Tensor, with_act_func:bool=True, aug:bool=False) -> torch.Tensor:

        x = self.visual_encoder(x)
        x = self.mlp_layer(x)
        if with_act_func:
            x = F.silu(x)
        return x

class DinoFrameObservationEncoderNet(nn.Module):
    def __init__(self, feature_dim:int):
        super().__init__()

        self.dim = feature_dim
        
        self.visual_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        self.mlp_layer = nn.Sequential(
            MLPLayer(384, 1024, nn.SiLU(), True),
            MLPLayer(1024, feature_dim, nn.Identity(), True),
        )

    def get_features(self, x:torch.Tensor) -> torch.Tensor:
        
        x = self.visual_encoder(x)
        x = self.mlp_layer(x)

        return x
        
    def forward(self, x:torch.Tensor, with_act_func:bool=True, aug:bool=False) -> torch.Tensor:
        x = self.visual_encoder(x)
        x = self.mlp_layer(x)
        if with_act_func:
            x = F.silu(x)
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