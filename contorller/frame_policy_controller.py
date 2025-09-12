import numpy as np
import omni
import torch
from torchvision.transforms import v2

from model.actor_critic import FrameObservationEncoderNet, StochasticDDPGActor, EncoderNet
from .policy_controller import PolicyController

class FramePolicyController(PolicyController):

    def __init__(
        self,
        robot,
        cube,
        camera
    ) -> None:
        super().__init__(robot, cube, camera)
        self.load_policy()
        self.load_config()

        self.transform = v2.Compose([
            v2.Resize((112, 112)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def load_policy(self):
        self.device = torch.device("cuda:0")
        self.frame_encoder = FrameObservationEncoderNet(6, 256).to(self.device)
        self.state_encoder = EncoderNet(6+6+3+4+3+4, [256, 256, 256]).to(self.device)
        self.actor = StochasticDDPGActor(self.frame_encoder.dim, [256, 256], 6).to(self.device)

        encoder_params, actor_params, _ = torch.load("frame_model.pth")
        self.frame_encoder.load_state_dict(encoder_params)
        self.actor.load_state_dict(actor_params)

        encoder_params, _, _ = torch.load("state_model.pth")
        self.state_encoder.load_state_dict(encoder_params)

        self.frame_encoder.eval()
        self.state_encoder.eval()
        self.actor.eval()

    def get_state_feature(self, obs):
        obs = obs.view(1, -1).float().to(self.device)
        feature = self.state_encoder(obs)

        return feature
    
    def get_frame_feature(self, obs):
        obs = self.transform(obs)
        obs = torch.concat(obs).unsqueeze(0).to(self.device)
        feature = self.frame_encoder(obs, True)

        return feature
    
    def compare_feature(self, state_obs, frame_obs):
        state_feat = self.get_state_feature(state_obs)
        frame_feat = self.get_frame_feature(frame_obs)

        print(torch.cosine_similarity(state_feat, frame_feat))

    def _compute_action(self, obs: np.ndarray, deterministic:bool) -> np.ndarray:
        """
        Computes the action from the observation using the loaded policy.

        Args:
            obs (np.ndarray): The observation.

        Returns:
            np.ndarray: The action.
        """
        with torch.no_grad():
            obs = self.transform(obs)
            obs = torch.concat(obs).unsqueeze(0).to(self.device)
            feature = self.frame_encoder(obs, True)
            step = self.actor(feature, std=1.0)
            if deterministic:
                action = step.mean
            else:    
                action = step.pi.rsample()
            action = action.cpu().detach().view(-1).numpy()
        return action