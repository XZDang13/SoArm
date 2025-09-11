import numpy as np
import omni
import torch
from torchvision.transforms import v2

from model.actor_critic import FrameObservationEncoderNet, StochasticDDPGActor
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
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def load_policy(self):
        self.device = torch.device("cuda:0")
        self.encoder = FrameObservationEncoderNet(6, 256).to(self.device)
        self.actor = StochasticDDPGActor(self.encoder.dim, [256, 256], 6).to(self.device)

        encoder_params, actor_params, _ = torch.load("frame_model.pth")
        self.encoder.load_state_dict(encoder_params)
        self.actor.load_state_dict(actor_params)

        self.encoder.eval()
        self.actor.eval()

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
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
            feature = self.encoder(obs)
            step = self.actor(feature, std=1.0)
            action = step.mean.cpu().detach().view(-1).numpy()
        return action