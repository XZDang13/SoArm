import numpy as np
import omni
import torch

from model.actor_critic import EncoderNet, StochasticDDPGActor
from .policy_controller import PolicyController

class StatePolicyController(PolicyController):

    def __init__(
        self,
        robot,
        cube,
        camera
    ) -> None:
        super().__init__(robot, cube, camera)
        self.load_policy()
        self.load_config()

    def load_policy(self):
        self.device = torch.device("cuda:0")
        self.encoder = EncoderNet(6+6+3+4+3+4, [256, 256, 256]).to(self.device)
        self.actor = StochasticDDPGActor(self.encoder.dim, [256, 256], 6).to(self.device)

        encoder_params, actor_params, _ = torch.load("state_model.pth")
        self.encoder.load_state_dict(encoder_params)
        self.actor.load_state_dict(actor_params)

        self.encoder.eval()
        self.actor.eval()

    def _compute_action(self, obs: np.ndarray, deterministic:bool=True) -> np.ndarray:
        """
        Computes the action from the observation using the loaded policy.

        Args:
            obs (np.ndarray): The observation.

        Returns:
            np.ndarray: The action.
        """
        with torch.no_grad():
            obs = obs.view(1, -1).float().to(self.device)
            feature = self.encoder(obs)
            step = self.actor(feature, std=1.0)
            if deterministic:
                action = step.mean
            else:    
                action = step.pi.rsample()
            action = action.cpu().detach().view(-1).numpy()
        return action