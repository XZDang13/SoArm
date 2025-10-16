import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium
import torch
import numpy as np

from RLAlg.nn.steps import DeterministicContinuousPolicyStep

from env.reach_cfg import REACH_TASK_CFG
from env.stack_cfg import STACK_TASK_CFG
from model.actor_critic import EncoderNet, StochasticDDPGActor

class Trainer:
    def __init__(self):
        cfg = STACK_TASK_CFG()
        cfg.scene.num_envs = 1
        cfg.is_training = False
        self.env = gymnasium.make("STACK-v0", cfg=cfg)

        self.device = self.env.unwrapped.device
        self.num_envs = cfg.scene.num_envs
        self.state_dim = cfg.state_space
        self.obs_dim = cfg.observation_space
        self.action_dim = cfg.action_space

        self.encoder = EncoderNet(self.obs_dim, [128, 128, 128]).to(self.device)
        self.actor = StochasticDDPGActor(self.encoder.dim, [256, 256], self.action_dim).to(self.device)

        encoder_params, actor_params, _ = torch.load("state_model.pth")
        self.encoder.load_state_dict(encoder_params)
        self.actor.load_state_dict(actor_params)

        self.encoder.eval()
        self.actor.eval()

    @torch.no_grad()
    def get_action(self, obs_dict:dict[str, torch.Tensor], deterministic=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = obs_dict["policy"]

        feature = self.encoder(obs)
        step:DeterministicContinuousPolicyStep = self.actor(feature, std=1.0)

        if deterministic:
            action = step.mean
        else:    
            action = step.pi.rsample()

        return action
    
    def rollout(self):
        obs_dict, info = self.env.reset()
        for i in range(1000):
            action = self.get_action(obs_dict, True)
            print(action)
            #print(self.env.unwrapped.end_effector.data.target_pos_source[:, :, :])
            #print(self.env.unwrapped.end_effector_tcp.data.target_pos_source[:, 0, :])

            #print(obs_dict["policy"][:, :, 7:14])
            #print("---------------")
            
            next_obs_dict, reward, terminate, timeout, info = self.env.step(action)
            done = terminate | timeout
            obs_dict = next_obs_dict

def main():
    trainer = Trainer()

    trainer.rollout()

    trainer.env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()