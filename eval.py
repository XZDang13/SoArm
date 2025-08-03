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

from env.reach_cfg import REACH_TASK_CFG

class Trainer:
    def __init__(self):
        cfg = REACH_TASK_CFG()
        cfg.scene.num_envs = 12
        self.env = gymnasium.make("REACH-v0", cfg=cfg)

        self.device = self.env.unwrapped.device

        obs_dim = cfg.observation_space
        action_dim = cfg.action_space

    @torch.no_grad()
    def get_action(self, obs_dict:dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass
    
    def rollout(self):
        obs_dict, info = self.env.reset()

        for i in range(1000):
            #action = self.get_action(obs_dict) 
            action = torch.randn(12, 6, device=self.device)
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