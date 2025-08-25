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
import torch.optim as optim

import numpy as np

from tqdm import trange

from RLAlg.alg.ddpg_double_q import DDPGDoubleQ
from RLAlg.utils import set_seed_everywhere
from RLAlg.buffer.replay_buffer import ReplayBuffer
from RLAlg.nn.steps import DeterministicContinuousPolicyStep, ValueStep

from model.actor_critic import EncoderNet, StochasticDDPGActor, Critic

from env.reach_cfg import REACH_TASK_CFG
from env.stack_cfg import STACK_TASK_CFG

def process_obs(obs):
    features = obs["policy"]

    return features

class Trainer:
    def __init__(self):
        cfg = STACK_TASK_CFG()
        cfg.scene.num_envs = 256
        self.env = gymnasium.make("STACK-v0", cfg=cfg)

        self.env_nums, self.obs_dim = self.env.observation_space.shape

        self.action_dim = self.env.action_space.shape[1]
        self.device = self.env.unwrapped.device

        self.encoder = EncoderNet(self.obs_dim, [256, 256, 256]).to(self.device)
        self.actor = StochasticDDPGActor(self.encoder.dim, [256, 256], self.action_dim).to(self.device)
        self.critic = Critic(self.encoder.dim, [256, 256], self.action_dim).to(self.device)
        self.critic_target = Critic(self.encoder.dim, [256, 256], self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=3e-4)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.steps = 25

        self.buffer_steps = 10000
        self.rollout_buffer = ReplayBuffer(self.env_nums, self.buffer_steps)

        self.rollout_buffer.create_storage_space("observations", (self.obs_dim,), torch.float32)
        self.rollout_buffer.create_storage_space("next_observations", (self.obs_dim,), torch.float32)
        self.rollout_buffer.create_storage_space("actions", (self.action_dim,), torch.float32)
        self.rollout_buffer.create_storage_space("rewards", (), torch.float32)
        self.rollout_buffer.create_storage_space("dones", (), torch.float32)

        self.batch_keys = ["observations", "next_observations", "actions", "rewards", "dones"]
        
        self.obs = None

        self.epochs = 500
        self.update_iteration = 50
        self.batch_size = self.env_nums * 10
        self.gamma = 0.99
        self.tau = 0.005
        self.regularization_weight = 1

        self.std = 1e-3

    @torch.no_grad()
    def get_action(self, obs_batch:list[list[float]], deterministic:bool=False):
        obs_batch = self.encoder(obs_batch)
        action_step:DeterministicContinuousPolicyStep = self.actor(obs_batch, self.std)
        
        if deterministic:
            action = action_step.mean
        else:    
            action = action_step.pi.rsample()
        
        return action
    
    def rollout(self):
        self.encoder.eval()
        self.actor.eval()
        self.critic.eval()

        obs = self.obs
        
        for i in range(self.steps):
            features = process_obs(obs)
            action = self.get_action(features)
            next_obs, reward, terminate, timeout, info = self.env.step(action)
            next_features = process_obs(next_obs)
            done = terminate | timeout

            record = {
                "observations": features,
                "next_observations": next_features,
                "actions": action,
                "rewards": reward,
                "dones": done
            }

            self.rollout_buffer.add_records(record)

            obs = next_obs

        self.obs = obs

        self.encoder.train()
        self.actor.train()
        self.critic.train()

    def update(self, num_iteration:int, batch_size:int):
        for _ in range(num_iteration):
            batch = self.rollout_buffer.sample_batch(self.batch_keys, batch_size)

            obs_batch = batch["observations"].to(self.device)
            next_obs_batch = batch["next_observations"].to(self.device)
            action_batch = batch["actions"].to(self.device)
            reward_batch = batch["rewards"].to(self.device)
            done_batch = batch["dones"].to(self.device)

            feature_batch = self.encoder(obs_batch, True)
            with torch.no_grad():
                next_feature_batch = self.encoder(next_obs_batch, True)

            self.encoder_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss = DDPGDoubleQ.compute_critic_loss(
                self.actor, self.critic, self.critic_target, feature_batch, action_batch, reward_batch, next_feature_batch, done_batch, self.std, self.gamma
            )

            critic_loss.backward()
            self.critic_optimizer.step()
            self.encoder_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = False

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss = DDPGDoubleQ.compute_policy_loss(self.actor, self.critic, feature_batch.detach(), self.std, self.regularization_weight)
            actor_loss.backward()
            self.actor_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = True

            DDPGDoubleQ.update_target_param(self.critic, self.critic_target, self.tau)

    def train(self):
        obs, info = self.env.reset()
        self.obs = obs
        for epoch in trange(self.epochs):
            self.rollout()
            self.update(self.update_iteration, self.batch_size)

            mix = np.clip(epoch/self.epochs, 0, 1)
            self.std = (1-mix) * 1 + mix * 0.1

        torch.save([self.encoder.state_dict(), self.actor.state_dict(), self.critic.state_dict()], "model.pth")
        

def main():
    trainer = Trainer()

    trainer.train()

    trainer.env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()