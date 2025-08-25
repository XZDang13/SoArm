import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import FrameTransformer
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import (subtract_frame_transforms, quat_from_euler_xyz,
                                 quat_error_magnitude, quat_mul,
                                 euler_xyz_from_quat)

from .stack_cfg import STACK_TASK_CFG

class StackTask(DirectRLEnv):
    cfg:STACK_TASK_CFG

    def __init__(self, cfg, render_mode = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._previous_joint_pos = self.robot.data.joint_pos.clone()

        self.visual_marker_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.visual_marker_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.visual_marker_quat[:, 0] = 1.0

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.cube = RigidObject(self.cfg.cube)
        self.end_effector = FrameTransformer(self.cfg.end_effector)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["cube"] = self.cube
        self.scene.sensors["end_effector"] = self.end_effector

        self.visual_marker = VisualizationMarkers(self.cfg.gripper_marker)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing

        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._joint_target_pos = self.cfg.action_scale * self._actions + self.robot.data.joint_pos

    def _apply_action(self):
        self.robot.set_joint_position_target(self._joint_target_pos)

    def _get_observations(self):
        self._previous_actions = self._actions.clone()

        cube_pos_w = self.cube.data.root_state_w[:, :3]
        cube_quat_w = self.cube.data.root_state_w[:, 3:7]
        
        cube_pos_b, cube_quat_b = subtract_frame_transforms(
            self.robot.data.root_state_w[:, :3], 
            self.robot.data.root_state_w[:, 3:7], 
            cube_pos_w, 
            cube_quat_w
        )

        joint_pos = self.robot.data.joint_pos              
        previous_actions = self._previous_actions             

        obs = torch.cat([
            cube_pos_b,#3
            cube_quat_b,#4
            joint_pos, #6
            self._previous_joint_pos, #6
            previous_actions, #6
        ], dim=-1)

        #end_effector_pos = self.end_effector.data.target_pos_source[:, 0, :]
        #end_effector_quat = self.end_effector.data.target_quat_source[:, 0, :]

        self._previous_joint_pos = self.robot.data.joint_pos.clone()

        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        cube_pos_w = self.cube.data.root_state_w[:, :3]
        cube_pos_w[:, 2] += 0.1

        cube_quat_w = self.cube.data.root_state_w[:, 3:7]


        end_effector_pos_w = self.end_effector.data.target_pos_w[:, 0, :]
        end_effector_quat_w = self.end_effector.data.target_quat_w[:, 0, :]

        distance_error = -torch.norm(cube_pos_w - end_effector_pos_w, p=2, dim=1)
        quat_error = -quat_error_magnitude(cube_quat_w, end_effector_quat_w)
        reward = distance_error * 5 + quat_error

        return reward

    def _get_action_rate_reward(self) -> torch.Tensor:
        return torch.sum((self._actions - self._previous_actions) ** 2, dim=1)
    
    def _joint_velocity_penalty(self) -> torch.Tensor:
        return torch.norm(self.robot.data.joint_vel, dim=1)

    def _difference_to_default_reward(self) -> torch.Tensor:
        return torch.sum((self.robot.data.joint_pos - self.robot.data.default_joint_pos) ** 2, dim=1)
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        return False, time_out
    
    def sample_cube_state(self, env_ids: torch.Tensor | None):
        sample_num = len(env_ids)
        
        root_state = self.cube.data.default_root_state[env_ids]
        root_state[:, :3] += self.terrain.env_origins[env_ids]
        
        offset_x = torch.empty(sample_num, device=self.device).uniform_(-0.15, 0.)
        offset_y = torch.empty(sample_num, device=self.device).uniform_(-0.2, 0.2)

        root_state[:, 0] += offset_x
        root_state[:, 1] += offset_y

        euler_x = torch.empty(sample_num, device=self.device).fill_(0.0)
        euler_y = torch.empty(sample_num, device=self.device).fill_(0.0)
        euler_z = torch.empty(sample_num, device=self.device).uniform_(-torch.pi/4, torch.pi/4)

        quat = quat_from_euler_xyz(euler_x, euler_y, euler_z)

        root_state[:, 3:7] = quat

        self.cube.write_root_state_to_sim(root_state, env_ids)

        self.visual_marker_pos[env_ids] = root_state[:, :3] + torch.tensor([0.0, 0.0, 0.1], device=self.device)
        self.visual_marker_quat[env_ids] = root_state[:, 3:7]
        self.visual_marker.visualize(self.visual_marker_pos, self.visual_marker_quat)

    def reset_robot(self, env_ids: torch.Tensor | None):
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
       
        self.reset_robot(env_ids)
        self.sample_cube_state(env_ids)

        self._previous_joint_pos = self.robot.data.joint_pos.clone()
        