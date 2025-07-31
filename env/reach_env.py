import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import combine_frame_transforms, quat_from_euler_xyz

from .reach_cfg import REACH_TASK_CFG

class ReachTask(DirectRLEnv):
    cfg:REACH_TASK_CFG

    def __init__(self, cfg, render_mode = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, 6, device=self.device)

        self.gripper_goal_pos_local = torch.zeros(self.num_envs, 3, device=self.device)
        self.gripper_goal_pos_world = torch.zeros_like(self.gripper_goal_pos_local, device=self.device)
        self.jaw_goal_pos_world = torch.zeros_like(self.gripper_goal_pos_local, device=self.device)

        self.gripper_goal_quat_local = torch.zeros(self.num_envs, 4, device=self.device)
        self.jaw_gola_quat_local = torch.zeros_like(self.gripper_goal_quat_local, device=self.device)
        self.gripper_goal_quat_world = torch.zeros_like(self.gripper_goal_quat_local, device=self.device)
        self.jaw_gola_quat_world = torch.zeros_like(self.gripper_goal_quat_world, device=self.device)

        self.gripper_jaw_offset = torch.as_tensor([0.0234, 0.0, 0.0211], device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self.gripper_marker = VisualizationMarkers(self.cfg.gripper_marker)
        self.jaw_marker = VisualizationMarkers(self.cfg.jaw_marker)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing

        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self.robot.data.default_joint_pos

    def _apply_action(self):
        self.robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self):
        self._previous_actions = self._actions.clone()

        joint_pos = self.robot.data.joint_pos - self.robot.data.default_joint_pos
        joint_vel = self.robot.data.joint_vel               
        previous_actions = self._previous_actions           
        

        obs = torch.cat([
            joint_pos,
            joint_vel,
            previous_actions,
        ], dim=-1)

        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        return self._get_action_rate_reward()


    def _get_action_rate_reward(self) -> torch.Tensor:
        return torch.sum((self._actions - self._previous_actions) ** 2, dim=1)
    
    def _joint_velocity_penalty(self) -> torch.Tensor:
        return torch.norm(self.robot.data.joint_vel, dim=1)

    def _difference_to_default_reward(self) -> torch.Tensor:
        return torch.sum((self.robot.data.joint_pos - self.robot.data.default_joint_pos) ** 2, dim=1)
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        return False, time_out
        
    def sample_pos(self, sample_num:int,
                   x_range:tuple[float, float],
                   y_range:tuple[float, float],
                   z_range:tuple[float, float]) -> torch.Tensor:
        pos_x = torch.empty(sample_num, device=self.device).uniform_(x_range[0], x_range[1])
        pos_y = torch.empty(sample_num, device=self.device).uniform_(y_range[0], y_range[1])
        pos_z = torch.empty(sample_num, device=self.device).uniform_(z_range[0], z_range[1])

        pos = torch.stack([pos_x, pos_y, pos_z], dim=-1)

        return pos
    
    def sample_quat(self, sample_num:int,
                   x_range:tuple[float, float],
                   y_range:tuple[float, float],
                   z_range:tuple[float, float]) -> torch.Tensor:
        
        rot_x = torch.empty(sample_num, device=self.device).uniform_(x_range[0], x_range[1])
        rot_y = torch.empty(sample_num, device=self.device).uniform_(y_range[0], y_range[1])
        rot_z = torch.empty(sample_num, device=self.device).uniform_(z_range[0], z_range[1])

        quat = quat_from_euler_xyz(rot_x, rot_y, rot_z)

        return quat

    def set_visual_markers(self, env_ids: torch.Tensor):
        # Transform to world frame
        root_pos = self.robot.data.root_pos_w[env_ids]
        root_quat = self.robot.data.root_quat_w[env_ids]

        self.gripper_goal_pos_world[env_ids], self.gripper_goal_quat_world[env_ids] = combine_frame_transforms(root_pos, root_quat, self.gripper_goal_pos_local[env_ids], self.gripper_goal_quat_local[env_ids])

        jaw_offset = self.gripper_jaw_offset.expand(len(env_ids), -1)
        self.jaw_goal_pos_world[env_ids], self.jaw_gola_quat_world[env_ids] = combine_frame_transforms(
            self.gripper_goal_pos_world[env_ids],
            self.gripper_goal_quat_world[env_ids],
            jaw_offset,
            self.jaw_gola_quat_local[env_ids]
        )

        self.gripper_marker.visualize(self.gripper_goal_pos_world, self.gripper_goal_quat_world)
        self.jaw_marker.visualize(self.jaw_goal_pos_world, self.jaw_gola_quat_world)

    def sample_end_effector_target(self, env_ids: torch.Tensor):
        self.gripper_goal_pos_local[env_ids] = self.sample_pos(len(env_ids), [0.1, 0.3], [-0.2, 0.2], [0.1, 0.35])

        self.gripper_goal_quat_local[env_ids] = self.sample_quat(len(env_ids), [-torch.pi, torch.pi], [0.0, (torch.pi /2)], [0.0, 0.0])

        self.jaw_gola_quat_local[env_ids] = self.sample_quat(len(env_ids), [0.0, 0.0], [-torch.pi/2, 0.0], [0.0, 0.0])

        self.set_visual_markers(env_ids)

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
       
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.sample_end_effector_target(env_ids)
        