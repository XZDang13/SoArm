import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import FrameTransformer, ContactSensor
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import (subtract_frame_transforms, quat_from_euler_xyz,
                                 quat_error_magnitude, quat_mul,
                                 euler_xyz_from_quat)

from .stack_cfg import STACK_TASK_CFG
from .utils import map_to_yaw_rep
from reward.stack_task import StackTaskReward

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

        self.end_effector_pre_state = torch.zeros(self.num_envs, 7, device=self.device)
        self.cube_pre_state = torch.zeros(self.num_envs, 7, device=self.device)

        self.goal_pos = torch.as_tensor([0.23, 0.0, 0.08], device=self.device)
        self.goal_quat = torch.as_tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.green_cube = RigidObject(self.cfg.green_cube)
        #self.red_cube = RigidObject(self.cfg.red_cube)
        self.end_effector = FrameTransformer(self.cfg.end_effector)
        self.gripper_contact = ContactSensor(self.cfg.gripper_contact)
        self.jaw_contact = ContactSensor(self.cfg.jaw_contact)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["green_cube"] = self.green_cube
        #self.scene.rigid_objects["red_cube"] = self.red_cube
        self.scene.sensors["end_effector"] = self.end_effector
        self.scene.sensors["gripper_contact"] = self.gripper_contact
        self.scene.sensors["jaw_contact"] = self.jaw_contact

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

        cube_pos_w = self.green_cube.data.root_state_w[:, :3]
        cube_quat_w = self.green_cube.data.root_state_w[:, 3:7]
        
        cube_pos_b, cube_quat_b = subtract_frame_transforms(
            self.robot.data.root_state_w[:, :3], 
            self.robot.data.root_state_w[:, 3:7], 
            cube_pos_w, 
            cube_quat_w
        )

        cube_quat_b = map_to_yaw_rep(cube_quat_b, xyzw=False)

        pre_cube_pos_w = self.cube_pre_state[:, :3]
        pre_cube_quat_b = self.cube_pre_state[:, 3:7]

        pre_cube_pos_b, pre_cube_quat_b = subtract_frame_transforms(
            self.robot.data.root_state_w[:, :3], 
            self.robot.data.root_state_w[:, 3:7], 
            pre_cube_pos_w, 
            pre_cube_quat_b
        )

        pre_cube_quat_b = map_to_yaw_rep(pre_cube_quat_b, xyzw=False)

        joint_pos = self.robot.data.joint_pos.clone()
        previous_joint_pos = self._previous_joint_pos.clone()

        if self.cfg.is_training:
            cube_pos_noise = torch.empty_like(cube_pos_b).uniform_(-0.01, 0.01)
            cube_quat_noise = torch.empty_like(cube_quat_b).uniform_(-0.1, 0.1)
            pre_cube_pos_noise = torch.empty_like(pre_cube_pos_b).uniform_(-0.01, 0.01)
            pre_cube_quat_noise = torch.empty_like(pre_cube_quat_b).uniform_(-0.1, 0.1)
            joint_pos_noise = torch.empty_like(joint_pos).uniform_(-0.02, 0.02)
            previous_joint_pos_noise = torch.empty_like(previous_joint_pos).uniform_(-0.02, 0.02)

            cube_pos_b += cube_pos_noise
            cube_quat_b += cube_quat_noise
            pre_cube_pos_b += pre_cube_pos_noise
            pre_cube_quat_b += pre_cube_quat_noise
            joint_pos += joint_pos_noise
            previous_joint_pos += previous_joint_pos_noise          

        obs = torch.cat([
            cube_pos_b,#3
            cube_quat_b,#4
            joint_pos, #6
            pre_cube_pos_b,
            pre_cube_quat_b,
            previous_joint_pos
        ], dim=-1)

        #end_effector_pos = self.end_effector.data.target_pos_source[:, 0, :]
        #end_effector_quat = self.end_effector.data.target_quat_source[:, 0, :]

        self._previous_joint_pos = self.robot.data.joint_pos.clone()
        self.end_effector_pre_state[:, :3] = self.end_effector.data.target_pos_w[:, 0, :].clone()
        self.end_effector_pre_state[:, 3:7] = self.end_effector.data.target_quat_w[:, 0, :].clone()
        self.cube_pre_state[:, :] = self.green_cube.data.root_state_w[:, :7].clone()

        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        cube_pos_w = self.green_cube.data.root_state_w[:, :3]
        #cube_pos_w[:, 2] += 0.1

        cube_quat_w = self.green_cube.data.root_state_w[:, 3:7]
        cube_quat_w = map_to_yaw_rep(cube_quat_w, xyzw=False)

        end_effector_pos_w = self.end_effector.data.target_pos_w[:, 0, :]
        end_effector_quat_w = self.end_effector.data.target_quat_w[:, 0, :]

        gripper_joint_pos_from = self._previous_joint_pos[:, -1]
        gripper_joint_pos_to = self.robot.data.joint_pos[:, -1]

        goal_pos = self.terrain.env_origins + self.goal_pos
        goal_quat = self.goal_quat.expand_as(end_effector_quat_w)

        gripper_contact_force = self.gripper_contact.data.force_matrix_w[:, 0, 0, :]
        jaw_contact_force = self.jaw_contact.data.force_matrix_w[:, 0, 0, :]
        is_gripper_touch_cube = torch.linalg.norm(gripper_contact_force, dim=-1) > 1.0
        is_jaw_touch_cube = torch.linalg.norm(jaw_contact_force, dim=-1) > 1.0
        is_gripper_jaw_touch_cube = is_gripper_touch_cube & is_jaw_touch_cube
        #print(gripper_contact_force)
        #print(torch.linalg.norm(gripper_contact_force, dim=-1))
        #print("----------------")



        motion_reward = StackTaskReward.compute_reward(
            self.end_effector_pre_state[:, :3],
            self.end_effector_pre_state[:, 3:7],
            end_effector_pos_w,
            end_effector_quat_w,
            cube_pos_w,
            cube_quat_w,
            goal_pos,
            goal_quat,
            gripper_joint_pos_from,
            gripper_joint_pos_to,
            is_gripper_jaw_touch_cube,
            not self.cfg.is_training
        )
        
        penlty = self._joint_velocity_penalty() + self._get_action_rate_reward() + self._difference_to_default_reward()
        
        reward = motion_reward * 2.5 + penlty * (-0.01)
        #print(motion_reward)
        #print("-----------------")

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
    
    def sample_in_annular_sector(self, n,
                                   r_min: float, r_max: float,
                                   theta_min: float, theta_max: float,
                                   center=(0.0, 0.0),
                                   degrees=False,
                                   device=None, dtype=torch.float32):
        if device is None:
            device = torch.device('cpu')

        # angles
        if degrees:
            tmin = torch.deg2rad(torch.tensor(theta_min, device=device, dtype=dtype))
            tmax = torch.deg2rad(torch.tensor(theta_max, device=device, dtype=dtype))
        else:
            tmin = torch.tensor(theta_min, device=device, dtype=dtype)
            tmax = torch.tensor(theta_max, device=device, dtype=dtype)

        theta = torch.empty(n, device=device, dtype=dtype).uniform_(float(tmin), float(tmax))

        # radii: sqrt trick for area-uniformity
        u = torch.empty(n, device=device, dtype=dtype).uniform_(0.0, 1.0)
        r = torch.sqrt(u * (r_max**2 - r_min**2) + r_min**2)

        cx = torch.as_tensor(center[0], device=device, dtype=dtype)
        cy = torch.as_tensor(center[1], device=device, dtype=dtype)
        x = cx + r * torch.cos(theta)
        y = cy + r * torch.sin(theta)
        return x, y
    
    def sample_cube_state(self, env_ids: torch.Tensor | None):
        sample_num = len(env_ids)
        
        green_cube_root_state = self.green_cube.data.default_root_state[env_ids]
        green_cube_root_state[:, :3] += self.terrain.env_origins[env_ids]

        offset_x, offset_y = self.sample_in_annular_sector(sample_num, 0.225, 0.325,
                                                           -torch.pi/3, torch.pi/3, device=self.device)
        
        green_cube_root_state[:, 0] += offset_x
        green_cube_root_state[:, 1] += offset_y


        euler_x = torch.empty(sample_num, device=self.device).fill_(0.0)
        euler_y = torch.empty(sample_num, device=self.device).fill_(0.0)
        euler_z = torch.empty(sample_num, device=self.device).uniform_(-torch.pi/4, torch.pi/4)

        quat = quat_from_euler_xyz(euler_x, euler_y, euler_z)

        green_cube_root_state[:, 3:7] = quat

        self.green_cube.write_root_state_to_sim(green_cube_root_state, env_ids)

    def reset_robot(self, env_ids: torch.Tensor | None):
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        #joint_pos[:, 0:3] += torch.empty_like(joint_pos[:, 0:3]).uniform_(-1.0, 1.0)

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs and self.cfg.is_training:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
       
        self.reset_robot(env_ids)
        self.sample_cube_state(env_ids)

        self._previous_joint_pos = self.robot.data.joint_pos.clone()
        self.end_effector_pre_state[env_ids, :3] = self.end_effector.data.target_pos_w[env_ids, 0, :].clone()
        self.end_effector_pre_state[env_ids, 3:7] = self.end_effector.data.target_quat_w[env_ids, 0, :].clone()
        self.cube_pre_state[env_ids, :] = self.green_cube.data.root_state_w[env_ids, :7].clone()

        '''
        cube_pos_w = self.cube.data.root_state_w[env_ids, :3]
        cube_quat_w = self.cube.data.root_state_w[env_ids, 3:7]
        
        cube_pos_b, cube_quat_b = subtract_frame_transforms(
            self.robot.data.root_state_w[env_ids, :3], 
            self.robot.data.root_state_w[env_ids, 3:7], 
            cube_pos_w, 
            cube_quat_w
        )

        state = torch.cat([cube_pos_b, cube_quat_b], dim=-1)
        print(state)
        print("-------------")
        '''
        