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

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.green_cube = RigidObject(self.cfg.green_cube)
        self.red_cube = RigidObject(self.cfg.red_cube)
        self.end_effector = FrameTransformer(self.cfg.end_effector)
        self.gripper_contact = ContactSensor(self.cfg.gripper_contact)
        self.jaw_contact = ContactSensor(self.cfg.jaw_contact)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["green_cube"] = self.green_cube
        self.scene.rigid_objects["red_cube"] = self.red_cube
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

        joint_pos = self.robot.data.joint_pos.clone()
        previous_joint_pos = self._previous_joint_pos.clone()              
        previous_actions = self._previous_actions.clone()

        if self.cfg.is_training:
            cube_pos_b += torch.randn_like(cube_pos_b) * 0.01
            cube_quat_b += torch.randn_like(cube_quat_b) * 0.01
            joint_pos += torch.randn_like(joint_pos) * 0.01
            previous_joint_pos += torch.randn_like(previous_joint_pos) * 0.01
            previous_actions += torch.randn_like(previous_actions) * 0.01             

        obs = torch.cat([
            cube_pos_b,#3
            cube_quat_b,#4
            joint_pos, #6
            previous_joint_pos, #6
            previous_actions, #6
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

        end_effector_pos_w = self.end_effector.data.target_pos_w[:, 0, :]
        end_effector_quat_w = self.end_effector.data.target_quat_w[:, 0, :]

        gripper_joint_pos_from = self._previous_joint_pos[:, -1]
        gripper_joint_pos_to = self.robot.data.joint_pos[:, -1]

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
            gripper_joint_pos_from,
            gripper_joint_pos_to,
            is_gripper_jaw_touch_cube,
            not self.cfg.is_training
        )
        
        penlty = self._get_action_rate_reward() * (-0.1) + self._joint_velocity_penalty() * (-0.05)
        
        reward = motion_reward * 5 + penlty
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
    
    def sample_in_ellipse_torch(self, n:int, a:float, b:float, angle:float=0.0,
                            center=(0.0, 0.0), device="cpu", dtype=torch.float32):
        """
        Uniform samples inside an ellipse:
        ((x-cx, y-cy) rotated by -angle) satisfies (x/a)^2 + (y/b)^2 <= 1
        a,b > 0 are the semi-axes, angle in radians (counter-clockwise).
        """
        u = torch.rand(n, device=device, dtype=dtype)
        v = torch.rand(n, device=device, dtype=dtype)
        r = torch.sqrt(u)                         # area-uniform radius
        theta = 2 * torch.pi * v
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        # scale to ellipse axes
        x, y = a * x, b * y

        # rotate by 'angle'
        c, s = torch.cos(torch.tensor(angle, device=device, dtype=dtype)), torch.sin(torch.tensor(angle, device=device, dtype=dtype))
        xr = c * x - s * y
        yr = s * x + c * y

        # translate
        cx, cy = center
        return xr + cx, yr + cy
    
    def _sample_pair_in_ellipse_far_enough(self, n:int, a:float, b:float,
                                       min_sep:float, angle:float=0.0,
                                       max_tries:int=32):
        """
        Returns four tensors: gx, gy, rx, ry (shape [n]) with both points in the
        same ellipse and ||g - r|| >= min_sep (as much as possible).
        """
        # quick feasibility check (max possible separation ~ 2*max(a,b))
        assert min_sep <= 2*max(a, b) + 1e-6, "min_sep too large for ellipse."

        gx, gy = self.sample_in_ellipse_torch(n, a, b, angle=angle, device=self.device)
        rx, ry = self.sample_in_ellipse_torch(n, a, b, angle=angle, device=self.device)
        g = torch.stack([gx, gy], dim=-1)   # [n,2]
        r = torch.stack([rx, ry], dim=-1)   # [n,2]

        min_sep2 = min_sep * min_sep
        tries = 0
        # resample red where too close to green
        while True:
            d2 = (r - g).pow(2).sum(dim=-1)
            bad = d2 < min_sep2
            if not bad.any() or tries >= max_tries:
                break
            k = int(bad.sum())
            rx_new, ry_new = self.sample_in_ellipse_torch(k, a, b, angle=angle, device=self.device)
            r[bad] = torch.stack([rx_new, ry_new], dim=-1)
            tries += 1

        # Fallback: push remaining bad cases along the ray from green to red,
        # and project to ellipse boundary if needed.
        if bad.any():
            delta = r[bad] - g[bad]
            zero = delta.norm(dim=-1) < 1e-9
            if zero.any():
                ang = 2*torch.pi*torch.rand(int(zero.sum()), device=self.device)
                delta[zero] = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
            unit = delta / delta.norm(dim=-1, keepdim=True)
            r_try = g[bad] + unit * min_sep

            # Project to ellipse if outside: scale by 1/sqrt(Q(x)) where Q is ellipse quad. form.
            c = torch.cos(torch.tensor(angle, device=self.device))
            s = torch.sin(torch.tensor(angle, device=self.device))
            xprime =  c*r_try[:, 0] + s*r_try[:, 1]
            yprime = -s*r_try[:, 0] + c*r_try[:, 1]
            val = (xprime / a)**2 + (yprime / b)**2
            outside = val > 1.0
            if outside.any():
                scale = (1.0 / torch.sqrt(val[outside])).unsqueeze(-1)
                r_try[outside] = r_try[outside] * scale
            r[bad] = r_try

        return g[:, 0], g[:, 1], r[:, 0], r[:, 1]
    
    def sample_cube_state(self, env_ids: torch.Tensor | None):
        sample_num = len(env_ids)
        
        green_cube_root_state = self.green_cube.data.default_root_state[env_ids]
        green_cube_root_state[:, :3] += self.terrain.env_origins[env_ids]

        red_cube_root_state = self.red_cube.data.default_root_state[env_ids]
        red_cube_root_state[:, :3] += self.terrain.env_origins[env_ids]
        
        a, b, angle = 0.05, 0.15, 0.0
        gx, gy, rx, ry = self._sample_pair_in_ellipse_far_enough(
            sample_num, a, b, 0.08, angle=angle
        )

        green_cube_root_state[:, 0] += gx
        green_cube_root_state[:, 1] += gy

        red_cube_root_state[:, 0] += rx
        red_cube_root_state[:, 1] += ry

        euler_x = torch.empty(sample_num, device=self.device).fill_(0.0)
        euler_y = torch.empty(sample_num, device=self.device).fill_(0.0)
        euler_z = torch.empty(sample_num, device=self.device).uniform_(-torch.pi/4, torch.pi/4)

        quat = quat_from_euler_xyz(euler_x, euler_y, euler_z)

        green_cube_root_state[:, 3:7] = quat

        self.green_cube.write_root_state_to_sim(green_cube_root_state, env_ids)
        self.red_cube.write_root_state_to_sim(red_cube_root_state, env_ids)

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
        