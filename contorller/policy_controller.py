import io
from typing import Optional

import carb
import numpy as np
import omni
import torch
from isaacsim.core.api.controllers.base_controller import BaseController
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from omni.physx import get_physx_simulation_interface
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction

from model.actor_critic import EncoderNet, StochasticDDPGActor
from env.utils import map_to_yaw_rep

from .load_config import get_articulation_props, get_physics_properties, get_robot_joint_properties, parse_env_config


def sample_in_annular_sector(
        r_min: float, r_max: float,
        theta_min: float, theta_max: float,
        center=(0.0, 0.0),
        degrees=False,
        dtype=torch.float32
    ):

    # angles
    if degrees:
        tmin = torch.deg2rad(torch.tensor(theta_min, dtype=dtype))
        tmax = torch.deg2rad(torch.tensor(theta_max, dtype=dtype))
    else:
        tmin = torch.tensor(theta_min, dtype=dtype)
        tmax = torch.tensor(theta_max, dtype=dtype)

    theta = torch.empty(1, dtype=dtype).uniform_(float(tmin), float(tmax))

    # radii: sqrt trick for area-uniformity
    u = torch.empty(1, dtype=dtype).uniform_(0.0, 1.0)
    r = torch.sqrt(u * (r_max**2 - r_min**2) + r_min**2)

    cx = torch.as_tensor(center[0], dtype=dtype)
    cy = torch.as_tensor(center[1], dtype=dtype)
    x = cx + r * torch.cos(theta)
    y = cy + r * torch.sin(theta)
    return x.numpy(), y.numpy()

class PolicyController(BaseController):
    """
    A controller that loads and executes a policy from a file.

    Args:
        name (str): The name of the controller.
        prim_path (str): The path to the prim in the stage.
        root_path (Optional[str], None): The path to the articulation root of the robot
        usd_path (Optional[str], optional): The path to the USD file. Defaults to None.
        position (Optional[np.ndarray], optional): The initial position of the robot. Defaults to None.
        orientation (Optional[np.ndarray], optional): The initial orientation of the robot. Defaults to None.

    Attributes:
        robot (SingleArticulation): The robot articulation.
    """

    def __init__(
        self,
        robot,
        cube
    ) -> None:
        
        self.robot = robot
        
        self.load_policy()
        self.load_config()

        self.cube = cube
        self._action_scale = 0.25
        self._policy_counter = 0

    def load_policy(self):
        self.device = torch.device("cuda:0")
        self.encoder = EncoderNet(6+6+3+4+3+4, [256, 256, 256]).to(self.device)
        self.actor = StochasticDDPGActor(self.encoder.dim, [256, 256], 6).to(self.device)

        encoder_params, actor_params, _ = torch.load("model.pth")
        self.encoder.load_state_dict(encoder_params)
        self.actor.load_state_dict(actor_params)

        self.encoder.eval()
        self.actor.eval()

    def load_config(self):
        self.policy_env_params = parse_env_config("/home/xdang/Desktop/SoArm/env.yaml")
        self._decimation, self._dt, self.render_interval = get_physics_properties(self.policy_env_params)

    def initialize(
        self,
        physics_sim_view: omni.physics.tensors.SimulationView = None,
        effort_modes: str = "force",
        control_mode: str = "position",
        set_gains: bool = True,
        set_limits: bool = True,
        set_articulation_props: bool = True,
    ) -> None:
        """
        Initializes the robot and sets up the controller.

        Args:
            physics_sim_view (optional): The physics simulation view.
            effort_modes (str, optional): The effort modes. Defaults to "force".
            control_mode (str, optional): The control mode. Defaults to "position".
            set_gains (bool, optional): Whether to set the joint gains. Defaults to True.
            set_limits (bool, optional): Whether to set the limits. Defaults to True.
            set_articulation_props (bool, optional): Whether to set the articulation properties. Defaults to True.
        """
        self.robot.initialize(physics_sim_view=physics_sim_view)
        self.robot.get_articulation_controller().set_effort_modes(effort_modes)

        # TODO: Must flush when FSD is enabled.
        # Otherwise the delayed FSD handling next frame will overwrite set_max_efforts below
        get_physx_simulation_interface().flush_changes()

        self.robot.get_articulation_controller().switch_control_mode(control_mode)
        max_effort, max_vel, stiffness, damping, self.default_pos, self.default_vel = get_robot_joint_properties(
            self.policy_env_params, self.robot.dof_names
        )
        if set_gains:
            self.robot._articulation_view.set_gains(stiffness, damping)
        if set_limits:
            self.robot._articulation_view.set_max_efforts(max_effort)

            # TODO: Must flush when FSD is enabled.
            # Otherwise the delayed FSD handling next frame will overwrite set_max_efforts below
            get_physx_simulation_interface().flush_changes()

            self.robot._articulation_view.set_max_joint_velocities(max_vel)
        if set_articulation_props:
            self._set_articulation_props()

    def _set_articulation_props(self) -> None:
        """
        Sets the articulation root properties from the policy environment parameters.
        """
        articulation_prop = get_articulation_props(self.policy_env_params)

        solver_position_iteration_count = articulation_prop.get("solver_position_iteration_count")
        solver_velocity_iteration_count = articulation_prop.get("solver_velocity_iteration_count")
        stabilization_threshold = articulation_prop.get("stabilization_threshold")
        enabled_self_collisions = articulation_prop.get("enabled_self_collisions")
        sleep_threshold = articulation_prop.get("sleep_threshold")

        if solver_position_iteration_count not in [None, float("inf")]:
            self.robot.set_solver_position_iteration_count(solver_position_iteration_count)
        if solver_velocity_iteration_count not in [None, float("inf")]:
            self.robot.set_solver_velocity_iteration_count(solver_velocity_iteration_count)
        if stabilization_threshold not in [None, float("inf")]:
            self.robot.set_stabilization_threshold(stabilization_threshold)
        if isinstance(enabled_self_collisions, bool):
            self.robot.set_enabled_self_collisions(enabled_self_collisions)
        if sleep_threshold not in [None, float("inf")]:
            self.robot.set_sleep_threshold(sleep_threshold)

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
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
            action = step.mean.cpu().detach().view(-1).numpy()
        return action

    def _compute_observation(self) -> NotImplementedError:
        """
        Computes the observation. Not implemented.
        """
        state = self.cube.get_current_dynamic_state()
        cube_pos = state.position
        cube_quat = state.orientation
        joint_pos = self.robot.get_joint_positions()

        cube_pos = torch.from_numpy(cube_pos)
        cube_quat = torch.from_numpy(cube_quat)

        cube_quat = map_to_yaw_rep(cube_quat)

        pre_cube_pos = self.pre_cube_state.position
        pre_cube_quat = self.pre_cube_state.orientation

        pre_cube_pos = torch.from_numpy(pre_cube_pos)
        pre_cube_quat = torch.from_numpy(pre_cube_quat)

        pre_cube_quat = map_to_yaw_rep(pre_cube_quat)

        joint_pos = torch.from_numpy(joint_pos)
        pre_joint_pos = torch.from_numpy(self.pre_joint_pos)

        obs = torch.cat([cube_pos, cube_quat, joint_pos, pre_cube_pos, pre_cube_quat, pre_joint_pos])

        self.pre_joint_pos = self.robot.get_joint_positions().copy()
        self.pre_cube_state = self.cube.get_current_dynamic_state()

        return obs

    def forward(self, dt) -> NotImplementedError:
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation()
            self.action = self._compute_action(obs)
            self.target_joint_pos = self.action * self._action_scale + self.robot.get_joint_positions()
        
        action = ArticulationAction(joint_positions=self.target_joint_pos)
        self.robot.apply_action(action)

        self._policy_counter += 1

    def post_reset(self) -> None:
        """
        Called after the controller is reset.
        """
        self.robot.post_reset()

        cube_pos = np.array([0.0, 0, 0.019])
        offset_x, offset_y = sample_in_annular_sector(0.225, 0.325, -torch.pi/3, torch.pi/3)
        cube_pos[0] += offset_x
        cube_pos[1] += offset_y

        euler = np.array([0.0, 0.0, np.random.uniform(-np.pi/4, np.pi/4)], dtype=np.float32)
        cube_quat = euler_angles_to_quat(euler)

        self.cube.set_world_pose(cube_pos, cube_quat)

        self.pre_joint_pos = self.robot.get_joint_positions().copy()
        self.pre_cube_state = self.cube.get_current_dynamic_state()

