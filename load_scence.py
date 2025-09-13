from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})  # start the simulation app, with GUI open

import sys

import torch
import json
from uuid import uuid4
import time

from pxr import Usd, UsdPhysics
from omni.usd import get_context
from omni.physx.scripts import physicsUtils
from pxr import PhysxSchema

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, RigidPrim, XFormPrim
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from omni.isaac.sensor import CameraView
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationActions
from isaacsim.core.cloner import GridCloner

import cv2
from PIL import Image
from model.actor_critic import EncoderNet, StochasticDDPGActor

from env.utils import map_to_yaw_rep
from contorller.state_policy_controller import PolicyController
from contorller.load_config import get_articulation_props, get_physics_properties, get_robot_joint_properties, parse_env_config

@torch.jit.script
def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as Euler angles in radians to Quaternions.

    Note:
        The euler angles are assumed in XYZ convention.

    Args:
        roll: Rotation around x-axis (in radians). Shape is (N,).
        pitch: Rotation around y-axis (in radians). Shape is (N,).
        yaw: Rotation around z-axis (in radians). Shape is (N,).

    Returns:
        The quaternion in (w, x, y, z). Shape is (N, 4).
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    # compute quaternion
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qw, qx, qy, qz], dim=-1)

def sample_in_annular_sector(n,
                             r_min: float, r_max: float,
                             theta_min: float, theta_max: float,
                             center=(0.0, 0.0),
                             degrees=False,
                             device=None,
                             dtype=torch.float32):
    
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

class Controller:
    def __init__(self, robots, cubes, cameras):
        self.robots = robots
        self.cubes = cubes
        self.cameras = cameras
        self.num_envs = len(self.robots.prims)

        self.device = torch.device("cuda:0")
        self.encoder = EncoderNet(6+6+3+4+3+4, [256, 256, 256]).to(self.device)
        self.actor = StochasticDDPGActor(self.encoder.dim, [256, 256], 6).to(self.device)

        encoder_params, actor_params, _ = torch.load("state_model.pth")
        self.encoder.load_state_dict(encoder_params)
        self.actor.load_state_dict(actor_params)

        self.encoder.eval()
        self.actor.eval()

        self._action_scale = 0.15

    def set_props(self):
        # joint PD gains (use float32, shapes [num_envs, dof])
        kp = torch.full((self.num_envs, 6), 17.8, dtype=torch.float32)
        kd = torch.full((self.num_envs, 6), 0.6,  dtype=torch.float32)
        self.robots.set_gains(kps=kp, kds=kd)

        # IMPORTANT: use plain Python ints / lists, not NumPy scalars
        pos_iters = [64] * self.num_envs
        vel_iters = [64] * self.num_envs
        self.robots.set_solver_position_iteration_counts(pos_iters)
        self.robots.set_solver_velocity_iteration_counts(vel_iters)

    def get_frame(self):
        frame = self.cameras.get_rgb_tiled()

        return frame
        
    def get_cube_state(self):
        cube_state = self.cubes.get_current_dynamic_state()
        cube_pos = cube_state.positions
        cube_quat = cube_state.orientations

        return cube_pos, cube_quat
    
    def get_state_obs(self):
        cube_pos, cube_quat = self.get_cube_state()
        joint_pos = self.robots.get_joint_positions()

        pre_cube_pos = self.pre_cube_pos.clone()
        pre_cube_quat = self.pre_cube_quat.clone()
        pre_joint_pos = self.pre_joint_pos.clone()

        self.pre_cube_pos = cube_pos.clone()
        self.pre_cube_quat = cube_quat.clone()
        self.pre_joint_pos = joint_pos.clone()

        cube_pos -= self.robots_position
        cube_quat = map_to_yaw_rep(cube_quat, xyzw=False)

        pre_cube_pos -= self.robots_position
        pre_cube_quat = map_to_yaw_rep(pre_cube_quat, xyzw=False)

        return torch.cat([cube_pos, cube_quat, joint_pos, pre_cube_pos, pre_cube_quat, pre_joint_pos], dim=1)

    def initialize(self):
        self.robots.initialize()
        self.set_props()
        self.robots_position = self.robots.get_default_state().positions
        self.cameras.initialize()

    def reset(self):
        self.robots.post_reset()
        cube_offset_x, cube_offset_y = sample_in_annular_sector(self.num_envs, 0.225, 0.325,
                                                                -np.pi/3, np.pi/3, device=self.device)
        cube_pos = self.robots_position.clone()
        cube_pos[:, 0] += cube_offset_x
        cube_pos[:, 1] += cube_offset_y

        euler_x = torch.empty(self.num_envs, device=self.device).fill_(0.0)
        euler_y = torch.empty(self.num_envs, device=self.device).fill_(0.0)
        euler_z = torch.empty(self.num_envs, device=self.device).uniform_(-torch.pi/4, torch.pi/4)

        cube_quat = quat_from_euler_xyz(euler_x, euler_y, euler_z)

        self.cubes.set_world_poses(cube_pos, cube_quat)

        self.target_joint_pos = self.robots.get_joint_positions().clone()
        self.pre_joint_pos = self.robots.get_joint_positions().clone()
        cube_pos, cube_quat = self.get_cube_state()
        self.pre_cube_pos = cube_pos.clone()
        self.pre_cube_quat = cube_quat.clone()
        self.pre_frame = self.get_frame().copy()

    def _compute_action(self, obs: np.ndarray, deterministic:bool=True) -> np.ndarray:
        """
        Computes the action from the observation using the loaded policy.

        Args:
            obs (np.ndarray): The observation.

        Returns:
            np.ndarray: The action.
        """
        with torch.no_grad():
            obs = obs.float().to(self.device)
            feature = self.encoder(obs)
            step = self.actor(feature, std=1.0)
            if deterministic:
                action = step.mean
            else:    
                action = step.pi.rsample()
            action = action
        return action
    
    def forward(self, obs, deterministic):
        self.action = self._compute_action(obs, deterministic)
        self.target_joint_pos = self.action * self._action_scale + self.robots.get_joint_positions()
        
        action = ArticulationActions(joint_positions=self.target_joint_pos)
        self.robots.apply_action(action)


class Writer:
    @staticmethod
    def save_data(trajectory_id: str, step: int, state_obs, frame_obs):
        pre_step = max(step, step-1)
        data = {
            "states": state_obs.cpu().tolist(),
            "frames": [f"replays/img/{trajectory_id}_{step}.png", f"replays/img/{trajectory_id}_{pre_step}.png"]
        }

        frames = Image.fromarray(frame_obs)
        frames.save(f"replays/img/{trajectory_id}_{step}.jpg")
        #cv2.imwrite(f"replays/img/{trajectory_id}_{step}.jpg", frame_obs, [cv2.IMWRITE_JPEG_QUALITY, 95])

        with open(f"replays/json/{trajectory_id}_{step}.json", "w") as f:
            json.dump(data, f)

# preparing the scene
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

open_stage(usd_path="scene.usd")

'''
stage = get_context().get_stage()
scene_path = "PhysicsScene"
if not stage.GetPrimAtPath(scene_path):
    physicsUtils.add_physics_scene(stage, scene_path)

scene_prim = stage.GetPrimAtPath(scene_path)
physx_api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)

physx_api.CreateEnableGPUDynamicsAttr().Set(True)
physx_api.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(4096)

physx_api.CreateGpuTotalAggregatePairsCapacityAttr().Set(8192)
physx_api.CreateGpuFoundLostPairsCapacityAttr().Set(1 << 22)
'''

my_world = World(physics_dt=1/120, rendering_dt=1/120, stage_units_in_meters=1.0,
                 backend="torch", device="cuda:0")

num_envs = 16
cloner = GridCloner(spacing=5.0)

clone_paths = cloner.generate_paths("/World/env", num_envs)
cloner.clone(
    source_prim_path="/World/env_0",
    prim_paths=clone_paths,
    copy_from_source=True,
)


robots = Articulation(prim_paths_expr=["/World/env_.*/so101"], name="robot")
cubes = RigidPrim(prim_paths_expr=["/World/env_.*/Cube/Cube"], name="cube")
color_cameras = CameraView(
    prim_paths_expr=["/World/env_.*/Realsense/RSD455/Camera_OmniVision_OV9782_Color"],
    camera_resolution=(640, 480)
)

controller = Controller(robots, cubes, color_cameras)

my_world.reset()
controller.initialize()


for _ in range(60):
    my_world.step(render=True)

start = time.perf_counter()
for _ in range(5):
    trajectory_id = str(uuid4())
    controller.reset()

    for _ in range(12):
        my_world.step(render=True)
    
    for i in range(50):
        state_obs = controller.get_state_obs()
        frame_obs = controller.get_frame()
        Writer.save_data(trajectory_id, i, state_obs, frame_obs)
        controller.forward(state_obs, True)
        for _ in range(4):
            my_world.step(render=True)

end = time.perf_counter()
print(f"took {end - start:.4f} seconds")


simulation_app.close()
