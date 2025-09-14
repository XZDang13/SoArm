from isaacsim import SimulationApp
import os

base_folder = "replays"

# Define subfolders
subfolders = ["json", "img"]

# Loop through and create them if needed
for sub in subfolders:
    path = os.path.join(base_folder, sub)
    os.makedirs(path, exist_ok=True)

simulation_app = SimulationApp({"headless": True})  # start the simulation app, with GUI open

import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import v2
import json
from uuid import uuid4
import time

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


from PIL import Image
from model.actor_critic import EncoderNet, FrameObservationEncoderNet, StochasticDDPGActor

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

def compute_cosine_loss(x, y):
    return 1. - F.cosine_similarity(x, y, dim=-1).mean()

def compute_mse_loss(x, y):
    return F.mse_loss(x, y, reduction="mean")

def untile_image(tiled_img, tile_rows=8, tile_cols=8, tile_h=480, tile_w=640):
    tiles = []
    for i in range(tile_rows):
        for j in range(tile_cols):
            tile = tiled_img[
                i*tile_h:(i+1)*tile_h,
                j*tile_w:(j+1)*tile_w,
                :
            ]
            tiles.append(Image.fromarray(tile))

    return tiles

class Controller:
    def __init__(self, robots, cubes, cameras):
        self.robots = robots
        self.cubes = cubes
        self.cameras = cameras
        self.num_envs = len(self.robots.prims)

        self.device = torch.device("cuda:0")
        self.frame_encoder = FrameObservationEncoderNet(6, 256).to(self.device)
        self.state_encoder = EncoderNet(6+6+3+4+3+4, [256, 256, 256]).to(self.device)
        self.actor = StochasticDDPGActor(self.frame_encoder.dim, [256, 256], 6).to(self.device)

        encoder_params, actor_params, _ = torch.load("state_model.pth")
        self.state_encoder.load_state_dict(encoder_params)
        self.actor.load_state_dict(actor_params)

        self.state_encoder.eval()
        self.actor.eval()

        for param in self.state_encoder.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.frame_encoder.parameters(), lr=1e-3, weight_decay=1e-4)

        self._action_scale = 0.15

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((112, 112)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

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
    
    def process_tile_image(self, frame):
        frame = untile_image(frame)
        frame = torch.stack(self.transform(frame))

        return frame

    def get_camera_obs(self):
        frame = self.get_frame()
        pre_frame = self.pre_frame.copy()

        self.pre_frame = frame.copy()

        frame = self.process_tile_image(frame)
        pre_frame = self.process_tile_image(pre_frame)

        return torch.cat([frame, pre_frame], dim=1)
    
    def get_state_feature(self, obs):
        obs = obs.to(self.device)
        feature = self.state_encoder(obs)

        return feature
    
    def get_frame_feature(self, obs):
        obs = obs.to(self.device)
        feature = self.frame_encoder(obs, True)

        return feature
    
    def update(self, state_feature, frame_feature):
        mse_loss = compute_mse_loss(state_feature, frame_feature)
        cosine_loss = compute_cosine_loss(state_feature, frame_feature)
        loss = 0.5 * mse_loss + 0.5 * cosine_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return cosine_loss.item()

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

    def _compute_action(self, feature, deterministic:bool=True) -> np.ndarray:
        """
        Computes the action from the observation using the loaded policy.

        Args:
            obs (np.ndarray): The observation.

        Returns:
            np.ndarray: The action.
        """
        with torch.no_grad():
            step = self.actor(feature, std=1.0)
            if deterministic:
                action = step.mean
            else:    
                action = step.pi.rsample()
            action = action
        return action
    
    def forward(self, feature, deterministic):
        self.action = self._compute_action(feature, deterministic)
        self.target_joint_pos = self.action * self._action_scale + self.robots.get_joint_positions()
        
        action = ArticulationActions(joint_positions=self.target_joint_pos)
        self.robots.apply_action(action)


# preparing the scene
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

open_stage(usd_path="scene.usd")

my_world = World(physics_dt=1/120, rendering_dt=1/120, stage_units_in_meters=1.0,
                 backend="torch", device="cuda:0")

tile_rows = 8
tile_cols = 8
num_envs = tile_rows * tile_cols

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
for _ in range(50):
    controller.reset()
    cosine_loss_buffer = []
    for _ in range(12):
        my_world.step(render=True)
    
    for i in range(30):
        state_obs = controller.get_state_obs()
        frame_obs = controller.get_camera_obs()

        state_feature = controller.get_state_feature(state_obs)
        frame_feature = controller.get_frame_feature(frame_obs)

        cosine_loss = controller.update(state_feature, frame_feature)
        cosine_loss_buffer.append(cosine_loss)

        controller.forward(state_feature, False)

        for _ in range(4):
            my_world.step(render=True)

    print(np.mean(cosine_loss_buffer))

end = time.perf_counter()
print(f"took {end - start:.4f} seconds")
controller.save()


simulation_app.close()
