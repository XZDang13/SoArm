import io
from typing import Optional

import carb
import numpy as np
import random
import omni
import torch
from torchvision.transforms import v2
from isaacsim.core.utils.types import ArticulationActions
from isaacsim.core.utils.rotations import quat_to_euler_angles
from isaacsim.core.api.materials import OmniPBR
from PIL import Image

from model.actor_critic import EncoderNet, MobileFrameObservationEncoderNet, StochasticDDPGActor, FrameObservationEncoderNet
from env.utils import map_to_yaw_rep

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

def sample_from_range(num_envs, range=None, device=None):
    if device is None:
        device = torch.device('cpu')

    if range is None:
        value = torch.empty(num_envs, device=device).fill_(0.0)
    else:
        value = torch.empty(num_envs, device=device).uniform_(range[0], range[1])

    return value

def sample_pos(num_envs, x_range=None, y_range=None, z_range=None, device=None):
    x = sample_from_range(num_envs, x_range, device)
    y = sample_from_range(num_envs, y_range, device)
    z = sample_from_range(num_envs, z_range, device)

    pos = torch.stack([x, y, z], dim=-1)

    return pos

def sample_quat(num_envs, x_range=None, y_range=None, z_range=None, device=None):
    
    euler_x = sample_from_range(num_envs, x_range, device)
    euler_y = sample_from_range(num_envs, y_range, device)
    euler_z = sample_from_range(num_envs, z_range, device)

    quat = quat_from_euler_xyz(euler_x, euler_y, euler_z)

    return quat

def untile_image(tiled_img, tile_rows=8, tile_cols=8, tile_h=720, tile_w=1280):
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
    def __init__(self, robots, cubes, cameras, tile_rows, tile_cols,
                 state_encoder_path=None, frame_encoder_path=None):
        self.robots = robots
        self.cubes = cubes
        self.cameras = cameras
        self.num_envs = len(self.robots.prims)

        self.device = torch.device("cuda:0")

        self.state_encoder = None
        self.frame_encoder = None
        self.actor = None

        if state_encoder_path is not None:
            self.state_encoder = EncoderNet(6+3+4, [128, 128, 128]).to(self.device)
            self.actor = StochasticDDPGActor(self.state_encoder.dim, [256, 256], 6).to(self.device)

            state_encoder_params, actor_params, _ = torch.load("state_model.pth")
            self.state_encoder.load_state_dict(state_encoder_params)
            self.actor.load_state_dict(actor_params)
            self.state_encoder.eval()
            self.actor.eval()

        if frame_encoder_path is not None:
            self.frame_encoder = FrameObservationEncoderNet(128).to(self.device)
            frame_encoder_params, _, _ = torch.load("frame_model.pth")
            self.frame_encoder.load_state_dict(frame_encoder_params)
            self.frame_encoder.eval()

        self._action_scale = 0.2

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((112, 112)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.tile_rows = tile_rows
        self.tile_cols = tile_cols

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
    
    def get_state(self):
        cube_pos, cube_quat = self.get_cube_state()
        joint_pos = self.robots.get_joint_positions()

        cube_pos -= self.robots_position
        cube_quat = map_to_yaw_rep(cube_quat, xyzw=False)

        current_state = torch.cat([
            cube_pos,#3
            cube_quat,#4
            joint_pos, #6)
        ], dim=-1)

        return current_state
    
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

        current_state = torch.cat([
            cube_pos,#3
            cube_quat,#4
            joint_pos, #6)
        ], dim=-1)

        pre_state = torch.cat([
            pre_cube_pos,
            pre_cube_quat,
            pre_joint_pos
        ], dim=-1)

        obs = torch.stack([current_state, pre_state], 1)

        return obs
    
    def process_tile_image(self, frame):
        frame = untile_image(frame, self.tile_rows, self.tile_cols)
        frame = torch.stack(self.transform(frame))

        return frame

    def get_camera_obs(self):
        frame = self.get_frame()
        pre_frame = self.pre_frame.copy()

        self.pre_frame = frame.copy()

        frame = self.process_tile_image(frame)
        pre_frame = self.process_tile_image(pre_frame)

        return torch.concat([frame, pre_frame], dim=0)
    
    def get_state_feature(self, obs):
        if self.state_encoder is None:
            return None
        
        obs = obs.to(self.device)
        feature = self.state_encoder(obs)

        return feature
    
    def get_frame_feature(self, obs):
        if self.frame_encoder is None:
            return None
        
        obs = obs.to(self.device)
        feature = self.frame_encoder(obs, True)
        feature = feature.view(-1, 256)

        return feature

    def initialize(self):
        self.robots.initialize()
        self.set_props()
        self.robots_position = self.robots.get_default_state().positions
        self.cameras.initialize()

        camera_states = self.cameras.get_default_state()
        self.default_camera_positions = camera_states.positions
        self.default_camera_orientations = camera_states.orientations
        print(self.default_camera_orientations)
        print(quat_to_euler_angles(self.default_camera_orientations[0].numpy()))
        

    def random_camera_state(self):
        pos_offset = sample_pos(self.num_envs, x_range=[-0.01, 0.01],
                         y_range=[-0.01, 0.01], z_range=[-0.01, 0.01],
                         device=self.device)
        
        pos = self.default_camera_positions + pos_offset
        
        quat = sample_quat(self.num_envs, x_range=[-0.0349, 0.0349], y_range=[1.5359, 1.6057], z_range=[-0.0349, 0.0349], device=self.device)

        self.cameras.set_world_poses(positions=pos, orientations=quat)

    def reset(self):
        self.robots.post_reset()
        cube_offset_x, cube_offset_y = sample_in_annular_sector(self.num_envs, 0.225, 0.325,
                                                                -np.pi/4, np.pi/4, device=self.device)
        cube_pos = self.robots_position.clone()
        cube_pos[:, 0] += cube_offset_x
        cube_pos[:, 1] += cube_offset_y
        cube_pos[:, 2] += 0.01

        cube_quat = sample_quat(self.num_envs, z_range=[-torch.pi/4, torch.pi/4], device=self.device)

        self.cubes.set_world_poses(cube_pos, cube_quat)
        velocities = torch.zeros((self.num_envs, 6), device=self.device)
        self.cubes.set_velocities(velocities)

        self.target_joint_pos = self.robots.get_joint_positions().clone()
        self.pre_joint_pos = self.robots.get_joint_positions().clone()
        cube_pos, cube_quat = self.get_cube_state()
        self.pre_cube_pos = cube_pos.clone()
        self.pre_cube_quat = cube_quat.clone()
        self.pre_frame = self.get_frame().copy()

    def _compute_action(self, feature, deterministic:bool=True) -> np.ndarray:
        
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



class RandomLights:
    def __init__(self, dome_light, distant_light, assets_root_path):
        self.dome_light = dome_light
        self.distant_light = distant_light
        self.assets_root_path = assets_root_path
        self.background_assets = [
            "/Isaac/Materials/Textures/Backgrounds/nv_airport.jpg",
            "/Isaac/Materials/Textures/Backgrounds/nv_alaska.jpg",
            "/Isaac/Materials/Textures/Backgrounds/nv_arches.jpg",
            "/Isaac/Materials/Textures/Backgrounds/nv_ariel_narita.jpg",
            "/Isaac/Materials/Textures/Backgrounds/nv_alaska.jpg",
            "/Isaac/Materials/Textures/Backgrounds/nv_australia.jpg",
            "/Isaac/Materials/Textures/Backgrounds/nv_banff_lake_1.jpg",
            "/Isaac/Materials/Textures/Backgrounds/nv_banff_lake_3.jpg",
            "/Isaac/Materials/Textures/Backgrounds/nv_beach_lagoon.jpg",
            "/Isaac/Materials/Textures/Backgrounds/nv_bay_sf.jpg",
            "/Isaac/Materials/Textures/Backgrounds/nv_city_foggy.jpg",
            "/Isaac/Materials/Textures/Backgrounds/nv_beach_rocky_lagoon.jpg",
            "/Isaac/Materials/Textures/Backgrounds/nv_coconino_mountain.jpg",
            "/Isaac/Materials/Textures/Backgrounds/nv_craterlake_2.jpg",
            "/Isaac/Materials/Textures/Backgrounds/nv_cy_nashville_night_river.jpg",
            "/Isaac/Materials/Textures/Backgrounds/nv_grand_canyon_2.jpg",
            "/Isaac/Materials/Textures/Backgrounds/nv_grass_rainbow_maui.jpg",
            "/Isaac/Materials/Textures/Backgrounds/nv_lagoon_ocean.jpg",
            "/Isaac/Materials/Textures/Backgrounds/nv_lake_trees_mountain.jpg",
            "/Isaac/Materials/Textures/Backgrounds/nv_mountain_overlook_fog.jpg",
        ]

    def set_lights(self):
        background_asset = random.choice(self.background_assets)
        asset_path = self.assets_root_path + background_asset
        self.dome_light.set_texture_files(texture_files=[asset_path])

        dome_light_quat = sample_quat(1, x_range=[-torch.pi, torch.pi],
                                    y_range=[-torch.pi, torch.pi]).tolist()
        
        self.dome_light.set_world_poses(orientations=dome_light_quat)

        distant_light_quat = sample_quat(1, x_range=[-torch.pi, torch.pi],
                                    y_range=[-torch.pi, torch.pi]).tolist()
        self.distant_light.set_world_poses(orientations=distant_light_quat)

class RandomMaterials:
    def __init__(self, robot_material, cube_material, table_material):
        self.robot_material = robot_material
        self.cube_material = cube_material
        self.table_material = table_material

    def random_color(self, num_envs, red_range=[0., 1.],
                     green_range=[0., 1.], blue_range=[0., 1.]):
        colors = np.zeros((num_envs, 3), dtype=np.float32)
        colors[:, 0] = np.random.uniform(red_range[0], red_range[1], size=num_envs)
        colors[:, 1] = np.random.uniform(green_range[0], green_range[1], size=num_envs)
        colors[:, 2] = np.random.uniform(blue_range[0], blue_range[1], size=num_envs) 

        return colors
    
    def apply_random_color(self, num_envs):
        robot_color = self.random_color(num_envs)
        cube_color = self.random_color(num_envs, [0., 0.4], [0.5, 1.0], [0., 0.4])
        table_color = self.random_color(num_envs)

        self.robot_material.set_input_values("diffuse_color_constant", values=robot_color)
        self.cube_material.set_input_values("diffuse_color_constant", values=cube_color)
        self.table_material.set_input_values("diffuse_color_constant", values=table_color)
        