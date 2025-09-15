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
from isaacsim.core.experimental.objects import DomeLight, DistantLight

from controller.controller import RandomLights, Controller
from controller.writer import Writer

# preparing the scene
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

open_stage(usd_path="scene.usd")

dome_light = DomeLight(
    "/Environment/DomeLight"
)
distant_light = DistantLight(
    "/Environment/DistantLight"
)

my_world = World(physics_dt=1/120, rendering_dt=1/120, stage_units_in_meters=1.0,
                 backend="torch", device="cuda:0")

tile_rows = 5
tile_cols = 5
num_envs = tile_rows * tile_cols

print(num_envs)

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
    camera_resolution=(1280, 720)
)

controller = Controller(robots, cubes, color_cameras, tile_rows, tile_cols, "state_model.pth")
lights = RandomLights(dome_light, distant_light, assets_root_path)

my_world.reset()
controller.initialize()

controller.reset()

for _ in range(60):
    my_world.step(render=True)


start = time.perf_counter()
for _ in range(60):
    trajectory_id = str(uuid4())
    controller.reset()
    lights.set_lights()

    for _ in range(12):
        my_world.step(render=True)
    
    for i in range(20):
        state_obs = controller.get_state_obs()
        frame_obs = controller.get_frame()

        Writer.save_data(trajectory_id, i, state_obs, frame_obs, tile_rows, tile_cols)
        state_features = controller.get_state_feature(state_obs)
        controller.forward(state_features, False)

        for _ in range(4):
            my_world.step(render=True)

for _ in range(20):
    trajectory_id = str(uuid4())
    controller.reset()
    lights.set_lights()

    for _ in range(12):
        my_world.step(render=True)
    
    for i in range(20):
        state_obs = controller.get_state_obs()
        frame_obs = controller.get_frame()

        Writer.save_data(trajectory_id, i, state_obs, frame_obs, tile_rows, tile_cols)
        state_features = controller.get_state_feature(state_obs)
        controller.forward(state_features, True)

        for _ in range(4):
            my_world.step(render=True)

end = time.perf_counter()
print(f"took {end - start:.4f} seconds")


simulation_app.close()
