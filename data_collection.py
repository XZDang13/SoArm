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

from uuid import uuid4

import torch.nn.functional as F
import random
import carb
import numpy as np
from PIL import Image
import torch
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
from isaacsim.core.experimental.materials import OmniPbrMaterial

from controller.controller import Controller, RandomLights, RandomMaterials
from controller.writer import Writer

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

set_camera_view(
    eye=[0.0, 2.5, 1.5], target=[0.00, 0.00, 0.00], camera_prim_path="/OmniverseKit_Persp"
)  # set camera view

tile_rows = 5
tile_cols = 5
num_envs = tile_rows * tile_cols

img_width = 640
img_height = 480

print(num_envs)

cloner = GridCloner(spacing=5.0)

clone_paths = cloner.generate_paths("/World/env", num_envs)
cloner.clone(
    source_prim_path="/World/env_0",
    prim_paths=clone_paths,
    copy_from_source=True,
)


robots = Articulation(prim_paths_expr=["/World/env_.*/so101"], name="robot", reset_xform_properties=True)
end_effector = RigidPrim(prim_paths_expr=["/World/env_.*/so101/gripper_link"])
cubes = RigidPrim(prim_paths_expr=["/World/env_.*/Cube"], name="cube")
tables = RigidPrim(prim_paths_expr=["/World/env_.*/Table"], name="table")
color_cameras = CameraView(
    prim_paths_expr=["/World/env_.*/Camera"],
    camera_resolution=(img_width, img_height)
)

robot_material_paths = [f"/World/env_{i}/so101/Looks/material_a_3d_printed" for i in range(num_envs)]
robot_material = OmniPbrMaterial(
    robot_material_paths
)


cube_material_paths = [f"/World/env_{i}/Cube/Looks/CubeColor" for i in range(num_envs)]
cube_material = OmniPbrMaterial(
    cube_material_paths
)

table_material_paths = [f"/World/env_{i}/Table/Looks/TableColor" for i in range(num_envs)]
table_material = OmniPbrMaterial(
    table_material_paths
)

lights = RandomLights(dome_light, distant_light, assets_root_path)
controller = Controller(robots, cubes, color_cameras, end_effector, tile_rows, tile_cols, "state_model.pth")
materails = RandomMaterials(robot_material, cube_material, table_material)

my_world.reset()
controller.initialize()

for _ in range(60):
    my_world.step(render=True)

controller.reset()
epoch = 200
#while simulation_app.is_running():
for e in range(epoch):
    controller.reset()
    controller.random_camera_state()
    
    trajectory_id = str(uuid4())

    for _ in range(12):
        my_world.step(render=True)
    
    for i in range(40):
        lights.set_lights()
        materails.apply_random_color(num_envs)

        current_states = controller.get_state()
        current_frames = controller.get_frame()

        Writer.save_data(trajectory_id, i, current_states, current_frames, tile_rows, tile_cols, img_width, img_height)

        state_obs = controller.get_state_obs()
        state_feature = controller.get_state_feature(state_obs)

        deterministic = (e >= (0.7*epoch) )

        controller.forward(state_feature, deterministic)

        for _ in range(4):
            my_world.step(render=True)

    if (e+1) % 10 == 0:
        print(f"Finish {e+1} episodes")

simulation_app.close()
