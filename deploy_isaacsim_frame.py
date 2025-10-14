from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open

import sys

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

first_step = True
reset_needed = False

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

set_camera_view(
    eye=[0.0, 2.5, 1.5], target=[0.00, 0.00, 0.00], camera_prim_path="/OmniverseKit_Persp"
)  # set camera view

tile_rows = 1
tile_cols = 1
num_envs = tile_rows * tile_cols

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
    camera_resolution=(640, 480)
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
controller = Controller(robots, cubes, color_cameras, end_effector, tile_rows, tile_cols, "state_model.pth", "frame_model.pth")
materails = RandomMaterials(robot_material, cube_material, table_material)

my_world.reset()
controller.initialize()

for _ in range(60):
    my_world.step(render=True)

count = 0

pre_init_state_feature = None
pre_init_frame_feature = None

controller.reset()
while simulation_app.is_running():
#for e in range(1):
    controller.reset()
    #controller.random_camera_state()
    #materails.apply_random_color(num_envs)
    

    for _ in range(12):
        my_world.step(render=True)
    
    for i in range(50):
        state_obs = controller.get_state_obs()
        frame_obs = controller.get_camera_obs()

        state_feature = controller.get_state_feature(state_obs)
        frame_feature = controller.get_frame_feature(frame_obs)

        print(F.cosine_similarity(state_feature, frame_feature))

        controller.forward(frame_feature, True)

        for _ in range(4):
            my_world.step(render=True)
    

simulation_app.close()