from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open

import sys

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

from PIL import Image

from contorller.state_policy_controller import PolicyController
from contorller.load_config import get_articulation_props, get_physics_properties, get_robot_joint_properties, parse_env_config

# preparing the scene
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

open_stage(usd_path="scene.usd")

my_world = World(physics_dt=1/120, rendering_dt=1/120, stage_units_in_meters=1.0)

robots = Articulation(prim_paths_expr=["/World/Env_.*/so101"], name="robot")
cubes = RigidPrim(prim_paths_expr=["/World/Env_.*/Cube/Cube"], name="cube")
color_cameras = CameraView(
    prim_paths_expr=["/World/Env_.*/Realsense/RSD455/Camera_OmniVision_OV9782_Color"],
    camera_resolution=(640, 480)
)

num_robots = len(robots.prims)
print(num_robots)

my_world.reset()

robots.initialize()
robots.post_reset()
color_cameras.initialize()

state = robots.get_default_state()
print(state.positions)

while simulation_app.is_running():
    my_world.step(render=True)



simulation_app.close()
