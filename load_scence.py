from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open

import sys

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from omni.isaac.sensor import Camera

from PIL import Image

from contorller.policy_controller import PolicyController
from contorller.load_config import get_articulation_props, get_physics_properties, get_robot_joint_properties, parse_env_config

# preparing the scene
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

open_stage(usd_path="env/assets/so101/scenario.usd")

my_world = World(physics_dt=1/120, rendering_dt=1/120, stage_units_in_meters=1.0)


while simulation_app.is_running():
    my_world.step(render=True)



simulation_app.close()
