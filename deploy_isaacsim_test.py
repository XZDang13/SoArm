from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open

import sys

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
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

my_world = World(physics_dt=1/120, rendering_dt=1/120, stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()  # add ground plane
set_camera_view(
    eye=[0.0, 2.5, 1.5], target=[0.00, 0.00, 0.00], camera_prim_path="/OmniverseKit_Persp"
)  # set camera view

asset_path = "env/assets/so101/so101.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Robot")  # add robot to stage
robot = SingleArticulation(prim_path="/World/Robot", name="robot",
                           position=np.array([0.0, 0.0, 0.8]))

robot.set_default_state(position=np.array([0.0, 0.0, 0.8]))

my_world.reset()
robot.initialize()
robot.post_reset()

for _ in range(100):
    my_world.step(render=True)

steps = 0
for _ in range(100):
    for _ in range(4):
        my_world.step(render=True)

simulation_app.close()
