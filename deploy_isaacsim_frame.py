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
from isaacsim.core.utils.stage import open_stage


from contorller.frame_policy_controller import FramePolicyController
from contorller.load_config import get_articulation_props, get_physics_properties, get_robot_joint_properties, parse_env_config

first_step = True
reset_needed = False

# preparing the scene
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()
    
open_stage(usd_path="scene.usd")

my_world = World(physics_dt=1/120, rendering_dt=1/120, stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()  # add ground plane
set_camera_view(
    eye=[0.0, 2.5, 1.5], target=[0.00, 0.00, 0.00], camera_prim_path="/OmniverseKit_Persp"
)  # set camera view

robot = SingleArticulation(prim_path="/World/env_0/so101", name="robot")
cube = SingleRigidPrim(
    prim_path="/World/env_0/Cube/Cube", name="cube"
)

color_camera = Camera(
    prim_path="/World/env_0/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
    resolution=(640, 480),
    frequency=30,
)

contorller = FramePolicyController(
    robot, cube, color_camera
)

my_world.reset()
contorller.initialize()
robot.post_reset()

for _ in range(120):
    my_world.step(render=True)

count = 0

while simulation_app.is_running():
#for _ in range(1):
    contorller.post_reset()

    for _ in range(12):
        my_world.step(render=True)

    for _ in range(100):
        frame_obs = contorller.get_camera_obs()
        state_obs = contorller.get_state_obs()
        
        contorller.compare_feature(state_obs, frame_obs)

        contorller.forward(frame_obs, True)
        for _ in range(4):
            my_world.step(render=True)
        count += 1

simulation_app.close()
