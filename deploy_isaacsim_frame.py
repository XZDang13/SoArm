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

my_world = World(physics_dt=1/120, rendering_dt=1/120, stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()  # add ground plane
set_camera_view(
    eye=[0.0, 2.5, 1.5], target=[0.00, 0.00, 0.00], camera_prim_path="/OmniverseKit_Persp"
)  # set camera view

table_xform_prim_path = "/World/TableXform"
table_rigidbody_prim_path = "/World/TableXform/Table"
table_usd_path = "env/assets/so101/Table.usd"

table_xform_name = "table_xform"
table_rigidbody_name = "table_rigidbody"
table_position = np.array([0.72, 0.0, 0.4])
table_orientation = np.array([1.0, 0.0, 0.0, 0.0])

add_reference_to_stage(table_usd_path, table_xform_prim_path)

table_xform = SingleXFormPrim(prim_path=table_xform_prim_path, name=table_xform_name)

table = SingleRigidPrim(
    prim_path=table_rigidbody_prim_path, name=table_rigidbody_name, position=table_position, orientation=table_orientation
)

camera_asset_path = assets_root_path + "/Isaac/Sensors/Intel/RealSense/rsd455.usd"
camera_xform_prim_path = "/World/Camera"
camera_xform_name = "RSD455_xform"
camera_rigid_prim_path = "/World/Camera/RSD455"
camera_rigid_name = "RSD455_rigid"

add_reference_to_stage(camera_asset_path, camera_xform_prim_path)

camera_xform = SingleXFormPrim(prim_path=camera_xform_prim_path, name=camera_xform_name)
camera_rigidbody = SingleRigidPrim(
    prim_path=camera_rigid_prim_path, name=camera_rigid_name,
    position=np.array([0.5, -0.5, 0.875]), orientation=np.array([0.9238795, 0.0, 0.0, -0.3826834])
)

camera_rigidbody.disable_rigid_body_physics()

color_camera = Camera(
    prim_path="/World/Camera/RSD455/Camera_OmniVision_OV9782_Color",
    resolution=(640, 480),
    frequency=30,
)


cube_xform_prim_path = "/World/GreenCube"
cube_rigidbody_prim_path = "/World/GreenCube/Cube"
cube_usd_path = "env/assets/so101/Cube.usd"

cube_xform_name = "cube_xform"
cube_rigidbody_name = "cube_rigidbody"
cube_position = np.array([0.27, 0.0, 0.8177])
cube_orientation = np.array([1.0, 0.0, 0.0, 0.0])

add_reference_to_stage(cube_usd_path, cube_xform_prim_path)

cube_xform = SingleXFormPrim(prim_path=cube_xform_prim_path, name=cube_xform_name)

cube = SingleRigidPrim(
    prim_path=cube_rigidbody_prim_path, name=cube_rigidbody_name, position=cube_position, orientation=cube_orientation
)

asset_path = "env/assets/so101/so101.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Robot")  # add robot to stage
robot = SingleArticulation(prim_path="/World/Robot", name="robot",
                           position=np.array([0.0, 0.0, 0.8]))

robot.set_default_state(position=np.array([0.0, 0.0, 0.8]))

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
        contorller.forward(frame_obs)
        for _ in range(4):
            my_world.step(render=True)
        count += 1

simulation_app.close()
