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
from isaacsim.core.experimental.objects import DomeLight, DistantLight
from isaacsim.core.utils.stage import open_stage
from omni.kit.app import get_app_interface
import omni.appwindow

from PIL import Image

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


color_camera = Camera(
    prim_path="/World/env_0/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
    resolution=(640, 480),
    frequency=120,
)

my_world.reset()
color_camera.initialize()

input_interface = carb.input.acquire_input_interface()
app_window = omni.appwindow.get_default_app_window()

keyboard = app_window.get_keyboard()

def on_key_event(event, *args, **kwargs):
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if event.input == carb.input.KeyboardInput.R:
            frame = color_camera.get_rgb()
            img = Image.fromarray(frame)
            img.save("sim.png")

keyboard_sub = input_interface.subscribe_to_keyboard_events(
    keyboard, on_key_event
)

while simulation_app.is_running():
    my_world.step(render=True)

input_interface.unsubscribe_from_keyboard_events(keyboard, keyboard_sub)

simulation_app.close()
