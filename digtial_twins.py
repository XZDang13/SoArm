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
from isaacsim.core.experimental.objects import DomeLight

from controller.controller import RandomLights
from PIL import Image

width, height = 640, 480
camera_matrix = [[626.07628932, 0.00000000e+00, 317.9777863],
                 [0.00000000e+00, 626.10882087, 241.71883685],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
distortion_coefficients = [2.13396756e-01, -6.35715523e-01, -1.10409410e-03, -5.67505690e-05, 6.25051766e-01]


pixel_size = 3

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
    prim_path="/World/env_0/Camera",
    resolution=(width, height),
    frequency=120
)

((fx, _, cx), (_, fy, cy), (_, _, _)) = camera_matrix  # fx, fy are in pixels, cx, cy are in pixels
horizontal_aperture = pixel_size * width * 1e-4  # convert to meters
vertical_aperture = pixel_size * height * 1e-4  # convert to meters
focal_length_x = pixel_size * fx * 1e-4  # convert to meters
focal_length_y = pixel_size * fy * 1e-4  # convert to meters
focal_length = (focal_length_x + focal_length_y) / 2  # convert to meters

color_camera.set_focal_length(focal_length)
color_camera.set_horizontal_aperture(horizontal_aperture)
color_camera.set_vertical_aperture(vertical_aperture)
color_camera.set_opencv_pinhole_properties(cx=cx, cy=cy, fx=fx, fy=fy, pinhole=distortion_coefficients)
#color_camera.set_focal_length(0.193)
#color_camera.set_horizontal_aperture(0.3896, True)
#color_camera.set_vertical_aperture(0.2453, False)

my_world.reset()
color_camera.initialize()


print(color_camera.get_focal_length())
print(color_camera.get_horizontal_aperture())
print(color_camera.get_vertical_aperture())
print(color_camera.get_horizontal_fov())
print(color_camera.get_vertical_fov())

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
