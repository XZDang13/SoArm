from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open

import sys

import torch
from model.actor_critic import EncoderNet, StochasticDDPGActor
from RLAlg.nn.steps import DeterministicContinuousPolicyStep

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path

device = torch.device("cuda:0")
encoder = EncoderNet(6+6+6+3+4+4, [256, 256, 256]).to(device)
actor = StochasticDDPGActor(encoder.dim, [256, 256], 6).to(device)

encoder_params, actor_params, _ = torch.load("model.pth")
encoder.load_state_dict(encoder_params)
actor.load_state_dict(actor_params)

encoder.eval()
actor.eval()

@torch.no_grad()
def get_action(obs):
    obs = torch.from_numpy(obs).float().to(device)

    feature = encoder(obs)
    step:DeterministicContinuousPolicyStep = actor(feature, std=1.0)
    action = step.mean.cpu().numpy()

    return action

# preparing the scene
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(physics_dt=1/60, rendering_dt=2/60, stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()  # add ground plane
set_camera_view(
    eye=[0.0, 2.5, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
)  # set camera view

# Add Franka
asset_path = "env/assets/so101/so101.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Arm")  # add robot to stage
robot = Articulation(prim_paths_expr="/World/Arm", name="my_arm")  # create an articulation object

robot.set_solver_position_iteration_counts(np.full((1,), 4))
robot.set_solver_velocity_iteration_counts(np.full((1,), 64))

robot.set_enabled_self_collisions(np.full((1,), True))
robot.set_gains(
    kps=np.array([17.8, 17.8, 17.8, 17.8, 17.8, 17.8]),
    kds=np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6]),
)


# set the initial poses of the arm and the car so they don't collide BEFORE the simulation starts
robot.set_world_poses(positions=np.array([[0.0, 0.0, 0.0]]) / get_stage_units())

# initialize the world
my_world.reset()

current_pos = robot.get_joint_positions().copy()
pre_pos = current_pos.copy()
pre_action = np.array([[0, 0, 0, 0, 0, 0]])
goal_state = np.array([[0.25, 0.0, 0.17, 1.0, 0.0, 0.0, 0.0, 0.7071, 0.7071, 0.0, 0.0]])

for i in range(1000):
    obs = np.concatenate([goal_state, current_pos, pre_pos, pre_action], axis=-1)
    action = get_action(obs)
    target_pos = current_pos + action * 0.25
    robot.set_joint_position_targets(target_pos)

    for _ in range(2):
        my_world.step(render=True)

    pre_pos = current_pos.copy()
    current_pos = robot.get_joint_positions().copy()
    pre_action = action.copy()

simulation_app.close()