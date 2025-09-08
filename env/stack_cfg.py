import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

from .so_arm_env_base_cfg import SO_ARM_101_BASE_ENV

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.02, 0.02, 0.02)

project_root = os.path.dirname(os.path.abspath(__file__))

@configclass
class STACK_TASK_CFG(SO_ARM_101_BASE_ENV):
    episode_length_s = 2.0
    
    observation_space = 6+6+3+4

    green_cube:RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/GreenCube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0, 0.019], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
            scale=(0.76, 0.76, 0.76),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    red_cube:RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/RedCube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0, 0.019], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
            scale=(0.76, 0.76, 0.76),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    end_effector: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/gripper_link",
                name="tcp",
                offset=OffsetCfg((0.02, 0.0, -0.095))
            )
        ]
    )

    gripper_marker = FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/Command/goal_pose")

    gripper_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gripper_link",
        # Only care about contacts with the cube
        filter_prim_paths_expr=["/World/envs/env_.*/GreenCube/Cube"]
    )

    jaw_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/moving_jaw_so101_v1_link",
        # Only care about contacts with the cube
        filter_prim_paths_expr=["/World/envs/env_.*/GreenCube/Cube"]
    )