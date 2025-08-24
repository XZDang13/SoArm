import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg

from .so_arm_env_base_cfg import SO_ARM_101_BASE_ENV

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.02, 0.02, 0.02)

project_root = os.path.dirname(os.path.abspath(__file__))

@configclass
class STACK_TASK_CFG(SO_ARM_101_BASE_ENV):
    episode_length_s = 5.0
    
    observation_space = 6+6+6+3+4

    cube:RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, 0.0, 0.015], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        )
    )

    end_effector: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/gripper_link",
                name="tcp",
                offset=OffsetCfg((0.0, 0.0, -0.075))
            )
        ]
    )

    gripper_marker = FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/Command/goal_pose")