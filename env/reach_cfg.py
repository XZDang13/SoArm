import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg, FrameTransformer
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform

from .so_arm_env_base_cfg import SO_ARM_101_BASE_ENV

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.02, 0.02, 0.02)

project_root = os.path.dirname(os.path.abspath(__file__))

@configclass
class REACH_TASK_CFG(SO_ARM_101_BASE_ENV):
    episode_length_s = 5.0

    observation_space = 6+6+6+3+4+4

    gripper_marker = VisualizationMarkersCfg(
        prim_path="/Visuals/GripperMarker",
        markers={
            "gripper": sim_utils.UsdFileCfg(
                usd_path=f"{project_root}/assets/so101/gripper_visual.usd",
                scale=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.25, 0.0)),
            ),
        }
    )

    jaw_marker = VisualizationMarkersCfg(
        prim_path="/Visuals/JawMarker",
        markers={
            "gripper": sim_utils.UsdFileCfg(
                usd_path=f"{project_root}/assets/so101/jaw_visual.usd",
                scale=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.25, 0.0)),
            ),
        }
    )
    
    x_range = [0.005, 0.1]
    y_range = [-0.1, 0.1]
    z_range = [0.01, 0.035]