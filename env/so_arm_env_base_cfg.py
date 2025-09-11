import os

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

from .so_arm_101_cfg import SO_ARM_101_CFG

project_root = os.path.dirname(os.path.abspath(__file__))

@configclass
class SO_ARM_101_BASE_ENV(DirectRLEnvCfg):
    episode_length_s = 10.0

    decimation = 4

    observation_space = 12
    action_space = 6
    state_space = 0

    action_scale = 0.15

    is_training = True
    debug = True

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            enable_ccd=True
        )
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene:InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, env_spacing=2.0, replicate_physics=True
    )

    robot:ArticulationCfg = SO_ARM_101_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    table:RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, 0.4], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/so101/Table.usd",
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