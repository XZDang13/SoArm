import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

project_root = os.path.dirname(os.path.abspath(__file__))

SO_ARM_101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{project_root}/assets/so101/so101.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0028),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,#-1.740,
            "elbow_flex": 0.0, #1.5708,
            "wrist_flex": 0.0,#1.13446,
            "wrist_roll": 0.0,
            "gripper": 0.0
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "all": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper"
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100.0,
            stiffness=17.8,
            damping=0.6,
            armature=0.028
        ),
    },
)