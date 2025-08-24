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
            max_depenetration_velocity=5.0,
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
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ],
            effort_limit_sim=1.9,
            velocity_limit_sim=1.5,
            stiffness={
                "shoulder_pan": 10.0,
                "shoulder_lift": 10.0,
                "elbow_flex": 10.0,
                "wrist_flex": 10.0,
                "wrist_roll": 10.0,
            },
            damping={
                "shoulder_pan": 2.0,
                "shoulder_lift": 2.0,
                "elbow_flex": 2.0,
                "wrist_flex": 2.0,
                "wrist_roll": 2.0,
            },
        ),


        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "gripper"
            ],
            effort_limit_sim=2.5,
            velocity_limit_sim=1.5,
            stiffness=10.0,
            damping=2.0
        ),
    },
)