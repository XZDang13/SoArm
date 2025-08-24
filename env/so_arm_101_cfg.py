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
            disable_gravity=False
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            fix_root_link=True
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
            effort_limit_sim=10.,
            velocity_limit_sim=10.,
            stiffness=17.8,
            damping=0.60
        ),


        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "gripper"
            ],
            effort_limit_sim=10.,
            velocity_limit_sim=10.,
            stiffness=17.8,
            damping=0.6
        ),
    },
    soft_joint_pos_limit_factor=1.0
)