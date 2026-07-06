# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from rl_training.assets import ISAACLAB_ASSETS_DATA_DIR

DEEPROBOTICS_XH_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/xh_usd/xh.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.1,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.22),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": -0.6,
            ".*_calf_joint": 0.7, 
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.99,
    actuators={
        "Hip": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=27.0,      # 额定 3Nm，hip 不需要大力矩
            velocity_limit=3.77,
            stiffness=50.0,
            damping=0.7,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=1,
        ),
        "Thigh": DelayedPDActuatorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=7.0,      # 额定 3Nm
            velocity_limit=12.57,
            stiffness=48.0,
            damping=0.7,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=1,
        ),
        "Calf": DelayedPDActuatorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=7.0,      # 额定 9Nm，换用 hip 电机规格
            velocity_limit=12.57,
            stiffness=30.0,
            damping=0.2,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=1,
        ),
    },
)
