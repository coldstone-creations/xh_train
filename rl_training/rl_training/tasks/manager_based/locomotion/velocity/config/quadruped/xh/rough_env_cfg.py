from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from rl_training.assets.xh import XH_CFG  # isort: skip


@configclass
class XhRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base_link"
    foot_link_name = ".*_foot"
    # fmt: off
    joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]

    link_names = [
       'base_link',
       'FL_hip', 'FR_hip', 'RL_hip', 'RR_hip',
       'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh',
       'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf',
       'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot',
    ]

    hip_joint_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    ]

    thigh_joint_names = [
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    ]

    calf_joint_names = [
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.robot = XH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner.pattern_cfg.resolution = 0.07

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel = None  # type: ignore
        self.observations.policy.height_scan = None  # type: ignore
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.2
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.125,
            ".*_thigh_joint": 0.25,
            ".*_calf_joint": 0.25,
        }
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.0, 0.0),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.0, 0.0),
            },
        }

        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = self.link_names
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_com_positions.params["asset_cfg"].body_names = self.base_link_name
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = self.joint_names

        # set terrain generation probability to 0 for boxes and stairs
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.4
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.3
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.3
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.0
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.04)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # ------------------------------Rewards------------------------------
        self.rewards.action_rate_l2.weight = -0.1

        self.rewards.base_height_l2.weight = -20.0
        # xh: 实机站立姿态 thigh=-0.6, calf=0.7
        #     USD 重新生成后坐标系统一, 无需偏移
        self.rewards.base_height_l2.params["target_height"] = 0.25
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

        self.rewards.feet_air_time_lin_xy.weight = 3.0
        self.rewards.feet_air_time_lin_xy.params["threshold"] = 0.2
        self.rewards.feet_air_time_lin_xy.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_x_neg.weight = 0.0
        self.rewards.feet_air_time_x_neg.params["threshold"] = 0.5
        self.rewards.feet_air_time_x_neg.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_ang_z.weight = 3.0
        self.rewards.feet_air_time_ang_z.params["threshold"] = 0.4
        self.rewards.feet_air_time_ang_z.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -0.5
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.05
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.foot_impact_velocity.weight = -1.0
        self.rewards.foot_impact_velocity.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.foot_impact_velocity.params["asset_cfg"].body_names = [self.foot_link_name]
        # default_joint_pos 来自 init_state (xh.py), 已设为实机站立姿态:
        #   hip=0, thigh=-0.55, calf=0.8
        # stand_still 会惩罚偏离站立姿态的关节, 配合 stand_still_vel_penalty
        self.rewards.stand_still.weight = -5.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.stand_still.params["command_threshold"] = 0.1
        self.rewards.stand_still_vel_penalty.weight = -5.0
        self.rewards.stand_still_vel_penalty.params["command_threshold"] = 0.1
        self.rewards.feet_height_body.weight = -0.0
        self.rewards.feet_height_body.params["target_height"] = -0.25
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = -0.0
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.contact_forces.weight = -1e-1
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.rewards.lin_vel_z_l2.weight = -20.0
        self.rewards.ang_vel_xy_l2.weight = -0.25

        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_ang_vel_z_exp.weight = 6.0
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5

        self.rewards.undesired_contacts.weight = -5.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]

        self.rewards.joint_torques_l2.weight = -2.5e-4
        self.rewards.joint_acc_l2.weight = -1e-8
        self.rewards.joint_deviation_l1.weight = -0.0
        self.rewards.joint_deviation_l1.params["asset_cfg"].joint_names = [".*_hip_joint"]
        self.rewards.joint_power.weight = -8e-4
        self.rewards.flat_orientation_l2.weight = -10.0

        # add the following rewards to improve the gait
        self.rewards.feet_gait.weight = 1.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = [
            ["FL_foot", "RR_foot"],
            ["FR_foot", "RL_foot"]
        ]

        self.rewards.phase_foot_trajectory_exp.weight = 2.0
        self.rewards.phase_foot_trajectory_exp.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
        ]

        self.rewards.joint_pos_limits.weight = -2.0
        self.rewards.feet_contact_without_cmd.weight = 3.0
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]

        # added rewards
        self.rewards.hipx_joint_pos_penalty.weight = -2.0
        self.rewards.hipx_joint_pos_penalty.params["asset_cfg"].joint_names = self.hip_joint_names
        self.rewards.hipy_joint_pos_penalty.weight = -3.0
        self.rewards.hipy_joint_pos_penalty.params["asset_cfg"].joint_names = self.thigh_joint_names
        self.rewards.knee_joint_pos_penalty.weight = -8.0
        self.rewards.knee_joint_pos_penalty.params["asset_cfg"].joint_names = self.calf_joint_names

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "XhRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = None
        # xh 身体较长，45° 阈值太高。用 bad_orientation (v1) 替换 bad_orientation_2，
        # 允许自定义倾斜角度阈值。0.5 rad ≈ 28.6°
        self.terminations.bad_orientation_2 = None
        from isaaclab.managers import TerminationTermCfg as DoneTerm
        from isaaclab.envs.mdp.terminations import bad_orientation
        self.terminations.bad_orientation = DoneTerm(
            func=bad_orientation,
            params={"limit_angle": 0.5},
            time_out=True,
        )

        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
