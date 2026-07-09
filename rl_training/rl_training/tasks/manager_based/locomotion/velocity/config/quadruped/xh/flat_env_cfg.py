from isaaclab.utils import configclass

from .rough_env_cfg import XhRoughEnvCfg


@configclass
class XhFlatEnvCfg(XhRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.observations.policy.height_scan = None  # type: ignore
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        # Flat terrain: gait_level must be 1.0 since no terrain curriculum exists
        # to update it. Otherwise all gait-scaled rewards (base_height_l2,
        # joint_torques_l2, feet_air_time_*, etc.) would be multiplied by 0.
        import rl_training.tasks.manager_based.locomotion.velocity.mdp.rewards as _rewards_mod
        _rewards_mod.gait_level = 1.0

        # If the weight of rewards is 0, set rewards to None
        self.disable_zero_weight_rewards()
