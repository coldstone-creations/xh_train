# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import DeeproboticsXhRoughEnvCfg


@configclass
class DeeproboticsXhFlatEnvCfg(DeeproboticsXhRoughEnvCfg):
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

        # If the weight of rewards is 0, set rewards to None
        self.disable_zero_weight_rewards()
