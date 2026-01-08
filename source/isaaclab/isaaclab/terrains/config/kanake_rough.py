# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(4.0, 4.0),
    border_width=4.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "dense_rubble": terrain_gen.MeshRandomGridTerrainCfg(
        #             proportion=0.2,
        #             grid_width=0.15,          
        #             grid_height_range=(0.02, 0.06), 
        #             platform_width=1.0,rndornrn
        #         ),
        "micro_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.4,
                    step_height_range=(0.05, 0.10), # 5~10cm 높이
                    step_width=0.2,           # 계단 폭 20cm 
                    platform_width=1.0,
                    border_width=1.0,
                    holes=False,
                ),
        # "sloped_ramp": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.2,
        #     slope_range=(0.15, 0.3), #26.5도
        #     platform_width=1.0,
        #     border_width=0.25,
        # ),
        # "rough_noise": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.2,
        #     noise_range=(0.02, 0.08), # 2~8cm의 자잘한 굴곡
        #     noise_step=0.02,
        #     border_width=0.25
        # ),
    },
)
"""Rough terrains configuration."""
