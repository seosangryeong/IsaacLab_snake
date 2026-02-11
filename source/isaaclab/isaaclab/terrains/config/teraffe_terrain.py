# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg

TERAFFE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(15.0, 15.0),        
    border_width=6.0,        
    num_rows=10,              
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pure_flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.4,     
        ),
        
        "giant_island_step": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.6,
            step_height_range=(0.1, 0.3), 
            step_width=10.0,      
            platform_width=5.0, 
            border_width=1.0,
            holes=False,
        ),
    },
)

TERAFFE_WAVE_TERRATIN_CFG = TerrainGeneratorCfg(
    size=(15.0, 15.0),
    border_width=2.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=1.0,
            amplitude_range=(0.5, 0.5),  
            num_waves=2,                   
        ),
    }
)