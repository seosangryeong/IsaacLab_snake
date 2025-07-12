# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg

KANAKE_PLANE_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=2.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.HfTerrainBaseCfg(proportion=1.0),
    },
)

KANAKE_RANDOM_TERRAIN_CFG = TerrainGeneratorCfg(
    # 가로세로 길이
    size=(5.0, 5.0),
    # 평평한 안전영역 생성(떨어짐 방지)
    border_width=1.0,
    num_rows=1,
    num_cols=1,
    # n m 마다 높이 데이터 포인트 생성
    horizontal_scale=0.03,
    #높이 변화 범위
    vertical_scale=0.001,
    # 0.75 라디안 이상 경사는 수직 처리
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.0, 0.008),  # 높이 변화
            noise_step=0.001,          # 최소 변화
        ),
    },
)