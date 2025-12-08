# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL with Waypoint Following and Visualization."""

import argparse
import math
import os
import time
import torch
import gymnasium as gym
from scipy.spatial.transform import Rotation as R

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import carb.input
import omni.appwindow
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# 🔧 올바른 마커 임포트
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG
import isaaclab.sim as sim_utils



def create_path():
    waypoints = []
    
    for i in range(5):
        waypoints.append([0.0, i * 0.4])  # (0, 0) → (0, 1.6)
    
    for i in range(1, 5):
        waypoints.append([i * 0.4, 1.6])  # (0.4, 1.6) → (1.6, 1.6)
    
    for i in range(4, 1, -1):
        waypoints.append([1.6, i * 0.4])  # (1.6, 1.6) → (1.6, 0.8)
    
    for i in range(3, 0, -1):
        waypoints.append([i * 0.4, 0.8])  # (1.2, 0.8) → (0.4, 0.8)
    
    for i in range(1, -1, -1):
        waypoints.append([0.4, i * 0.4])  # (0.4, 0.4) → (0.4, 0.0)
    
    return waypoints

def main():
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy logic
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(ppo_runner.alg.policy, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(ppo_runner.alg.policy, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # ---------------------------------------------------------
    # [설정 1] 웨이포인트 생성 (Global Path)
    # ---------------------------------------------------------
    # X축 방향으로 0.3m 간격으로 점을 생성합니다. (예: 0.3, 0.6, 0.9 ...)
    waypoints = create_path()
    current_wp_idx = 0
    acceptance_radius = 0.1 # 목표 지점 도달 판정 거리 (m)


    waypoint_marker_cfg = CUBOID_MARKER_CFG.replace(prim_path="/Visuals/Waypoints")
    
    waypoint_marker_cfg.markers["cuboid"].size = (0.1, 0.1, 0.1)
    waypoint_marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(1.0, 0.0, 0.0)  # (R, G, B)
    )
    
    waypoint_visualizer = VisualizationMarkers(waypoint_marker_cfg)

    num_points = len(waypoints)
    marker_locations = torch.zeros((num_points, 3), device=env.unwrapped.device)
    
    for i, wp in enumerate(waypoints):
        marker_locations[i, 0] = wp[0]  # X
        marker_locations[i, 1] = wp[1]  # Y
        marker_locations[i, 2] = 0.1    # Z (바닥에서 0.1m 띄워서 표시)

    # 5. 시각화 적용
    waypoint_visualizer.set_visibility(True)
    waypoint_visualizer.visualize(marker_locations)
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # [설정 2] 로봇 객체 및 커맨드 매니저 가져오기
    # ---------------------------------------------------------
    # 'robot' 키는 cfg 파일에 정의된 로봇의 이름이어야 합니다.
    try:
        robot = env.unwrapped.scene["robot"]
    except KeyError:
        print(f"[ERROR] 'robot' not found in scene. Available keys: {env.unwrapped.scene.keys()}")
        return

    # 커맨드 매니저
    try:
        # kanake_command = env.unwrapped.command_manager.get_term("kanake_command")
        kanake_command = env.unwrapped.command_manager.get_term("kanake_command")

    except Exception as e:
        print(f"[ERROR] Command term 'kanake_command' not found. Check your config. Error: {e}")
        return

    # ✅ 커맨드 리샘플링 비활성화
    kanake_command.cfg.resampling_time_range = (1e9, 1e9)
    
    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    print("[INFO] Starting Waypoint Navigation...")

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        
        # 현재 위치 가져오기
        current_pos = robot.data.root_pos_w[0, :2].cpu().numpy()
        
        # 기존 코드 수정
        if current_wp_idx < len(waypoints):
            target_pos = waypoints[current_wp_idx]
            
            # 거리 계산 (도달 확인용)
            dx_world = target_pos[0] - current_pos[0]
            dy_world = target_pos[1] - current_pos[1]
            dist = math.sqrt(dx_world**2 + dy_world**2)
            
            # 도달 확인
            if dist < acceptance_radius:
                print(f"[Nav] ✅ Reached Waypoint {current_wp_idx} at {target_pos}")
                current_wp_idx += 1
                
                if current_wp_idx >= len(waypoints):
                    print("[Nav] 🎉 All waypoints cleared!")
                    current_wp_idx = len(waypoints) - 1
                
                target_pos = waypoints[current_wp_idx]

            # ✅ 올바른 방법: pos_command_w 직접 설정
            kanake_command.pos_command_w[0, 0] = target_pos[0]  # 월드 X 좌표
            kanake_command.pos_command_w[0, 1] = target_pos[1]  # 월드 Y 좌표
            kanake_command.pos_command_w[0, 2] = 0.25           # 기본 Z 높이
            kanake_command.heading_command_w[0] = 0.0           # 기본 heading
            
            # 디버깅 출력
            if timestep % 10 == 0:
                print(f"[DEBUG] WP{current_wp_idx}: Target={target_pos}, Current=[{current_pos[0]:.3f}, {current_pos[1]:.3f}], Dist={dist:.3f}")
                print(f"[DEBUG] Command_w: [{kanake_command.pos_command_w[0, 0]:.3f}, {kanake_command.pos_command_w[0, 1]:.3f}]")
                print(f"[DEBUG] Command_b: [{kanake_command.pos_command_b[0, 0]:.3f}, {kanake_command.pos_command_b[0, 1]:.3f}]")
                    
            else:
                # 모든 웨이포인트 완료 시 마지막 지점 유지
                final_target = waypoints[-1]
                kanake_command.command[0, 0] = final_target[0]
                kanake_command.command[0, 1] = final_target[1]
                kanake_command.command[0, 2] = 0.025  # 기본 Z 높이
        # ---------------------------------------------------------

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        
        # Video recording logic
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        else:
            timestep += 1

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()