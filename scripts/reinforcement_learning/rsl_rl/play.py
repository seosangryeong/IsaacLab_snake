# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

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

import gymnasium as gym
import os
import time
import torch
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

# PLACEHOLDER: Extension template (do not remove this comment)


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
    
    
    input_interface = carb.input.acquire_input_interface()
    app_window = omni.appwindow.get_default_app_window()
    keyboard_dev = app_window.get_keyboard()
    # from isaaclab.markers import VisualizationMarkers
    # from isaaclab_tasks.manager_based.classic.kanake.kanake_env_cfg import TARGET_MARKER_CFG,TARGET_BOX

    # import torch
    # target_marker = VisualizationMarkers(TARGET_BOX)



    # from scipy.spatial.transform import Rotation as R
    # import torch

    # device = env.unwrapped.device if hasattr(env.unwrapped, "device") else "cuda:0"
    # 오일러 각 순서: 'xyz', 각도 단위: 도
    # r = R.from_euler('xyz', [0, -90, 0], degrees=True)
    # quat = r.as_quat()  # [x, y, z, w]
    # orientation = torch.tensor([quat], device=device)  # shape: (1, 4)

    # position = torch.tensor([[3.0, 0.0, 0.1]], device=device)
    # scales = torch.tensor([[0.5, 0.5, 0.5]], device=device)
    # target_marker.set_visibility(True)
    # target_marker.visualize(position, orientation, scales)
    # ------------------------



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

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.policy, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.policy, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # env.reset()


    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    # while simulation_app.is_running():
    #     start_time = time.time()
        
    #     # run everything in inference mode
    #     with torch.inference_mode():
    #         # agent stepping
    #         actions = policy(obs)
    #         # env stepping
    #         obs, _, _, _ = env.step(actions)
    #     if args_cli.video:
    #         timestep += 1
    #         # Exit the play loop after recording one video
    #         if timestep == args_cli.video_length:
    #             break

    #     # time delay for real-time evaluation
    #     sleep_time = dt - (time.time() - start_time)
    #     if args_cli.real_time and sleep_time > 0:
    #         time.sleep(sleep_time)

    # # close the simulator
    # env.close()
    # while simulation_app.is_running():
    #     start_time = time.time()
        
    #     # 🔧 순차적 커맨드 제어 추가
    #     command_interval = 160  # 2초 = 160 steps (80Hz * 2초)
        
    #     if timestep % command_interval == 0:
    #         # 순차적 커맨드 패턴 (4단계 반복)
    #         cycle = (timestep // command_interval) % 2
    #         kanake_command = env.unwrapped.command_manager.get_term("kanake_command")

    #         if cycle == 0:
    #             # 1단계: 전진 (x = 0.5)
    #             kanake_command.command[0, 0] = 0.5   # x속도
    #             kanake_command.command[0, 1] = 0.0   # y속도
    #             kanake_command.command[0, 2] = 0.0   # 회전
    #             print(f"Step {timestep}: Forward (0.5)")
                
    #         elif cycle == 1:
    #             # 2단계: 후진 (x = -0.5)
    #             kanake_command.command[0, 0] = -0.5  # x속도
    #             kanake_command.command[0, 1] = 0.0   # y속도
    #             kanake_command.command[0, 2] = 0.0   # 회전
    #             print(f"Step {timestep}: Backward (-0.5)")
                
    #         # elif cycle == 2:
    #         #     # 3단계: 우회전 (z = 1.0)
    #         #     env.unwrapped.command_manager.get_term("kanake_command").command[0, 0] = 0.0   # x속도
    #         #     env.unwrapped.command_manager.get_term("kanake_command").command[0, 1] = 0.0   # y속도
    #         #     env.unwrapped.command_manager.get_term("kanake_command").command[0, 2] = 1.0   # 회전
    #         #     print(f"Step {timestep}: Turn Right (1.0)")
                
    #         # else:  # cycle == 3
    #         #     # 4단계: 좌회전 (z = -1.0)
    #         #     env.unwrapped.command_manager.get_term("kanake_command").command[0, 0] = 0.0   # x속도
    #         #     env.unwrapped.command_manager.get_term("kanake_command").command[0, 1] = 0.0   # y속도
    #         #     env.unwrapped.command_manager.get_term("kanake_command").command[0, 2] = -1.0  # 회전
    #         #     print(f"Step {timestep}: Turn Left (-1.0)")
        
    #     # run everything in inference mode
    #     with torch.inference_mode():
    #         # agent stepping
    #         actions = policy(obs)
    #         # env stepping
    #         obs, _, _, _ = env.step(actions)
        
    #     if args_cli.video:
    #         timestep += 1
    #         # Exit the play loop after recording one video
    #         if timestep == args_cli.video_length:
    #             break
    #     else:
    #         timestep += 1  # 🔧 비디오 모드가 아닐 때도 timestep 증가

    #     # time delay for real-time evaluation
    #     sleep_time = dt - (time.time() - start_time)
    #     if args_cli.real_time and sleep_time > 0:
    #         time.sleep(sleep_time)

    # # close the simulator
    # env.close()
    while simulation_app.is_running():
        start_time = time.time()
        
        # ---------------------------------------------------------
        # [수정된 부분 2] 키 입력 감지 및 커맨드 설정
        # ---------------------------------------------------------
        kanake_command = env.unwrapped.command_manager.get_term("kanake_command")
        
        # 초기화
        x_vel = 0.0
        y_vel = 0.0
        z_vel = 0.0
        
        # W / S / UP / DOWN (전진/후진)
        if input_interface.get_keyboard_value(keyboard_dev, carb.input.KeyboardInput.W) > 0 or \
           input_interface.get_keyboard_value(keyboard_dev, carb.input.KeyboardInput.UP) > 0:
            y_vel = 0.8
            print("Forward")
        elif input_interface.get_keyboard_value(keyboard_dev, carb.input.KeyboardInput.S) > 0 or \
             input_interface.get_keyboard_value(keyboard_dev, carb.input.KeyboardInput.DOWN) > 0:
            y_vel = -0.8
            print("Backward")

        # A / D / LEFT / RIGHT (좌우 이동)
        if input_interface.get_keyboard_value(keyboard_dev, carb.input.KeyboardInput.A) > 0 or \
           input_interface.get_keyboard_value(keyboard_dev, carb.input.KeyboardInput.LEFT) > 0:
            x_vel = -0.5
            print("Left")
        elif input_interface.get_keyboard_value(keyboard_dev, carb.input.KeyboardInput.D) > 0 or \
             input_interface.get_keyboard_value(keyboard_dev, carb.input.KeyboardInput.RIGHT) > 0:
            x_vel = 0.5
            print("Right")

        # Q / E (회전)
        if input_interface.get_keyboard_value(keyboard_dev, carb.input.KeyboardInput.Q) > 0:
            z_vel = 1.0
            print("Turn Left")
        elif input_interface.get_keyboard_value(keyboard_dev, carb.input.KeyboardInput.E) > 0:
            z_vel = -1.0
            print("Turn Right")

        # 커맨드 적용
        kanake_command.command[0, 0] = x_vel
        kanake_command.command[0, 1] = y_vel
        kanake_command.command[0, 2] = z_vel
        # ---------------------------------------------------------

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        
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

    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
