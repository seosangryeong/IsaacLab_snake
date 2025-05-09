import sys, os, glob, argparse, torch
sys.path.append("/home/nuc/IsaacLab")
import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args
from isaaclab.app import AppLauncher



parser = argparse.ArgumentParser()
parser.add_argument("--task",      type=str, required=True)
parser.add_argument("--num_envs",  type=int, default=64)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if hasattr(args_cli, "video") and args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.manager_based.classic.kanake.clustering import run_clustering
import gymnasium as gym
import isaaclab.envs.ui as ui





def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    action_files = sorted(glob.glob("action_dumps/actions_iter*.pt"), key=os.path.getmtime, reverse=True)
    if not action_files:
        raise FileNotFoundError("action_dumps/actions_iter*.pt 가 없습니다.")
    actions_path = action_files[0]
    out_prefix = os.path.splitext(os.path.basename(actions_path))[0].replace("actions_", "")
    print(f"[INFO] Using actions file: {actions_path}")

    _, centroids_path = run_clustering(actions_path, out_prefix, n_clusters=64)
    centroids = torch.load(centroids_path)
    n_clusters, T, action_dim = centroids.shape

    env_cfg.scene.num_envs = n_clusters
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="human")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)
    obs = env.reset()

    # 카메라 컨트롤러 준비
    camera_controller = None
    if hasattr(env.unwrapped, "viewport_camera_controller"):
        camera_controller = env.unwrapped.viewport_camera_controller

    import time
    print("엔터를 누르면 다음 클러스터(환경)로 넘어갑니다. q 입력시 종료.")

    cluster_idx = 0
    while simulation_app.is_running():
        # 카메라가 해당 환경을 따라가도록 설정
        if camera_controller is not None:
            camera_controller.set_view_env_index(cluster_idx)
            print(f"[INFO] 카메라가 환경 {cluster_idx+1}번을 따라갑니다.")

        done_flag = False
        t = 0
        while not done_flag and simulation_app.is_running():
            actions = centroids[cluster_idx:cluster_idx+1, t % T, :]
            obs, rewards, dones, infos = env.step(torch.tensor(actions, dtype=torch.float32))
            done_flag = bool(dones[0])
            time.sleep(1 / 120.0)
            t += 1

        print(f"클러스터 {cluster_idx+1}번 에피소드 종료. 엔터=다음, q=종료")
        user_input = input()
        if user_input.strip().lower() == "q":
            break
        cluster_idx += 1
        if cluster_idx >= n_clusters:
            print("모든 클러스터를 확인했습니다.")
            break
        obs = env.reset()

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()