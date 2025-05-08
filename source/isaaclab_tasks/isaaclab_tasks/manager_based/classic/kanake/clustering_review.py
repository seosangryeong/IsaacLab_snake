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
# import omni.isaac.ui




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

    import time
    print("q를 입력하면 반복 재생을 종료합니다.")
    while simulation_app.is_running():
        done_flags = torch.zeros(n_clusters, dtype=torch.bool, device="cpu")
        t = 0
        while not done_flags.all() and simulation_app.is_running():
            actions = centroids[:, t % T, :]
            obs, rewards, dones, infos = env.step(torch.tensor(actions, dtype=torch.float32))
            dones_tensor = torch.as_tensor(dones, device=done_flags.device).bool()
            done_flags |= dones_tensor
            time.sleep(1 / 120.0)
            t += 1
        # 에피소드가 끝나면 반복 재생
        print("에피소드 종료. Enter를 누르면 반복 재생, q 입력시 종료.")
        user_input = input()
        if user_input.strip().lower() == "q":
            break
        obs = env.reset()

    print("UI에서 terminate할 환경을 선택하세요.")
    terminated_str = input("Terminate할 환경 인덱스(쉼표로 구분): ")
    terminated_indices = [int(x) for x in terminated_str.split(",") if x.strip().isdigit()]
    print("Terminated indices:", terminated_indices)

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()