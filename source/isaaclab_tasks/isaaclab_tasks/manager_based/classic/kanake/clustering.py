import os, torch, numpy as np
from sklearn.cluster import KMeans

# ───────── KIT 파일명 찾기 (파일명만 반환) ─────────
def _find_kit(headless=True) -> str:
    fname = "isaacsim.exp.base.python.kit" if headless else "isaacsim.exp.full.kit"
    if os.path.exists(f"_isaac_sim/apps/" + fname):
        return fname
    raise FileNotFoundError(f"{fname} 을 _isaac_sim/apps 에서 찾지 못했습니다.")

# ───────── 클러스터링 ─────────
def run_clustering(actions_path, out_prefix, n_clusters=64):
    seqs = torch.load(actions_path)            # list[Tensor]
    X = np.stack([s.cpu().numpy().reshape(-1) for s in seqs])
    km = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    os.makedirs("cluster_labels", exist_ok=True)
    labels_path    = f"cluster_labels/labels_{out_prefix}.pt"
    centroids_path = f"cluster_labels/centroids_{out_prefix}.pt"
    torch.save(km.labels_, labels_path)

    T, dim = seqs[0].shape
    torch.save(torch.tensor(km.cluster_centers_, dtype=torch.float32)
               .view(n_clusters, T, dim), centroids_path)
    print(f"클러스터 라벨 저장 → {labels_path}")
    return labels_path, centroids_path

# ───────── 클러스터 검토 ─────────
def review_clusters_with_env(centroids, env_cfg, args_cli, sim_app):
    import time, gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    n_clusters, T, _ = centroids.shape
    env_cfg.scene.num_envs = n_clusters
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="human")
    env = RslRlVecEnvWrapper(env)
    env.reset()

    for t in range(T):
        env.step(torch.as_tensor(centroids[:, t, :], dtype=torch.float32))
        time.sleep(0.05)
        if not sim_app.is_running():
            break

    ids = input("Terminate 인덱스(쉼표, 엔터=skip): ")
    terminated = [int(x) for x in ids.split(",") if x.strip().isdigit()]

    env.close()
    return terminated
