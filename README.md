

---

# Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)

## IsaacSim 설치 방법(4.5.0, CUDA 12)
<pre><code>pip install torch==2.5.1 </code></pre>
<pre><code>torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 pip install --upgrade pip pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com</code></pre>
설치 확인 - 아이작심 실행: 
<pre><code>isaacsim</code></pre>



<<<<<<< HEAD
## IsaacLab 설치 방법
<pre><code>git clone https://github.com/seosangryeong/IsaacLab_snake.git</code></pre>  
<pre><code>sudo apt install cmake build-essential</code></pre>  
디렉토리 접속  
<pre><code>./isaaclab.sh --install</code></pre>  
IsaacLab 설치 확인  
<pre><code>./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py</code></pre>  
=======
Isaac Lab provides developers with a range of essential features for accurate sensor simulation, such as RTX-based cameras, LIDAR, or contact sensors. The framework's GPU acceleration enables users to run complex simulations and computations faster, which is key for iterative processes like reinforcement learning and data-intensive tasks. Moreover, Isaac Lab can run locally or be distributed across the cloud, offering flexibility for large-scale deployments.


## Key Features

Isaac Lab offers a comprehensive set of tools and environments designed to facilitate robot learning:
- **Robots**: A diverse collection of robots, from manipulators, quadrupeds, to humanoids, with 16 commonly available models.
- **Environments**: Ready-to-train implementations of more than 30 environments, which can be trained with popular reinforcement learning frameworks such as RSL RL, SKRL, RL Games, or Stable Baselines. We also support multi-agent reinforcement learning.
- **Physics**: Rigid bodies, articulated systems, deformable objects
- **Sensors**: RGB/depth/segmentation cameras, camera annotations, IMU, contact sensors, ray casters.
>>>>>>> 91ad4944f2b7fad29d52c04a5264a082bcaad71d


## 학습 방법
PPO알고리즘 강화학습 - rsl rl
<pre><code>./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-kanake-v0 --headless</code></pre> 
task는 init에서 설정, headless는 UI없이 학습하는 명령어

학습 이후 play
<pre><code>./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-kanake-v0 --num_envs 1 --checkpoint /home/nuc/IsaacLab/logs/rsl_rl/kanake/2025-06-10_14-10-15/model_550.pt </code></pre> 
num_envs는 병렬화할 환경 개수, checkpoint는 학습한 pt파일이 저장된 경로(isaaclab/logs 에 저장됨)

<<<<<<< HEAD
=======
- [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
- [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
- [Available environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)


## Isaac Sim Version Dependency

Isaac Lab is built on top of Isaac Sim and requires specific versions of Isaac Sim that are compatible with each release of Isaac Lab.
Below, we outline the recent Isaac Lab releases and GitHub branches and their corresponding dependency versions for Isaac Sim.

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

Note that the `feature/isaacsim_5_0` will contain active updates and may contain some breaking changes
until the official Isaac Lab 2.2 release.
It currently requires the [Isaac Sim 5.0 branch](https://github.com/isaac-sim/IsaacSim) available on GitHub built from source.
Please refer to the README in the `feature/isaacsim_5_0` branch for instructions for using Isaac Lab with Isaac Sim 5.0.
We are actively working on introducing backwards compatibility support for Isaac Sim 4.5 for this branch.


## Contributing to Isaac Lab

We wholeheartedly welcome contributions from the community to make this framework mature and useful for everyone.
These may happen as bug reports, feature requests, or code contributions. For details, please check our
[contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Show & Tell: Share Your Inspiration

We encourage you to utilize our [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) area in the
`Discussions` section of this repository. This space is designed for you to:

* Share the tutorials you've created
* Showcase your learning content
* Present exciting projects you've developed

By sharing your work, you'll inspire others and contribute to the collective knowledge
of our community. Your contributions can spark new ideas and collaborations, fostering
innovation in robotics and simulation.

## Troubleshooting

Please see the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for
common fixes or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For issues related to Isaac Sim, we recommend checking its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
or opening a question on its [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

* Please use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussing ideas, asking questions, and requests for new features.
* Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) should only be used to track executable pieces of work with a definite scope and a clear deliverable. These can be fixing bugs, documentation issues, new features, or general updates.

## Connect with the NVIDIA Omniverse Community

Have a project or resource you'd like to share more widely? We'd love to hear from you! Reach out to the
NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com to discuss potential opportunities
for broader dissemination of your work.

Join us in building a vibrant, collaborative ecosystem where creativity and technology intersect. Your
contributions can make a significant impact on the Isaac Lab community and beyond!

## License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). The license files of its dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework. We would appreciate if you would cite it in academic publications as well:

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```
>>>>>>> 91ad4944f2b7fad29d52c04a5264a082bcaad71d
