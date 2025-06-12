

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



## IsaacLab 설치 방법
<pre><code>git clone https://github.com/seosangryeong/IsaacLab_snake.git</code></pre>  
<pre><code>sudo apt install cmake build-essential</code></pre>  
디렉토리 접속  
<pre><code>./isaaclab.sh --install</code></pre>  
IsaacLab 설치 확인  
<pre><code>./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py</code></pre>  


## 학습 방법
PPO알고리즘 강화학습 - rsl rl
<pre><code>./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-kanake-v0 --headless</code></pre> 
task는 init에서 설정, headless는 UI없이 학습하는 명령어

학습 이후 play
<pre><code>./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-kanake-v0 --num_envs 1 --checkpoint /home/nuc/IsaacLab/logs/rsl_rl/kanake/2025-06-10_14-10-15/model_550.pt </code></pre> 
num_envs는 병렬화할 환경 개수, checkpoint는 학습한 pt파일이 저장된 경로(isaaclab/logs 에 저장됨)

