

---

# Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)

## IsaacSim 설치 방법(4.5.0, CUDA 12)
<pre><code> pip install torch==2.5.1 </code></pre>
<pre><code> torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 pip install --upgrade pip pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com </code></pre>
설치 확인 - 아이작심 실행: 
<pre><code> isaacsim  </code></pre>



## IsaacLab 설치 방법
<pre><code> git clone https://github.com/Ryudolf2020/IssacSim_RL_snake_2025.git</code></pre>  
<pre><code> sudo apt install cmake build-essential</code></pre>  
디렉토리 접속  
<pre><code> ./isaaclab.sh --install</code></pre>  
IsaacLab 설치 확인  
<pre><code> ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py</code></pre>  


## Getting Started



