# 开发环境配置指南

**项目**: 基于多源时空数据融合的城市级人流异常检测与预警系统  
**适用平台**: Windows 11 (WSL2 Ubuntu) / Linux / macOS  
**Python**: 3.10+  &nbsp;|&nbsp; **PyTorch**: 2.0+  &nbsp;|&nbsp; **CUDA**: 11.8 / 12.1（可选）

---

## 一、前置要求

### 1.1 硬件要求

| 组件 | 最低 | 推荐 |
|------|------|------|
| CPU | 4 核 | 8 核+ |
| 内存 | 16 GB | 32 GB |
| GPU | 无（CPU模式） | NVIDIA 8GB+ 显存（RTX 3060+） |
| 磁盘 | 10 GB 空闲 | 50 GB+（含数据集） |

> GPU 非强制要求。无 GPU 时选择 `cpu` 模式，模型训练会显著变慢，但数据预处理和验证不受影响。

### 1.2 软件前置

**Windows (WSL2) 用户：**
```bash
# 确认 WSL2 已启用（PowerShell 管理员）
wsl --set-default-version 2
wsl --install -d Ubuntu-22.04

# 进入 WSL2 后更新系统
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3 python3-pip python3-venv git curl wget build-essential
```

**Linux / macOS 用户：**
```bash
# Ubuntu/Debian
sudo apt-get install -y python3.10 python3.10-venv python3-pip git curl wget

# macOS (Homebrew)
brew install python@3.10 git curl wget
```

### 1.3 确认 Python 版本

```bash
python3 --version   # 需要 3.10 或以上
# 如显示 3.8/3.9，需升级：
sudo apt-get install -y python3.10 python3.10-venv
```

---

## 二、CUDA 配置（GPU 用户）

> 跳过此节不影响基础功能，CPU 模式同样可以运行所有代码。

### 2.1 确认显卡型号及驱动

```bash
nvidia-smi
```
输出示例：
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12   Driver Version: 525.85.12   CUDA Version: 12.0     |
```

记录 **CUDA Version**（驱动支持的最高版本），据此选择安装参数：

| GPU 型号 | 驱动 CUDA 版本 | 安装参数 |
|---------|--------------|---------|
| RTX 50xx (Blackwell) | ≥ 13.x / 12.8 | `cu128` ← **本机推荐** |
| RTX 40xx / 30xx | ≥ 12.1 | `cu121` |
| RTX 20xx / 10xx | ≥ 11.8 | `cu118` |
| 无 GPU | — | `cpu` |

### 2.2 WSL2 GPU 直通（Windows 用户）

WSL2 通过 Windows 驱动直接使用 GPU，**无需在 WSL2 内单独安装 CUDA**，只需安装 PyTorch 时选择对应版本即可。

---

## 三、一键安装（推荐）

### 3.1 克隆项目并运行配置脚本

```bash
# 进入项目目录
cd /path/to/DidiRemote

# 根据 GPU 情况选择一条命令执行
bash scripts/setup_env.sh cu128   # CUDA 12.8 — RTX 50xx (Blackwell) 推荐 ← 本机
bash scripts/setup_env.sh cu121   # CUDA 12.1 — RTX 40xx / 30xx
bash scripts/setup_env.sh cu118   # CUDA 11.8 — RTX 20xx / 10xx
bash scripts/setup_env.sh cpu     # 无 GPU

# 预计耗时：10~30 分钟（取决于网速）
```

脚本执行以下 7 步：
1. 检查 Python ≥ 3.10
2. 创建虚拟环境 `.venv/`
3. 安装 PyTorch（对应 CUDA 版本）
4. 安装 PyTorch Geometric（图神经网络）
5. 安装 DGL（Deep Graph Library）
6. 安装地理数据处理及深度学习依赖（共 30+ 包）
7. 尝试安装 Flash Attention（可选，加速 Transformer）

### 3.2 激活虚拟环境

安装完成后，**每次开始工作前**都需要激活虚拟环境：

```bash
source .venv/bin/activate
# 命令行前缀会变为 (.venv)，表示已激活

# 退出虚拟环境
deactivate
```

---

## 四、手动安装（备选）

网络不稳定或脚本报错时，可逐步手动安装：

```bash
# 1. 创建虚拟环境
python3 -m venv .venv && source .venv/bin/activate

# 2. PyTorch（选择对应 CUDA 版本）
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# 3. PyTorch Geometric
pip install torch-geometric

# 4. 地理数据 + 核心依赖
pip install geopandas rasterio h5py pyarrow osmium shapely \
            pyproj pandas numpy scipy scikit-learn

# 5. 深度学习辅助
pip install transformers einops optuna shap pyod mlflow

# 6. 工程与可视化
pip install fastapi uvicorn streamlit folium plotly python-dotenv requests
```

---

## 五、环境验证

安装完成后，运行验证脚本确认所有依赖正确安装：

```bash
source .venv/bin/activate
python scripts/verify_env.py
```

**预期输出（全部通过）：**
```
=== CrowdFlowPrediction 环境验证 ===

[ Python ]
  ✓ Python 版本 >= 3.10               3.10.12

[ 深度学习框架 ]
  ✓ PyTorch                           torch 2.0.1 | CUDA 11.8 | GPU: NVIDIA GeForce RTX 3060
  ✓ PyTorch Geometric                 torch_geometric 2.3.1
  ✓ DGL                               dgl 1.1.0
  ✓ Hugging Face Transformers         transformers 4.33.0

[ 地理数据处理 ]
  ✓ GeoPandas                         geopandas 0.14.0
  ✓ Rasterio                          rasterio 1.3.8
  ✓ osmium (OSM parsing)              ok
  ✓ pyproj (坐标系转换)               pyproj 3.6.0

[ 数据文件 ]
  ✓ TaxiBJ 2013                       77.1 MB
  ✓ TaxiBJ 2016                       113.4 MB
  ...

全部通过 (21/21) — 环境就绪！
```

---

## 六、IDE 配置

### VS Code（推荐）

```bash
# 在项目根目录下
code .
```

1. 安装插件：`Python`、`Pylance`、`Jupyter`
2. 选择解释器：`Ctrl+Shift+P` → `Python: Select Interpreter` → 选择 `.venv/bin/python`
3. WSL2 用户需额外安装 `Remote - WSL` 插件

### PyCharm

`File` → `Settings` → `Project` → `Python Interpreter` → `Add` → `Existing environment` → 选择 `.venv/bin/python`

---

## 七、API 密钥配置

项目根目录的 `.env` 文件存储敏感配置（已加入 `.gitignore`，**不要提交到 Git**）：

```bash
# .env 文件内容
OPENWEATHER_API_KEY=your_api_key_here
```

获取方式：[openweathermap.org/api](https://openweathermap.org/api) 免费注册，Key 激活需 10 分钟~2 小时。

---

## 八、常见问题排查

### Q1: `pip install torch` 下载极慢

国内网络访问 PyPI 慢，切换镜像源：
```bash
pip install torch ... -i https://pypi.tuna.tsinghua.edu.cn/simple
# 注意：PyTorch 的 --index-url 参数指定的是 whl 源，不能用 -i 替换，需单独处理
```

或者使用清华 conda 镜像安装 PyTorch：
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 \
    -c pytorch -c nvidia
```

### Q2: `CUDA error: no kernel image is available`

PyTorch CUDA 版本与驱动不匹配。检查 `nvidia-smi` 显示的 CUDA 版本，重新选择正确的安装参数（cu118 / cu121）。

### Q3: `ImportError: libGL.so.1: cannot open shared object file`（WSL2）

```bash
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### Q4: `osmium` 安装失败

```bash
sudo apt-get install -y libosmium-dev
pip install osmium
```

### Q5: `torch_geometric` 安装后 `import` 报错

PyG 的依赖包（torch-scatter 等）需要与 PyTorch 版本严格对应：
```bash
# 确认 torch 版本
python -c "import torch; print(torch.__version__)"  # e.g. 2.0.1+cu118

# 从官方源重装依赖
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

### Q6: Flash Attention 安装失败

Flash Attention 是可选组件，安装失败不影响模型运行，只影响 Transformer 训练速度，忽略即可。

---

## 九、依赖版本锁定

项目核心依赖版本参考：

| 包名 | 推荐版本（本机） | PDF 要求 | 备注 |
|------|---------------|---------|------|
| Python | 3.12.3 ✅ | 3.10+ | 已满足 |
| torch | 2.11.0+cu128 | 2.0+ | RTX 5080 需 cu128 |
| torch-geometric | 2.6.x | any | 匹配 torch 2.11 |
| dgl | 2.x | any | 图神经网络备选 |
| geopandas | 1.x | any | 地理数据处理 |
| h5py | 3.16.0 ✅ | any | 已安装 |
| transformers | 4.x | any | HuggingFace |
| fastapi | 0.128.8 ✅ | any | 已安装 |
| streamlit | 1.x | any | 可视化前端 |
| CUDA | 12.8 (cu128) | 11.8+ | 驱动支持 13.2 |

生成当前环境完整锁定文件：
```bash
source .venv/bin/activate
pip freeze > requirements_lock.txt
```

---

## 十、快速验证（安装后 5 分钟检查）

```python
# 运行此代码片段，确认核心链路可用
import torch
import torch_geometric
import geopandas as gpd
import h5py
import numpy as np

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"PyG: {torch_geometric.__version__}")

# 读取 TaxiBJ 数据
with h5py.File("data/raw/taxibj/BJ13_M32x32_T30_InOut.h5", "r") as f:
    data = f["data"][:]
    print(f"TaxiBJ shape: {data.shape}")  # (4888, 2, 32, 32)

print("环境就绪！")
```
