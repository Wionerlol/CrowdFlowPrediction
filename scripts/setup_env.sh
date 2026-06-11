#!/usr/bin/env bash
# 环境一键配置脚本 — CrowdFlowPrediction
# 用法: bash scripts/setup_env.sh [cuda_version]
#
# RTX 50xx (Blackwell) / 驱动 >= 555:   bash scripts/setup_env.sh cu128   ← 推荐
# RTX 40xx / 30xx      / 驱动 >= 525:   bash scripts/setup_env.sh cu121
# RTX 20xx / 10xx      / 驱动 >= 450:   bash scripts/setup_env.sh cu118
# 无 GPU / CPU 模式:                     bash scripts/setup_env.sh cpu

set -e

CUDA_TAG=${1:-cu128}
PYTHON_MIN="3.10"
VENV_DIR=".venv"

# CUDA tag -> PyTorch whl index
case "$CUDA_TAG" in
    cu128) TORCH_IDX="https://download.pytorch.org/whl/cu128" ; TORCH_VER="2.11.0" ;;
    cu121) TORCH_IDX="https://download.pytorch.org/whl/cu121" ; TORCH_VER="2.5.1"  ;;
    cu118) TORCH_IDX="https://download.pytorch.org/whl/cu118" ; TORCH_VER="2.5.1"  ;;
    cpu)   TORCH_IDX="https://download.pytorch.org/whl/cpu"   ; TORCH_VER="2.5.1"  ;;
    *)     echo "未知 CUDA tag: $CUDA_TAG"; echo "可用: cu128 cu121 cu118 cpu"; exit 1 ;;
esac

echo "============================================"
echo " CrowdFlowPrediction 环境配置"
echo " CUDA tag : $CUDA_TAG"
echo " PyTorch  : $TORCH_VER"
echo "============================================"

# 检查 Python 版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "[1/7] Python 版本: $PYTHON_VERSION"
python3 -c "import sys; v=sys.version_info; assert v>=(3,10), f'需要 Python 3.10+, 当前 {v.major}.{v.minor}'"

# 创建虚拟环境
echo "[2/7] 创建虚拟环境 $VENV_DIR ..."
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate
pip install --upgrade pip -q

# 安装 PyTorch
echo "[3/7] 安装 PyTorch $TORCH_VER+$CUDA_TAG ..."
pip install torch==${TORCH_VER} torchvision torchaudio \
    --index-url $TORCH_IDX -q

# 验证 PyTorch CUDA
python3 -c "
import torch
cuda_ok = torch.cuda.is_available()
print(f'  PyTorch {torch.__version__} | CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0) if cuda_ok else \"None\"}')
assert cuda_ok or '$CUDA_TAG' == 'cpu', 'CUDA 不可用，请确认 GPU 驱动和 CUDA tag 匹配'
"

# 安装 PyTorch Geometric
echo "[4/7] 安装 PyTorch Geometric ..."
pip install torch-geometric -q
# pyg 扩展包：根据 torch+cuda 版本选择
PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f $PYG_URL -q 2>/dev/null || \
pip install torch-scatter torch-sparse \
    -f $PYG_URL -q 2>/dev/null || true

# 安装 DGL
echo "[5/7] 安装 DGL ..."
if [ "$CUDA_TAG" = "cpu" ]; then
    pip install dgl -f https://data.dgl.ai/wheels/repo.html -q 2>/dev/null || pip install dgl -q
elif [ "$CUDA_TAG" = "cu128" ]; then
    # DGL cu128 轮子：优先尝试官方，回退到通用
    pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html -q 2>/dev/null || pip install dgl -q
else
    pip install dgl -f https://data.dgl.ai/wheels/torch-2.0/${CUDA_TAG}/repo.html -q 2>/dev/null || pip install dgl -q
fi

# 安装地理 + 数据科学 + 深度学习依赖
echo "[6/7] 安装其他依赖 ..."
pip install -q \
    geopandas \
    rasterio \
    dask \
    pyarrow \
    h5py \
    osmium \
    shapely \
    fiona \
    pyproj \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    xgboost \
    catboost \
    pyod \
    optuna \
    shap \
    transformers \
    einops \
    fastapi \
    uvicorn \
    streamlit \
    folium \
    plotly \
    keplergl \
    mlflow \
    tensorboard \
    matplotlib \
    seaborn \
    tqdm \
    pyyaml \
    python-dotenv \
    requests

# Flash Attention（可选，Blackwell/Ampere 有加速效果）
echo "[7/7] 尝试安装 Flash Attention (可选) ..."
if [ "$CUDA_TAG" != "cpu" ]; then
    pip install flash-attn --no-build-isolation -q 2>/dev/null && \
        echo "  Flash Attention 安装成功" || \
        echo "  Flash Attention 跳过（编译环境不足，非必须）"
fi

echo ""
echo "============================================"
echo " 安装完成！运行验证脚本检查环境："
echo "   source $VENV_DIR/bin/activate"
echo "   python scripts/verify_env.py"
echo "============================================"
