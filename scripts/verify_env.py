"""
环境验证脚本 — 检查所有核心依赖是否正确安装
用法: python scripts/verify_env.py
"""

import sys

RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"

results = []

def check(label, fn, required=True):
    try:
        info = fn()
        results.append((True, label, info))
        print(f"  {GREEN}✓{RESET} {label:<40} {info}")
    except Exception as e:
        tag = "REQUIRED" if required else "optional"
        results.append((not required, label, str(e)))
        color = RED if required else YELLOW
        print(f"  {color}✗{RESET} {label:<40} [{tag}] {e}")

print(f"\n{BOLD}=== CrowdFlowPrediction 环境验证 ==={RESET}\n")

# Python
print(f"{BOLD}[ Python ]{RESET}")
check("Python 版本 >= 3.10",
      lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
              if sys.version_info >= (3, 10) else (_ for _ in ()).throw(RuntimeError("需要 3.10+")))

# PyTorch
print(f"\n{BOLD}[ 深度学习框架 ]{RESET}")
def check_torch():
    import torch
    cuda = f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "CPU only"
    return f"torch {torch.__version__} | {cuda} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}"
check("PyTorch", check_torch)

def check_pyg():
    import torch_geometric
    return f"torch_geometric {torch_geometric.__version__}"
check("PyTorch Geometric", check_pyg)

def check_dgl():
    import dgl
    return f"dgl {dgl.__version__}"
check("DGL", check_dgl, required=False)

def check_transformers():
    import transformers
    return f"transformers {transformers.__version__}"
check("Hugging Face Transformers", check_transformers)

# 地理数据
print(f"\n{BOLD}[ 地理数据处理 ]{RESET}")
def check_geopandas():
    import geopandas
    return f"geopandas {geopandas.__version__}"
check("GeoPandas", check_geopandas)

def check_rasterio():
    import rasterio
    return f"rasterio {rasterio.__version__}"
check("Rasterio", check_rasterio)

def check_osmium():
    import osmium
    return "ok"
check("osmium (OSM parsing)", check_osmium)

def check_pyproj():
    import pyproj
    return f"pyproj {pyproj.__version__}"
check("pyproj (坐标系转换)", check_pyproj)

# 数据科学
print(f"\n{BOLD}[ 数据科学 ]{RESET}")
for lib in [("pandas", "pandas"), ("numpy", "numpy"), ("scipy", "scipy"),
            ("scikit-learn", "sklearn"), ("h5py", "h5py"), ("pyarrow", "pyarrow"),
            ("dask", "dask")]:
    label, mod = lib
    def _check(m=mod):
        mod_obj = __import__(m)
        return getattr(mod_obj, "__version__", "ok")
    check(label, _check)

# 异常检测
print(f"\n{BOLD}[ 异常检测 & 优化 ]{RESET}")
for lib in [("pyod", "pyod"), ("optuna", "optuna"), ("shap", "shap"),
            ("xgboost", "xgboost"), ("catboost", "catboost")]:
    label, mod = lib
    def _check(m=mod):
        mod_obj = __import__(m)
        return getattr(mod_obj, "__version__", "ok")
    check(label, _check)

# 系统工程
print(f"\n{BOLD}[ 系统工程 & 可视化 ]{RESET}")
for lib in [("fastapi", "fastapi"), ("streamlit", "streamlit"),
            ("folium", "folium"), ("plotly", "plotly"),
            ("mlflow", "mlflow"), ("tensorboard", "tensorboard")]:
    label, mod = lib
    def _check(m=mod):
        mod_obj = __import__(m)
        return getattr(mod_obj, "__version__", "ok")
    check(label, _check, required=False)

# 数据文件检查
print(f"\n{BOLD}[ 数据文件 ]{RESET}")
import os
data_files = [
    ("TaxiBJ 2013", "data/raw/taxibj/BJ13_M32x32_T30_InOut.h5"),
    ("TaxiBJ 2016", "data/raw/taxibj/BJ16_M32x32_T30_InOut.h5"),
    ("TaxiBJ 气象", "data/raw/taxibj/BJ_Meteorology.h5"),
    ("METR-LA train", "data/raw/metr_la/metrla_train.parquet"),
    ("METR-LA 邻接矩阵", "data/raw/metr_la/adj_mx.pkl"),
    ("OSM 北京", "data/raw/osm/beijing-latest.osm.pbf"),
]
for label, path in data_files:
    def _check(p=path):
        size = os.path.getsize(p)
        return f"{size/1e6:.1f} MB"
    check(label, _check)

# 汇总
passed = sum(1 for ok, _, _ in results if ok)
total = len(results)
print(f"\n{BOLD}{'='*50}{RESET}")
if passed == total:
    print(f"{GREEN}{BOLD}全部通过 ({passed}/{total}) — 环境就绪！{RESET}")
else:
    failed = [(l, e) for ok, l, e in results if not ok]
    print(f"{YELLOW}通过 {passed}/{total}，以下项目需要关注:{RESET}")
    for label, err in failed:
        print(f"  - {label}: {err}")
print()
