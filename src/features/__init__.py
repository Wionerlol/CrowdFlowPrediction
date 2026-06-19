"""
特征工程代码库 (src.features)

模块结构：

  base          — 共享工具函数（网格分配、Haversine距离、归一化）
  temporal      — 时间周期编码（sin/cos + 节假日标志，12维）
  graphs        — 多模态图构建（G_spatial / G_poi / G_flow，存储/加载）
  node_features — 节点静态特征矩阵组装（41维，NaN=0）

特征工程执行脚本（scripts/ 目录）：

  build_spatial_features.py   → spatial_features_{city}.csv   (N×16)
  build_poi_features.py       → poi_features_{city}.csv        (N×22)
  build_satellite_features.py → satellite_features_{city}.csv  (N×5)
  build_transit_features.py   → transit_features_{city}.csv    (N×6)
  build_subway_features.py    → subway_grid_features_{city}.csv (N×2)
                                 graph_transit_{city}.pt
  build_bus_features.py       → bus_route_features_{city}.csv  (N×1)
  build_temporal_features.py  → temporal_features_{city}.csv   (T×12)
  build_graphs.py             → node_features_{city}.npy (N×41)
                                 graph_{spatial,poi,flow}_{city}.pt

典型用法：

  from src.features.base import assign_to_grid, haversine_km
  from src.features.temporal import build_temporal, load_bj_holidays
  from src.features.graphs import (build_spatial_graph, build_poi_graph,
                                    build_flow_graph, save_graph, load_graph)
  from src.features.node_features import build_node_features, hist_stats
"""

from .base import (
    assign_to_grid,
    haversine_km,
    haversine_matrix,
    minmax,
    zscore,
)
from .temporal import (
    build_temporal,
    load_bj_holidays,
    us_federal_holidays,
)
from .graphs import (
    build_spatial_graph,
    build_poi_graph,
    build_flow_graph,
    save_graph,
    load_graph,
    graph_stats,
)
from .node_features import (
    build_node_features,
    hist_stats,
)

__all__ = [
    # base
    "assign_to_grid", "haversine_km", "haversine_matrix", "minmax", "zscore",
    # temporal
    "build_temporal", "load_bj_holidays", "us_federal_holidays",
    # graphs
    "build_spatial_graph", "build_poi_graph", "build_flow_graph",
    "save_graph", "load_graph", "graph_stats",
    # node_features
    "build_node_features", "hist_stats",
]
