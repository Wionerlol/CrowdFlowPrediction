# Week6 系统部署指南

更新时间：2026-08-09
适用环境：Ubuntu/Linux、Windows 11 WSL2
项目目录示例：`/home/louis/projects/DidiRemote`

## 1. 部署目标与架构

Week6 系统由四个相互隔离的部分组成：

```text
历史 batch pipeline ──► Week5 历史异常 + Week6 历史预警
实时微批 pipeline ───► scores / forecast_24h / events / alerts
                                      │
                                      ▼
FastAPI 只读 artifact 服务 :8000
                                      │
                                      ▼
Streamlit 可视化与预警界面 :8501
```

职责边界：

- pipeline 负责数据校验、模型推理、异常检测、事件构建和预警分级；
- FastAPI 只读取快照，不加载模型、不训练、不占用 GPU；
- Streamlit 通过 HTTP 调用 FastAPI，负责地图、动画、查询和弹窗/声音；
- pipeline 以临时文件加原子替换方式发布 realtime NPZ/CSV/JSON，避免 API 读到半写入文件；
- API 按文件修改时间自动刷新 artifact 缓存，更新数据后通常不需要重启 API。

## 2. 硬件与软件要求

### 2.1 推荐配置

| 组件 | 最低 | 推荐 |
|---|---:|---:|
| CPU | 4 核 | 8 核以上 |
| 内存 | 16 GB | 32 GB |
| GPU | 可选 | NVIDIA 8 GB 以上 |
| 磁盘 | 10 GB 可用空间 | 50 GB 以上 |
| Python | 3.10 | 3.12 |

GPU 只用于首次生成或刷新推理 artifact。若部署机只运行 FastAPI 和 Streamlit，可以不安装 GPU。CPU 也能执行 pipeline，但 AGFormer 四随机种子 ensemble 和重构模型推理会明显更慢。

### 2.2 网络端口

| 端口 | 服务 | 建议暴露范围 |
|---:|---|---|
| 8000 | FastAPI | 本机或内网；不要无认证暴露公网 |
| 8501 | Streamlit | 演示网/内网浏览器 |

## 3. 获取代码与安装环境

```bash
git clone https://github.com/Wionerlol/CrowdFlowPrediction.git
cd CrowdFlowPrediction
```

根据设备选择环境安装方式：

```bash
bash scripts/setup_env.sh cu128   # RTX 50xx / CUDA 12.8
bash scripts/setup_env.sh cu121   # RTX 30xx/40xx
bash scripts/setup_env.sh cu118   # 较旧 NVIDIA GPU
bash scripts/setup_env.sh cpu     # 仅部署 API/UI 或无 GPU
```

激活并验证：

```bash
source .venv/bin/activate
python scripts/verify_env.py
```

所有项目命令都应从仓库根目录执行。直接切换到 `src/visualization` 后运行脚本可能导致相对路径或 `src` 包解析失败。

## 4. 准备数据、模型与 artifact

原始数据和大型 checkpoint 通常不随 Git 仓库分发。执行 realtime pipeline 前至少需要：

```text
data/processed/taxibj_clean.npz
data/processed/node_features_bj.npy
data/processed/graph_spatial_bj.pt
data/processed/grid_meta_bj.csv

outputs/checkpoints/agformer_gate_context_linear_seed42/best.pt
outputs/checkpoints/agformer_gate_context_linear_seed7/best.pt
outputs/checkpoints/agformer_gate_context_linear_seed21/best.pt
outputs/checkpoints/agformer_gate_context_linear_seed123/best.pt

outputs/week5_anomaly/checkpoints/vae.pt
outputs/week5_anomaly/checkpoints/transformer_ae.pt
outputs/week5_anomaly/scores.npz
```

具体 AGFormer checkpoint 内保存的配置决定是否还需要 POI、flow、transit 图或时间/静态特征文件。部署前可让 pipeline 的 artifact 校验和一次 smoke 查询确认文件完整性。

Web/API 还需要以下展示与历史产物：

```text
data/processed/grid_display_meta_bj.csv
src/visualization/assets/beijing_landsat_20150416.webp
src/visualization/assets/beijing_landsat_20150416.json
src/visualization/assets/beijing_ring_roads.geojson
outputs/week5_anomaly/scores.npz
outputs/week6_alerts/alerts.csv
```

展示网格、轻量卫星图和环路 GeoJSON 已作为项目资产提交。仅在原始地理文件变化时重新生成：

```bash
.venv/bin/python scripts/build_week6_display_assets.py
.venv/bin/python scripts/build_week6_satellite_basemap.py
```

上述重建需要项目原始 OSM PBF 和 Landsat GeoTIFF。地图校准只影响 Web 展示，不得用 `grid_display_meta_bj.csv` 替换训练配置中的 `grid_meta_bj.csv`。

## 5. 配置文件

### 5.1 Pipeline：`configs/week6_pipeline.yaml`

重点字段：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `device` | `cuda` | 默认推理设备，可用 CLI 覆盖 |
| `preprocessing.run` | `false` | 默认仅校验已处理数据 |
| `feature_engineering.run` | `false` | 默认仅校验特征 artifact |
| `batch.week5_config` | `configs/week5_anomaly.yaml` | Week5 历史异常配置 |
| `realtime.window_start` | `-24` | 从测试集末尾倒数第 24 个窗口开始 |
| `realtime.window_count` | `24` | 每次实时微批窗口数 |
| `realtime.forecast_window_start` | `-1` | 24h 预测使用最新可用 origin |

realtime 模式强制 reconstruction 为 inference-only；若 VAE/Transformer AE checkpoint 缺失，应停止并补齐 checkpoint，而不是在部署机自动训练。

### 5.2 API：`configs/week6_web.yaml`

重点字段：

```yaml
city: bj
api:
  host: 127.0.0.1
  port: 8000
  cors_origins:
    - http://localhost:8501
    - http://127.0.0.1:8501
artifacts:
  grid_meta: data/processed/grid_display_meta_bj.csv
  realtime_summary: outputs/week6_pipeline/realtime/summary.json
  realtime_scores: outputs/week6_pipeline/realtime/scores.npz
  forecast_24h: outputs/week6_pipeline/realtime/forecast_24h.npz
  realtime_alerts: outputs/week6_pipeline/realtime/alerts.csv
  historical_scores: outputs/week5_anomaly/scores.npz
  historical_alerts: outputs/week6_alerts/alerts.csv
```

修改 artifact 路径或 CORS 后需要重启 FastAPI。仅替换同一路径下的 artifact 内容时，API 会根据 mtime 自动重载。

### 5.3 预警：`configs/week6_alerts.yaml`

Week5 severity 到 Week6 三级预警的默认映射：

| Week5 severity | Week6 等级 | 弹窗 | 声音 |
|---|---|---|---|
| 1–2 | 一般 | 否 | 否 |
| 3 | 重要 | 是 | 否 |
| 4 | 紧急 | 是 | 是 |

修改映射后需重新生成历史预警，并重新运行 realtime pipeline 才能让新规则同时应用于历史和实时事件。

## 6. 首次生成系统产物

### 6.1 历史 batch

若 Week5 异常结果已经存在，仅重用它并生成 Week6 历史预警：

```bash
.venv/bin/python scripts/run_week6_pipeline.py \
  --config configs/week6_pipeline.yaml \
  --mode batch \
  --skip-anomaly
```

如需完整重跑 Week5 异常检测，去掉 `--skip-anomaly`。这可能包含昂贵模型推理；只有 checkpoint 缺失且 Week5 配置允许训练时才可能触发重构模型训练，生产部署前应先核对配置。

主要历史输出：

```text
outputs/week6_alerts/alerts.csv
outputs/week6_alerts/alert_summary.json
outputs/week6_pipeline/batch_summary.json
```

### 6.2 实时微批与 24h 预测

推荐的受控首次运行：

```bash
.venv/bin/python scripts/run_week6_pipeline.py \
  --config configs/week6_pipeline.yaml \
  --mode realtime \
  --window-count 24 \
  --window-start -24 \
  --device cuda
```

无 GPU 时可使用 `--device cpu`。不要同时启动多个 realtime pipeline，它们会竞争 GPU 和同一组输出路径。

主要实时输出：

```text
outputs/week6_pipeline/realtime/summary.json
outputs/week6_pipeline/realtime/scores.npz
outputs/week6_pipeline/realtime/forecast_24h.npz
outputs/week6_pipeline/realtime/events.csv
outputs/week6_pipeline/realtime/alerts.csv
outputs/week6_pipeline/realtime/time_summary.csv
```

确认 `summary.json` 中所有 stages 为 `ok`，并确认 `forecast_24h.npz` 含 48 个 30 分钟预测帧。

## 7. 启动服务

建议打开两个终端，并保持工作目录为项目根目录。

### 7.1 启动 FastAPI

```bash
.venv/bin/python scripts/run_week6_api.py \
  --config configs/week6_web.yaml \
  --host 127.0.0.1 \
  --port 8000
```

开发时可增加 `--reload`；长期运行不建议开启 reload。

验证：

```bash
curl http://127.0.0.1:8000/api/v1/status
curl http://127.0.0.1:8000/docs
```

### 7.2 启动 Streamlit

```bash
.venv/bin/streamlit run src/visualization/week6_app.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --server.headless true
```

浏览器访问：

```text
http://127.0.0.1:8501
```

页面侧边栏默认 API 地址为 `http://127.0.0.1:8000/api/v1`。如果 API 和 Streamlit 分开部署，应在侧边栏填写 Streamlit 服务端可以访问的 API 地址。

### 7.3 局域网演示

API 与 UI 位于同一台机器时：

```bash
.venv/bin/python scripts/run_week6_api.py --host 127.0.0.1 --port 8000

.venv/bin/streamlit run src/visualization/week6_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true
```

浏览器访问 `http://服务器IP:8501`。API 仍只绑定 loopback，由同机 Streamlit 后端访问，攻击面更小。

## 8. 使用 systemd 长期运行

以下示例假设 Linux 用户为 `louis`，项目路径为 `/home/louis/projects/DidiRemote`。按实际环境修改。

### 8.1 FastAPI 服务

创建 `/etc/systemd/system/didiremote-api.service`：

```ini
[Unit]
Description=DidiRemote Week6 FastAPI
After=network.target

[Service]
Type=simple
User=louis
WorkingDirectory=/home/louis/projects/DidiRemote
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/louis/projects/DidiRemote/.venv/bin/python scripts/run_week6_api.py --config configs/week6_web.yaml --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 8.2 Streamlit 服务

创建 `/etc/systemd/system/didiremote-web.service`：

```ini
[Unit]
Description=DidiRemote Week6 Streamlit
After=network.target didiremote-api.service
Requires=didiremote-api.service

[Service]
Type=simple
User=louis
WorkingDirectory=/home/louis/projects/DidiRemote
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/louis/projects/DidiRemote/.venv/bin/streamlit run src/visualization/week6_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

加载并启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now didiremote-api.service
sudo systemctl enable --now didiremote-web.service
sudo systemctl status didiremote-api.service
sudo systemctl status didiremote-web.service
```

查看日志：

```bash
journalctl -u didiremote-api.service -f
journalctl -u didiremote-web.service -f
```

## 9. 定时刷新 realtime artifact

项目当前使用离线微批模拟。若要周期刷新，可建立 oneshot service 和 timer；真实上线时应把该步骤替换为数据接入任务。

创建 `/etc/systemd/system/didiremote-pipeline.service`：

```ini
[Unit]
Description=DidiRemote Week6 realtime micro-batch
After=network.target

[Service]
Type=oneshot
User=louis
WorkingDirectory=/home/louis/projects/DidiRemote
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/louis/projects/DidiRemote/.venv/bin/python scripts/run_week6_pipeline.py --config configs/week6_pipeline.yaml --mode realtime --window-count 24 --window-start -24 --device cuda
```

创建 `/etc/systemd/system/didiremote-pipeline.timer`：

```ini
[Unit]
Description=Refresh DidiRemote realtime artifacts every 30 minutes

[Timer]
OnBootSec=5min
OnUnitActiveSec=30min
Persistent=true

[Install]
WantedBy=timers.target
```

启用：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now didiremote-pipeline.timer
systemctl list-timers didiremote-pipeline.timer
```

在当前离线数据不变化的情况下，定时运行只会重复刷新相同末尾窗口，主要用于演示。不要把它描述成真实在线数据更新。

## 10. 反向代理与安全建议

当前 API 没有身份认证，推荐：

1. FastAPI 保持绑定 `127.0.0.1:8000`；
2. 只向内网开放 Streamlit 8501；
3. 若必须公网访问，在 Nginx/Caddy 后配置 HTTPS、访问控制和认证；
4. 不要把 `.env`、原始 API key、checkpoint 元数据中的敏感路径提交到 Git；
5. 防火墙只开放实际需要的端口；
6. 不要以 root 用户运行 API/UI/pipeline。

FastAPI 使用多 worker 并不会提升 artifact 生成速度，而且每个 worker 有独立缓存。当前轻量只读服务通常使用单 worker 即可。

## 11. 验收检查

### 11.1 自动化测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

### 11.2 API 检查

```bash
curl -s http://127.0.0.1:8000/api/v1/status

curl -s 'http://127.0.0.1:8000/api/v1/heatmap?source=realtime&value_type=observed&row_min=0&row_max=0'

curl -s 'http://127.0.0.1:8000/api/v1/predictions?horizon_steps=2&grid_ids=0,1'

curl -s 'http://127.0.0.1:8000/api/v1/alerts?source=realtime&min_alert_level=2'
```

验收标准：

- `/status` 的 `status` 为 `ok`；
- `forecast_24h` ready，预测响应的 `origin_time` 唯一且帧间隔为 30 分钟；
- 热力图坐标落在展示边界 `116.125–116.625°E、39.671875–40.171875°N`；
- 异常事件能返回事件类型和三级预警字段；
- Streamlit 四个页面均能打开，实时热力图能显示卫星底图和北京环路参照；
- 紧急预警能显示弹窗；声音需先与浏览器页面交互后验证。

## 12. 更新、备份与回滚

更新前备份：

```text
configs/week6_pipeline.yaml
configs/week6_web.yaml
configs/week6_alerts.yaml
outputs/checkpoints/
outputs/week5_anomaly/checkpoints/
outputs/week5_anomaly/scores.npz
outputs/week6_alerts/
outputs/week6_pipeline/realtime/
```

推荐更新步骤：

1. 停止 pipeline timer，避免更新过程中写 artifact；
2. 拉取代码并检查配置差异；
3. 运行完整测试；
4. 手动执行一次 realtime pipeline；
5. 检查 `/api/v1/status`；
6. 重启 API/UI；
7. 确认页面后重新启用 timer。

```bash
sudo systemctl stop didiremote-pipeline.timer
git pull --ff-only
.venv/bin/python -m unittest discover -s tests -v
sudo systemctl restart didiremote-api.service didiremote-web.service
sudo systemctl start didiremote-pipeline.timer
```

回滚代码时也必须确保 artifact 格式与旧代码兼容。最安全做法是同时恢复对应版本的配置和 artifact 备份。

## 13. 常见故障

### 13.1 `ModuleNotFoundError: No module named 'src'`

确认从项目根目录使用提供的入口脚本启动：

```bash
cd /home/louis/projects/DidiRemote
.venv/bin/streamlit run src/visualization/week6_app.py
```

### 13.2 `/status` 为 `degraded` 或业务接口返回 503

查看响应中的 `artifacts`，找到 `ready: false` 的路径。根据缺失类型重新运行 batch/realtime pipeline，或修正 `configs/week6_web.yaml` 路径。

### 13.3 API 可用但页面连接失败

检查：

```bash
curl http://127.0.0.1:8000/api/v1/status
ss -ltnp | grep -E '8000|8501'
```

若 API/UI 分开部署，确认 Streamlit 侧边栏使用的是 Streamlit 服务端可访问地址。

### 13.4 GPU 不可用或显存不足

```bash
nvidia-smi
.venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
```

安全处理顺序：

1. 确认没有重复 pipeline 进程；
2. 保持 `window_count=24`、预测 `batch_size=8`、`num_workers=2`；
3. 必要时暂用 `--device cpu`；
4. 不要在 FastAPI 中直接加载模型。

### 13.5 realtime alerts 为空

这不一定是故障。当前微批窗口可能没有达到融合阈值和最小异常网格数。检查：

```text
outputs/week6_pipeline/realtime/summary.json
outputs/week6_pipeline/realtime/time_summary.csv
outputs/week6_pipeline/realtime/events.csv
```

不要为了页面出现预警而直接降低生产阈值；阈值应基于验证集重新标定。

### 13.6 地图显示但位置不理想

使用 Streamlit 侧边栏“地图空间校准”在限定范围内调节。该操作只移动展示覆盖层，不修改训练数据。二至五环参照线用于人工核对。

### 13.7 浏览器没有播放紧急预警声音

现代浏览器通常禁止页面在首次用户交互前自动播放声音。先点击页面任意控件，再触发或刷新紧急预警；同时确认侧边栏已开启声音。

## 14. 相关文档

- API 详细说明：[Week6 REST API 接口文档](week6_api_reference.md)
- Web 使用说明：[Week6 Web App](week6_web_app.md)
- Pipeline 设计：[Week6 Pipeline 设计](week6_pipeline_design.md)
- 三级预警设计：[Week6 预警设计](week6_alerting_design.md)
- 环境配置：[开发环境配置指南](environment_setup_guide.md)
