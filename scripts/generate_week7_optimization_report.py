#!/usr/bin/env python3
"""Generate the Week7 Optuna plan report without launching any training."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_week7_reports import COLORS, REPORT_DIR, configure_plotting, render
from src.optimization.week7 import load_optimization_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "week7_optuna_agformer.yaml"
BASELINE_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "agformer_gate_context_linear_seed42" / "best.pt"
FIGURE_DIR = REPORT_DIR / "week7_figures" / "optimization"


def workflow_figure() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    stages = [
        (0.2, "冻结实验口径", "TaxiBJ 时序切分\n48→48 · 9D+33D\n不访问测试指标", COLORS["blue"]),
        (2.35, "Optuna 搜索", "30 trials / 18h soft\nTPE + MedianPruner\n单 GPU 串行", COLORS["purple"]),
        (4.5, "全预算晋级", "验证集 Top 5\n50 epochs · patience 10\n固定 seed 42", COLORS["orange"]),
        (6.65, "多 seed 复验", "7 / 21 / 42 / 123\n报告均值与标准差\n门槛 ≥0.5%", COLORS["green"]),
        (8.8, "最终测试", "赢家仅评估一次\n同一 4,022 窗口\n决定是否替换模型", COLORS["red"]),
    ]
    for index, (x, title, detail, color) in enumerate(stages):
        box = FancyBboxPatch(
            (x, 0.75), 1.8, 2.65,
            boxstyle="round,pad=0.08,rounding_size=0.1",
            linewidth=1.5, edgecolor=color, facecolor=f"{color}18",
        )
        ax.add_patch(box)
        ax.text(x + 0.9, 2.9, title, ha="center", va="center", fontsize=11, weight="bold", color=color)
        ax.text(x + 0.9, 1.85, detail, ha="center", va="center", fontsize=8.8, linespacing=1.55)
        if index < len(stages) - 1:
            ax.annotate("", xy=(x + 2.12, 2.08), xytext=(x + 1.83, 2.08),
                        arrowprops={"arrowstyle": "->", "lw": 1.6, "color": COLORS["gray"]})
    ax.set_title("Week7 AGFormer 超参数优化决策链", fontsize=15, pad=10)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "optuna_workflow.png", bbox_inches="tight")
    plt.close(fig)


def report_source(cfg: dict, baseline_val: float, baseline_epoch: int) -> str:
    space = cfg["search_space"]
    promotion_threshold = baseline_val * (1 - float(cfg["promotion"]["minimum_relative_val_improvement"]))
    return f"""# Week7 超参数优化方案报告

<div class="meta">AGFormer gated-context + linear residual · Optuna 自动优化设计 · 生成日期 {datetime.now():%Y-%m-%d}</div>

> **方案结论：** 保持 Week4 最终模型的数据规则、特征集合和测试窗口不变，使用 Optuna TPE 在固定验证集上搜索学习率、批次、隐藏维度、注意力头数等 9 组参数。整个 study 的总预算为 **{cfg['n_trials']} trials / 自首次启动起 {cfg['timeout_hours']} 小时软墙钟目标**，单 GPU 串行运行并启用 MedianPruner、早停、显存门禁和 SQLite 状态恢复。搜索与全预算晋级均不构建测试 Dataset；四随机种子确认胜出后，只有显式 `--run-final-test` 才会访问一次测试集。

## 1. 当前基线与优化目标

优化对象是当前最终预测模型，而非重新比较模型家族：

| 项目 | 固定值 |
| --- | --- |
| 模型 | AGFormer，gated 时间/静态上下文，parallel linear residual head |
| 数据 | TaxiBJ，32×32=1,024 节点，真实时间戳，gap-aware 连续窗口 |
| 输入/输出 | 48 个历史半小时步 → 48 个未来半小时步（24 小时） |
| 动态输入 | inflow/outflow + 7 维时间/日历，共 9 维 |
| 静态输入 | 33 维无泄漏节点特征 |
| 图结构 | 单 spatial graph + adaptive adjacency |
| 当前 seed42 基线 | checkpoint epoch {baseline_epoch}，验证 avg MAE **{baseline_val:.4f}** |
| 已知测试参考 | seed42 avg MAE 16.1132；四 seed ensemble avg MAE 15.9030 |

主目标为固定验证集上、全部 48 个预测步和所有非零真实流量位置的反归一化平均绝对误差：

`objective = masked_MAE(pred × std + mean, true × std + mean)`

Optuna 的方向为 `minimize`。搜索阶段额外记录 train loss、参数量、训练时长和峰值显存作为诊断信息，但不组成加权目标；RMSE/MAPE 只在获得明确授权的最终测试中计算，避免改变项目一直使用 MAE 作为主指标的口径。

## 2. 搜索边界

### 2.1 可优化参数

| 参数 | 搜索范围 | 分布/约束 | 设计理由 |
| --- | --- | --- | --- |
| `lr` | {space['lr']['low']}–{space['lr']['high']} | 对数均匀 | 覆盖当前 1e-3 附近一个数量级 |
| `batch` | {space['batch']['choices']} | categorical | 只允许已知安全的 4/8，不冒险并发或 batch 16 |
| `d_model` | {space['d_model']['choices']} | categorical | 同时探索更小与更大表示容量 |
| `n_heads` | {space['n_heads']['choices']} | categorical | 运行时强制 `d_model % n_heads == 0` |
| `d_ff` | `d_model × {space['d_ff_multiplier']['choices']}` | 条件派生 | 保持 FFN 宽度与隐藏维度成比例 |
| `num_layers` | {space['num_layers']['choices']} | categorical | 检查 1–3 层的容量/过拟合权衡 |
| `dropout` | {space['dropout']['low']}–{space['dropout']['high']} | 步长 {space['dropout']['step']} | 调整正则化强度 |
| `weight_decay` | {space['weight_decay']['low']}–{space['weight_decay']['high']} | 对数均匀 | 搜索 AdamW 正则化尺度 |
| `embed_dim` | {space['embed_dim']['choices']} | categorical | 调整 adaptive adjacency 节点嵌入容量 |

### 2.2 明确冻结的项目

城市、时序切分、`T_in/T_out`、真实时间编码、gap filtering、9 维动态输入、33 维静态特征、single spatial graph、gated fusion、linear residual head、Huber loss、AdamW 和指标掩码规则全部冻结。天气、周期滞后、多图、future decoder context 等特征不进入本轮搜索，因为它们会混淆“超参数优化”和“特征/架构消融”。

## 3. 分阶段执行方案

![Optuna workflow](week7_figures/optimization/optuna_workflow.png)
<div class="figure-caption">图1　搜索、晋级、多随机种子和最终测试之间的隔离关系。</div>

### 阶段 A：配置与资源预检

- 默认命令只解析 YAML、检查冻结字段与搜索空间，不加载数据、不创建 study；
- 正式运行必须显式提供 `--run`；
- CUDA 必须可见，study 启动前至少保留 {cfg['resource_guard']['min_free_cuda_mib_before_study']} MiB 空闲显存；
- `n_jobs=1` 被代码强制，禁止多个 trial 同时占用 GPU。

### 阶段 B：Optuna 低预算搜索

- 当前生产参数会作为 incumbent control 主动加入 study，确保搜索结果能在同一代码和预算下直接比较；TPE sampler 的 startup 阶段共 {cfg['sampler']['n_startup_trials']} 个 trials，之后利用历史结果采样；
- 每个 trial 最多 {cfg['max_epochs']} epochs，patience={cfg['patience']}；
- MedianPruner 在至少 {cfg['pruner']['n_warmup_steps']} epochs 后比较中间验证 MAE，且前 {cfg['pruner']['n_startup_trials']} 个 trial 不剪枝；
- 所有 trial 固定 seed={cfg['trial_seed']}，降低随机噪声对结构比较的影响；
- 每轮保存独立 config、`history.jsonl` 和最佳 checkpoint，Optuna 状态保存到 SQLite；重启后继续剩余总预算，而不是重新获得 30 trials/18 小时；
- 中断时处于 RUNNING 的 trial 不原地续训，但会计入总 trial 预算，避免重启突破资源上限；软墙钟预算从 study 首次启动计时并包含停机时间，Optuna 在 trial 之间检查 timeout，正在运行的单个 trial 可能略微越过 18 小时边界；
- study 保存配置、数据和实现文件的 SHA-256 protocol fingerprint；同名数据库若规则漂移将拒绝恢复。

### 阶段 C：全预算晋级

排除主动入队的 incumbent 后，按验证 MAE 选择 Top-{cfg['promotion']['top_k']} challengers，自动导出带 `optimization_validation_only=true` 的 YAML，并额外导出 `incumbent_control.yaml`。这些配置会被通用 `scripts/train.py` 主动拒绝，只能通过专用 `train_optuna_promotion.py --run-promotion` 训练；该入口仅构建 train/validation。候选与 control 都用 seed={cfg['trial_seed']}、{cfg['promotion']['full_epochs']} epochs、patience={cfg['promotion']['full_patience']} 在同一环境重训。只有相对同轮 incumbent control 至少改善 {cfg['promotion']['minimum_relative_val_improvement']:.1%} 才晋级多 seed；按历史 checkpoint 估算，对应验证 MAE 约低于 **{promotion_threshold:.4f}**，正式判断以同轮 control 为准。

### 阶段 D：稳健性确认与最终测试

晋级配置使用 seeds {cfg['promotion']['confirmation_seeds']} 复验，报告验证 MAE 的 mean±std。确认收益稳定后，使用独立 `evaluate_optuna_final.py --run-final-test` 门禁，才会构建与现有模型完全相同的 4,022 个测试窗口并评估 checkpoint；同一 checkpoint 已存在 `final_test_metrics.json` 时会拒绝重复测试。若验证收益不足 0.5%、跨 seed 方差覆盖收益，或最终测试 MAE 未改善，则保留现有最终模型。

## 4. 资源保护与故障恢复

| 风险 | 代码措施 | 结果处理 |
| --- | --- | --- |
| GPU OOM | batch 仅 4/8、参数量上限 {cfg['resource_guard']['max_parameter_millions']}M、trial 后清 CUDA cache | OOM 标记为 pruned，不中断整个 study |
| Windows/WSL 重启 | SQLite 状态 + 首次启动时间 + 已消费 trial 计数 | 继续剩余 study 总预算；中断 RUNNING trial 计入预算 |
| 配置或数据漂移 | config/data/code SHA-256 protocol fingerprint | 同名 study 指纹不一致时拒绝恢复 |
| checkpoint 覆盖 | `trial_XXXX/best.pt` 独立路径 | 非最佳 trial 仅删除大 checkpoint，保留 config/history/SQLite |
| 测试泄漏 | objective 只构建 train/validation Dataset | trial 排名中没有 test 指标 |
| 参数不可用 | 配置加载时校验冻结字段、batch 白名单与 head 整除关系 | 在占用 GPU 前失败 |
| 主机卡死 | 单 trial、2 workers、{cfg['timeout_hours']}h soft timeout | 不启用并行 GPU 搜索；单个在途 trial 允许收尾 |

## 5. 代码与产物设计

| 文件 | 作用 |
| --- | --- |
| `configs/week7_optuna_agformer.yaml` | 搜索空间、预算、剪枝、资源和晋级规则 |
| `scripts/optimize_agformer_optuna.py` | 安全 CLI；无 `--run` 时只校验 |
| `src/optimization/week7.py` | 数据冻结、Optuna objective、验证 MAE、SQLite study 与配置导出 |
| `scripts/train_optuna_promotion.py` | Top-5/control 专用 validation-only 全预算训练入口 |
| `scripts/evaluate_optuna_final.py` | 需要 `--run-final-test` 的独立测试集门禁 |
| `tests/test_week7_optimization.py` | 不训练的安全性与条件搜索测试 |

正式搜索完成后，预期产物位于 `outputs/week7_optimization/week7_agformer_optuna_v1/`：

- `study.db`：带总预算和 protocol fingerprint 的可恢复 Optuna study；
- `trials.csv`：所有 trial 的参数、状态和目标值；
- `trials/trial_XXXX/config.yaml` 与 `history.jsonl`：逐 trial 留痕；
- `best_config.yaml`：排名第一的 50-epoch 晋级配置；
- `promotion/configs/rank*_trial*.yaml`：Top-{cfg['promotion']['top_k']} 晋级配置；
- `promotion/configs/incumbent_control.yaml`：与晋级候选同预算复跑的当前生产参数；
- `summary.json`：最佳 trial、验证 MAE、候选列表与“搜索未访问测试集”声明。

## 6. 预定命令（本次未执行训练）

仅校验配置：

```bash
.venv/bin/python scripts/optimize_agformer_optuna.py \\
  --config configs/week7_optuna_agformer.yaml
```

未来获得明确授权后启动搜索：

```bash
.venv/bin/python -u scripts/optimize_agformer_optuna.py \\
  --config configs/week7_optuna_agformer.yaml --run
```

搜索结束后，晋级配置仍默认只校验；全预算验证训练需要单独授权：

```bash
.venv/bin/python -u scripts/train_optuna_promotion.py \\
  --candidate-config outputs/week7_optimization/week7_agformer_optuna_v1/promotion/configs/rank1_trialXXXX.yaml \\
  --run-promotion
```

只有完成多 seed 决策后才执行最终测试门禁：

```bash
.venv/bin/python scripts/evaluate_optuna_final.py \\
  --candidate-config outputs/week7_optimization/week7_agformer_optuna_v1/promotion/configs/rank1_trialXXXX.yaml \\
  --run-final-test
```

代码不在后台自行启动，不包含自动轮询，也不会因生成本报告而创建 study 或 checkpoint。

## 7. 评估边界与决策规则

1. 首轮 trial 固定一个随机种子，TPE 最优值存在 selection bias；四 seed 复验是必须步骤；
2. 30 trials 对 9 组混合参数不是穷举，只是在单机预算下寻找局部更优区域；
3. MedianPruner 可能提前淘汰慢热配置，Top-{cfg['promotion']['top_k']} 全预算阶段用于缓解该问题；
4. batch size 会改变梯度噪声，不能只按吞吐量选择；所有候选仍以验证 MAE 排名；
5. 最终替换模型必须同时满足：数据规则一致、验证改善达到门槛、四 seed 结果稳定、相同测试窗口 MAE 改善；
6. 本文是**可执行方案**而不是优化结果报告；截至生成时没有运行任何 Optuna trial，因此不报告虚构的 best params 或性能提升。
"""


def main() -> None:
    cfg = load_optimization_config(CONFIG_PATH)
    checkpoint = torch.load(BASELINE_CHECKPOINT, map_location="cpu", weights_only=False)
    baseline_val = float(checkpoint["val_mae"])
    baseline_epoch = int(checkpoint["epoch"])
    configure_plotting()
    workflow_figure()
    pages = render(
        "week7_hyperparameter_optimization_plan",
        "Week7 超参数优化方案报告",
        report_source(cfg, baseline_val, baseline_epoch),
    )
    print(f"Rendered outputs/reports/week7_hyperparameter_optimization_plan.pdf ({pages} pages)")
    print("No Optuna study or training trial was started.")


if __name__ == "__main__":
    main()
