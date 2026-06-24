"""
T_in 消融实验：按天 / 周 / 两周 / 月的输入长度对比模型性能。

各时间长度（30分钟/步）：
  1天  = 48步
  1周  = 336步
  2周  = 672步
  1月  = 1440步

batch size 随 T_in 自动缩减（防止显存溢出）：
  T_in ≤ 48   → batch=32
  T_in ≤ 336  → batch=16
  T_in ≤ 672  → batch=8
  T_in ≤ 1440 → batch=4

用法：
  python scripts/ablation_tin.py                        # 全部模型
  python scripts/ablation_tin.py --models lstm gru      # 只跑指定模型
  python scripts/ablation_tin.py --models lstm --dry_run # 打印命令不执行
"""
import argparse
import subprocess
import sys
from pathlib import Path

# T_in 消融配置
TIN_VARIANTS = [
    (48,   "1d",  "1天"),
    (336,  "1w",  "1周"),
    (672,  "2w",  "2周"),
    (1440, "1m",  "1月"),
]

# T_in → 推荐 batch size（按 RTX 5080 16GB 估算）
def auto_batch(t_in: int) -> int:
    if t_in <= 48:   return 32
    if t_in <= 336:  return 16
    if t_in <= 672:  return 8
    return 4

# 模型 → 对应 config 文件
MODEL_CONFIGS = {
    "lstm": "configs/lstm.yaml",
    "gru":  "configs/gru.yaml",
    "gcn":  "configs/gcn.yaml",
    "gat":  "configs/gat.yaml",
}

# ARIMA / Prophet 不用 T_in，跳过
SKIP_MODELS = {"arima", "prophet", "stgcn", "stgcn_multigraph"}


def build_command(model: str, t_in: int, tag: str) -> list[str]:
    cfg   = MODEL_CONFIGS[model]
    batch = auto_batch(t_in)
    return [
        sys.executable, "scripts/train.py",
        "--config", cfg,
        "--t_in",   str(t_in),
        "--batch",  str(batch),
        "--tag",    tag,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",  nargs="+",
                        default=list(MODEL_CONFIGS.keys()),
                        help="要消融的模型列表")
    parser.add_argument("--t_in_list", nargs="+", type=int,
                        default=None,
                        help="自定义 T_in 列表（覆盖默认四档）")
    parser.add_argument("--dry_run", action="store_true",
                        help="只打印命令，不实际执行")
    args = parser.parse_args()

    variants = TIN_VARIANTS
    if args.t_in_list:
        variants = [(t, str(t), f"{t}步") for t in args.t_in_list]

    models = [m for m in args.models if m not in SKIP_MODELS]
    total  = len(models) * len(variants)
    done   = 0

    print(f"消融实验：{len(models)} 个模型 × {len(variants)} 种 T_in = {total} 次训练")
    print(f"模型：{models}")
    print(f"T_in：{[v[0] for v in variants]}\n")
    if args.dry_run:
        print("【dry_run 模式，仅打印命令】\n")

    result_path = Path("outputs/results/ablation_tin.csv")
    result_path.parent.mkdir(parents=True, exist_ok=True)

    for model in models:
        for t_in, tag, label in variants:
            done += 1
            cmd = build_command(model, t_in, tag)
            print(f"[{done}/{total}] {model}  T_in={t_in}（{label}）  batch={auto_batch(t_in)}")
            print(f"  > {' '.join(cmd)}\n")

            if not args.dry_run:
                ret = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
                if ret.returncode != 0:
                    print(f"  ⚠️  {model} T_in={t_in} 训练失败（returncode={ret.returncode}），跳过\n")
                else:
                    print(f"  ✅ 完成\n")

    if not args.dry_run:
        # 从 week3_metrics.csv 中提取本次消融的结果，生成单独汇总
        _summarize(models, variants)


def _summarize(models: list, variants: list):
    """从 week3_metrics.csv 提取消融结果，打印对比表。"""
    import csv
    result_path = Path("outputs/results/week3_metrics.csv")
    if not result_path.exists():
        return

    tags = {v[1] for v in variants}
    rows = []
    with open(result_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("tag") in tags and row.get("horizon") == "avg":
                rows.append(row)

    if not rows:
        return

    print(f"\n{'='*70}")
    print(f"消融汇总（avg horizon，测试集）")
    print(f"{'模型':<8}{'T_in':>6}{'标签':>6}  {'MAE':>8}{'RMSE':>8}{'MAPE':>8}")
    print("-" * 70)

    for model in models:
        for t_in, tag, label in variants:
            matched = [r for r in rows
                       if r["model"] == model and r["tag"] == tag]
            if matched:
                r = matched[-1]
                print(f"{model:<8}{t_in:>6}({label:>3})  "
                      f"{r['MAE']:>8}{r['RMSE']:>8}{r['MAPE']:>7}%")
            else:
                print(f"{model:<8}{t_in:>6}({label:>3})  {'—':>8}")

    # 保存单独汇总 CSV
    abl_path = Path("outputs/results/ablation_tin.csv")
    with open(abl_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "T_in", "label", "MAE", "RMSE", "MAPE"])
        for model in models:
            for t_in, tag, label in variants:
                matched = [r for r in rows
                           if r["model"] == model and r["tag"] == tag]
                if matched:
                    r = matched[-1]
                    w.writerow([model, t_in, label,
                                r["MAE"], r["RMSE"], r["MAPE"]])
    print(f"\n汇总已保存到 {abl_path}")


if __name__ == "__main__":
    main()
