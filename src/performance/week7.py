"""Reproducible before/after benchmarks for the Week6 production system."""

from __future__ import annotations

import io
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from src.anomaly.prediction_error import (
    build_split_dataset,
    load_model_from_checkpoint,
    predict_ensemble_horizon,
    predict_ensemble_sequence,
)
from src.anomaly.fusion import confidence_from_threshold, fuse_weighted_vote
from src.data.flow_dataset import load_graphs
from src.evaluation.week7 import _MemorySampler, benchmark_api, json_safe, resolve_path


ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run_pipeline_variant(
    name: str,
    base_cfg: dict[str, Any],
    overrides: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    variant_dir = output_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(base_cfg)
    cfg["output_dir"] = str(variant_dir / "orchestration")
    cfg["realtime"] = {**base_cfg.get("realtime", {}), **overrides}
    cfg["realtime"]["output_dir"] = str(variant_dir / "realtime")
    config_path = variant_dir / "config_used.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    log_path = variant_dir / "pipeline.log"
    command = [
        sys.executable, "scripts/run_week6_pipeline.py", "--config", str(config_path),
        "--mode", "realtime", "--device", str(cfg.get("device", "cuda")),
    ]
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        sampler = _MemorySampler(process.pid)
        sampler.start()
        return_code = process.wait()
        sampler.stop()
    wall_seconds = time.perf_counter() - started
    if return_code != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-30:])
        raise RuntimeError(f"Pipeline variant {name} failed ({return_code}):\n{tail}")
    summary_path = variant_dir / "realtime" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stages = {item["name"]: item["duration_seconds"] for item in summary["stages"]}
    return {
        "name": name,
        "wall_seconds": wall_seconds,
        "peak_rss_mib": sampler.peak_bytes / 1024**2,
        "pipeline_seconds": summary["performance"]["pipeline_duration_seconds"],
        "cuda_peak_allocated_mib": summary["performance"]["cuda_peak_allocated_mib"],
        "cuda_peak_reserved_mib": summary["performance"]["cuda_peak_reserved_mib"],
        "model_inference_seconds": stages.get("model_inference"),
        "forecast_inference_seconds": stages.get("forecast_24h_inference"),
        "statistical_seconds": stages.get("statistical_detection"),
        "reconstruction_seconds": stages.get("reconstruction_detection"),
        "statistical_cache_hit": summary["performance"]["statistical_cache_hit"],
        "forecast_reused": summary["performance"]["forecast_reused"],
        "prediction_precision": summary["performance"]["prediction_precision"],
        "prediction_batch_size": summary["performance"]["prediction_batch_size"],
        "prediction_num_workers": summary["performance"]["prediction_num_workers"],
        "scores_path": str(variant_dir / "realtime" / "scores.npz"),
        "forecast_path": str(variant_dir / "realtime" / "forecast_24h.npz"),
        "summary_path": str(summary_path),
        "log_path": str(log_path),
    }


def _compare_npz(reference_path: str | Path, candidate_path: str | Path) -> dict[str, Any]:
    with np.load(reference_path, allow_pickle=False) as reference, np.load(
        candidate_path, allow_pickle=False
    ) as candidate:
        common = sorted(set(reference.files) & set(candidate.files))
        numeric = {}
        exact = {}
        for key in common:
            left, right = reference[key], candidate[key]
            if left.shape != right.shape:
                exact[key] = False
            elif np.issubdtype(left.dtype, np.number):
                diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
                numeric[key] = {
                    "max_abs": float(diff.max(initial=0.0)),
                    "mean_abs": float(diff.mean()) if diff.size else 0.0,
                }
            else:
                exact[key] = bool(np.array_equal(left, right))
        mask_mismatch = None
        if "fusion_mask" in common:
            mask_mismatch = float(np.mean(reference["fusion_mask"] != candidate["fusion_mask"]))
        return {
            "common_arrays": common,
            "numeric": numeric,
            "exact": exact,
            "fusion_mask_mismatch_rate": mask_mismatch,
        }


@torch.inference_mode()
def benchmark_dynamic_int8(tag: str, *, repeats: int = 3) -> dict[str, Any]:
    """Evaluate CPU dynamic INT8 PTQ on one fixed latest test window."""

    device = torch.device("cpu")
    model, cfg, mean, std, checkpoint = load_model_from_checkpoint(tag, device=device)
    dataset, _, _ = build_split_dataset(cfg, "test", mean=mean, std=std)
    x = dataset[-1][0].unsqueeze(0)
    graphs = load_graphs(cfg.get("city", "bj"), cfg.get("graphs", ["spatial"]), device=device)

    def timed(current: torch.nn.Module) -> tuple[np.ndarray, list[float]]:
        current(x, graphs)
        values = []
        output = None
        for _ in range(repeats):
            started = time.perf_counter_ns()
            output = current(x, graphs)
            values.append((time.perf_counter_ns() - started) / 1e6)
        assert output is not None
        return output.float().numpy(), values

    fp32_output, fp32_ms = timed(model)
    try:
        quantized = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False
        )
        int8_output, int8_ms = timed(quantized)
        fp_buffer, int8_buffer = io.BytesIO(), io.BytesIO()
        torch.save(model.state_dict(), fp_buffer)
        torch.save(quantized.state_dict(), int8_buffer)
        difference = np.abs(fp32_output - int8_output)
        return {
            "status": "evaluated",
            "checkpoint": str(checkpoint),
            "scope": "single_model_single_latest_window_cpu_dynamic_linear_int8",
            "fp32_latency_ms_mean": float(np.mean(fp32_ms)),
            "int8_latency_ms_mean": float(np.mean(int8_ms)),
            "speedup": float(np.mean(fp32_ms) / np.mean(int8_ms)),
            "fp32_state_bytes": fp_buffer.tell(),
            "int8_state_bytes": int8_buffer.tell(),
            "size_reduction": 1.0 - int8_buffer.tell() / fp_buffer.tell(),
            "prediction_max_abs_diff": float(difference.max()),
            "prediction_mean_abs_diff": float(difference.mean()),
        }
    except Exception as exc:  # candidate failure must not abort the production benchmark
        return {
            "status": "unsupported",
            "checkpoint": str(checkpoint),
            "error": f"{type(exc).__name__}: {exc}",
        }


def benchmark_inference_batches(
    tags: list[str],
    batch_sizes: list[int],
    *,
    repeats: int = 2,
    device: str = "cuda",
) -> pd.DataFrame:
    """Measure complete four-model sequence inference for safe micro-batch sizes."""

    rows: list[dict[str, Any]] = []
    reference = None
    torch_device = torch.device(device)
    for batch_size in batch_sizes:
        outputs = []
        durations = []
        peaks = []
        for repeat in range(repeats):
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(torch_device)
                torch.cuda.synchronize(torch_device)
            started = time.perf_counter()
            result = predict_ensemble_sequence(
                tags,
                split="test",
                batch_size=int(batch_size),
                num_workers=0,
                device=torch_device,
                max_windows=24,
                window_start=-24,
                precision="fp32",
                materialize_attention_weights=False,
            )
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
                peaks.append(torch.cuda.max_memory_allocated(torch_device) / 1024**2)
            durations.append(time.perf_counter() - started)
            outputs.append(result["pred"])
        if reference is None:
            reference = outputs[0]
        difference = np.abs(reference.astype(np.float64) - outputs[0].astype(np.float64))
        rows.append({
            "batch_size": int(batch_size),
            "repeats": int(repeats),
            "latency_seconds_median": float(np.median(durations)),
            "latency_seconds_min": float(np.min(durations)),
            "latency_seconds_max": float(np.max(durations)),
            "cuda_peak_allocated_mib_max": float(max(peaks)) if peaks else None,
            "max_abs_diff_vs_first_batch": float(difference.max(initial=0.0)),
        })
    return pd.DataFrame(rows)


def validate_fp16_segments(
    validation_cfg: dict[str, Any],
    gates: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Gate FP16 on test periods spread across the complete test timeline."""

    output_dir.mkdir(parents=True, exist_ok=True)
    week5_path = resolve_path(validation_cfg["week5_config"])
    with week5_path.open(encoding="utf-8") as handle:
        week5_cfg = yaml.safe_load(handle) or {}
    pieces = []
    for start, count in validation_cfg["segments"]:
        pieces.append(predict_ensemble_horizon(
            list(week5_cfg["prediction"]["ensemble_tags"]),
            split="test",
            horizon=int(week5_cfg.get("horizon", 1)),
            batch_size=int(validation_cfg.get("batch_size", 8)),
            num_workers=0,
            device="cuda",
            max_windows=int(count),
            window_start=int(start),
            precision="fp16",
            materialize_attention_weights=False,
        ))
    target_indices = np.concatenate([item["target_indices"] for item in pieces])
    pred_fp16 = np.concatenate([item["pred"] for item in pieces])
    true_fp16 = np.concatenate([item["true"] for item in pieces])

    scores_path = resolve_path(validation_cfg["scores"])
    with np.load(scores_path, allow_pickle=False) as scores:
        lookup = {int(index): position for position, index in enumerate(scores["target_indices"])}
        positions = np.asarray([lookup[int(index)] for index in target_indices], dtype=np.int64)
        pred_fp32 = scores["pred_flow"][positions]
        true_fp32 = scores["true_flow"][positions]
        stat_conf = confidence_from_threshold(scores["stat_score"][positions], 1.0)
        recon_conf = scores["recon_score"][positions]
        pred_conf_fp16 = confidence_from_threshold(
            np.abs(pred_fp16 - true_fp16), scores["pred_threshold"]
        )
        fused_fp16 = fuse_weighted_vote(
            stat_conf=stat_conf,
            recon_conf=recon_conf,
            pred_conf=pred_conf_fp16,
            weights=week5_cfg["fusion"]["weights"],
            decision_threshold=float(week5_cfg["fusion"]["decision_threshold"]),
        )
        fusion_mask_fp32 = scores["fusion_mask"][positions]

    difference = np.abs(pred_fp16 - pred_fp32)
    nonzero = np.abs(true_fp32) > 0.0
    fp32_masked_mae = float(np.mean(np.abs(pred_fp32[nonzero] - true_fp32[nonzero])))
    fp16_masked_mae = float(np.mean(np.abs(pred_fp16[nonzero] - true_fp16[nonzero])))
    fp32_unmasked_mae = float(np.mean(np.abs(pred_fp32 - true_fp32)))
    fp16_unmasked_mae = float(np.mean(np.abs(pred_fp16 - true_fp16)))
    mask_mismatch = float(np.mean(fused_fp16["mask"] != fusion_mask_fp32))
    passed = (
        float(difference.mean()) <= float(gates["fp16_max_mean_abs_prediction_diff"])
        and abs(fp16_masked_mae - fp32_masked_mae)
        <= float(validation_cfg["max_mae_delta"])
        and mask_mismatch <= float(gates["fp16_max_mask_mismatch_rate"])
    )
    sample_path = output_dir / "sample_predictions.npz"
    np.savez_compressed(
        sample_path,
        target_indices=target_indices,
        pred_fp32=pred_fp32,
        pred_fp16=pred_fp16,
        true_flow=true_fp16,
        fusion_mask_fp32=fusion_mask_fp32,
        fusion_mask_fp16=fused_fp16["mask"],
    )
    summary = {
        "scope": "stratified_contiguous_test_segments",
        "segments": validation_cfg["segments"],
        "sample_windows": int(len(target_indices)),
        "node_predictions": int(pred_fp16.size),
        "prediction_max_abs_diff": float(difference.max()),
        "prediction_mean_abs_diff": float(difference.mean()),
        "mae_semantics": "masked where abs(true_flow) > 0, matching project metrics",
        "fp32_masked_mae": fp32_masked_mae,
        "fp16_masked_mae": fp16_masked_mae,
        "masked_mae_delta": fp16_masked_mae - fp32_masked_mae,
        "fp32_unmasked_mae_diagnostic": fp32_unmasked_mae,
        "fp16_unmasked_mae_diagnostic": fp16_unmasked_mae,
        "fusion_mask_mismatch_rate": mask_mismatch,
        "passed": passed,
        "quality_gate_candidate_precision": "fp16",
        "quality_gate_passed": passed,
        "week5_config": str(week5_path),
        "week5_scores": str(scores_path),
        "week5_scores_sha256": _file_sha256(scores_path),
        "sample_predictions": str(sample_path),
        "sample_predictions_sha256": _file_sha256(sample_path),
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    summary["summary_sha256"] = _file_sha256(summary_path)
    return summary


def _write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    variants = pd.DataFrame(summary["pipeline_variants"])
    columns = [
        "name", "pipeline_seconds", "model_inference_seconds",
        "forecast_inference_seconds", "statistical_seconds", "peak_rss_mib",
        "cuda_peak_allocated_mib",
    ]
    table_frame = variants[columns]
    table_lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in table_frame.itertuples(index=False, name=None):
        table_lines.append(
            "| " + " | ".join(
                f"{value:.4f}" if isinstance(value, float) else str(value)
                for value in row
            ) + " |"
        )
    table = "\n".join(table_lines)
    optimized = next(item for item in summary["pipeline_variants"] if item["name"] == "optimized_warm_fp32")
    baseline = next(item for item in summary["pipeline_variants"] if item["name"] == "baseline_fp32")
    speedup = baseline["pipeline_seconds"] / optimized["pipeline_seconds"]
    quant = summary["quantization"]
    recommendation = summary["recommendation"]
    fp16_validation = summary["fp16_validation"]
    report = f"""# Week7 系统性能优化报告

## 结论

在同一组 TaxiBJ 最新 24 个有效窗口、同一组 checkpoint 和同一异常阈值下，优化后暖启动流水线端到端加速 **{speedup:.2f}×**。数值一致性与量化采用显式门槛，未通过的候选不会进入生产配置。

## 固定条件

- 推理窗口：测试集末尾 24 个连续有效窗口
- 预测模型：AGFormer gated+linear 四随机种子 ensemble
- 预测目标：h1 异常残差，并输出同一 origin 的 48 步预测
- 对照原则：相同数据、checkpoint、阈值和输出数组；只改变执行策略

## 流水线结果

{table}

主要改动：关闭预测路径未使用的注意力矩阵物化；以实测最优 batch=8 处理 24 个窗口；h1 异常预测与 48 步 Web 预测共享一次模型加载/前向；按数据 SHA256 和规则指纹缓存训练集统计量。批大小扫描见 `{summary.get('batch_sweep_csv')}`。

## 数值一致性

- FP32 优化路径 fusion mask 不一致率：`{summary['comparisons']['optimized_fp32']['fusion_mask_mismatch_rate']}`
- FP32 复用预测的 forecast origin/timestamps/target indices 完全一致，48×1024 预测最大绝对差：`{summary['comparisons']['optimized_forecast_fp32']['numeric']['pred_flow']['max_abs']}`
- FP16 候选 fusion mask 不一致率：`{summary['comparisons']['fp16']['fusion_mask_mismatch_rate']}`
- FP16 是否部署：`{recommendation['deploy_fp16']}`（{recommendation['fp16_reason']}）

补充分层测试集核验覆盖 `{fp16_validation['sample_windows']}` 个窗口（`{fp16_validation['node_predictions']}` 个节点预测）：FP16 相对 FP32 的平均预测差为 `{fp16_validation['prediction_mean_abs_diff']:.6f}`，按项目口径排除真实值为 0 后的 masked MAE 变化为 `{fp16_validation['masked_mae_delta']:.6f}`，融合异常掩码不一致率为 `{fp16_validation['fusion_mask_mismatch_rate']}`，结论为 `{fp16_validation['passed']}`。核验文件及 SHA256 已写入 `summary.json`。

## 模型量化评估

- 状态：`{quant.get('status')}`
- 范围：`{quant.get('scope', 'n/a')}`
- CPU INT8 加速比：`{quant.get('speedup', 'n/a')}`
- 模型 state_dict 体积下降：`{quant.get('size_reduction', 'n/a')}`
- 是否部署：`{recommendation['deploy_dynamic_int8']}`
- CPU 降级路径候选：`{recommendation['dynamic_int8_cpu_fallback_candidate']}`
- 判断：{recommendation['dynamic_int8_reason']}

动态 INT8 仅量化 `Linear`，图邻接乘法与注意力主体仍是 CPU 浮点运算；因此只有在实测延迟和偏差同时通过门槛时才值得部署。当前 API 继续读取原子发布的 NPZ/CSV 快照，不在多个 Web worker 中复制 GPU 模型。

## API 热路径

API 增加了基于源文件 mtime 的派生视图缓存，并将事件质心坐标映射改为向量化 merge。完整结果见 `api_optimized.csv`；历史基线来自 Week7 固定评估产物，来源记录在 `summary.json`。

## 测量边界

- 流水线 variant 各运行一次且顺序固定，适合确认数量级与回归，不应把很小的差异解释为稳定收益。
- `peak_rss_mib` 是主进程与 DataLoader 子进程 RSS 求和，共享页可能重复计算；它用于资源风险筛查，不等同于物理 PSS。
- API 是本机 loopback、单客户端顺序请求；事件查询明显受益，但热力图和完整 48×1024 JSON 仍主要受序列化/压缩限制，单次结果可能小幅波动。
- FP16 和动态 INT8 都是候选路径；只有联合 gate 通过才允许修改生产精度。本轮生产保持 FP32。

## 可复现产物

- `summary.json`：所有耗时、资源、数值偏差和部署判断
- `pipeline_variants.csv`：各执行策略对照
- `api_optimized.csv`：优化后 API 延迟
- 各 variant 目录：实际配置、日志、scores 与 forecast 快照
"""
    (output_dir / "week7_performance_report.md").write_text(report, encoding="utf-8")


def run_performance_optimization(cfg: dict[str, Any], *, config_path: str) -> dict[str, Any]:
    output_dir = resolve_path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with resolve_path(cfg["base_pipeline_config"]).open(encoding="utf-8") as handle:
        base_cfg = yaml.safe_load(handle) or {}
    cache_path = output_dir / "cache" / "statistical_bj_v1.npz"
    common = {"window_start": -24, "window_count": 24, "forecast_window_start": -1}
    specifications = [
        ("baseline_fp32", {
            **common, "prediction_batch_size": 8, "prediction_precision": "fp32",
            "prediction_num_workers": 2,
            "materialize_attention_weights": True, "reuse_forecast_inference": False,
            "statistical_cache": None,
        }),
        ("optimized_cold_fp32", {
            **common, "prediction_batch_size": 8, "prediction_precision": "fp32",
            "prediction_num_workers": 0,
            "materialize_attention_weights": False, "reuse_forecast_inference": True,
            "statistical_cache": str(cache_path),
        }),
        ("optimized_warm_fp32", {
            **common, "prediction_batch_size": 8, "prediction_precision": "fp32",
            "prediction_num_workers": 0,
            "materialize_attention_weights": False, "reuse_forecast_inference": True,
            "statistical_cache": str(cache_path),
        }),
        ("optimized_warm_fp16", {
            **common, "prediction_batch_size": 8, "prediction_precision": "fp16",
            "prediction_num_workers": 0,
            "materialize_attention_weights": False, "reuse_forecast_inference": True,
            "statistical_cache": str(cache_path),
        }),
    ]
    variants = [
        _run_pipeline_variant(name, base_cfg, overrides, output_dir=output_dir)
        for name, overrides in specifications
    ]
    by_name = {item["name"]: item for item in variants}
    comparisons = {
        "optimized_fp32": _compare_npz(
            by_name["baseline_fp32"]["scores_path"],
            by_name["optimized_warm_fp32"]["scores_path"],
        ),
        "fp16": _compare_npz(
            by_name["optimized_warm_fp32"]["scores_path"],
            by_name["optimized_warm_fp16"]["scores_path"],
        ),
        "optimized_forecast_fp32": _compare_npz(
            by_name["baseline_fp32"]["forecast_path"],
            by_name["optimized_warm_fp32"]["forecast_path"],
        ),
    }
    forecast_comparison = comparisons["optimized_forecast_fp32"]
    fp32_execution_ok = (
        comparisons["optimized_fp32"]["fusion_mask_mismatch_rate"] == 0.0
        and all(forecast_comparison["exact"].values())
        and forecast_comparison["numeric"]["target_indices"]["max_abs"] == 0.0
        and forecast_comparison["numeric"]["step_minutes"]["max_abs"] == 0.0
        and forecast_comparison["numeric"]["pred_flow"]["max_abs"]
        <= float(cfg["gates"].get("fp32_max_abs_prediction_diff", 0.001))
    )
    fp16_numeric = comparisons["fp16"]["numeric"].get("pred_flow", {})
    fp16_speedup = (
        by_name["optimized_warm_fp32"]["model_inference_seconds"]
        / by_name["optimized_warm_fp16"]["model_inference_seconds"]
    )
    fp16_realtime_ok = (
        fp16_speedup >= float(cfg["gates"].get("fp16_min_speedup", 1.05))
        and fp16_numeric.get("mean_abs", float("inf"))
        <= float(cfg["gates"].get("fp16_max_mean_abs_prediction_diff", 0.1))
        and comparisons["fp16"]["fusion_mask_mismatch_rate"]
        <= float(cfg["gates"].get("fp16_max_mask_mismatch_rate", 0.001))
    )
    fp16_validation = validate_fp16_segments(
        cfg["fp16_validation"],
        cfg["gates"],
        output_dir=output_dir / "fp16_validation",
    )
    fp16_ok = fp16_realtime_ok and bool(fp16_validation["passed"])
    quantized = benchmark_dynamic_int8(
        str(cfg["quantization"]["checkpoint_tag"]),
        repeats=int(cfg["quantization"].get("repeats", 3)),
    )
    int8_cpu_fallback_ok = (
        quantized.get("status") == "evaluated"
        and quantized.get("speedup", 0.0) >= float(cfg["gates"].get("int8_min_speedup", 1.05))
        and quantized.get("prediction_mean_abs_diff", float("inf"))
        <= float(cfg["gates"].get("int8_max_mean_abs_prediction_diff", 0.1))
    )

    api_cfg = dict(cfg["api_benchmark"])
    api_frame, api_resources = benchmark_api(api_cfg, output_dir=output_dir)
    api_frame.to_csv(output_dir / "api_optimized.csv", index=False)
    baseline_api_path = resolve_path(cfg["api_baseline_csv"])
    baseline_api = pd.read_csv(baseline_api_path)
    api_comparison = baseline_api[["endpoint", "mean_ms", "p95_ms"]].merge(
        api_frame[["endpoint", "mean_ms", "p95_ms"]], on="endpoint",
        suffixes=("_baseline", "_optimized"), validate="one_to_one",
    )
    api_comparison["mean_speedup"] = (
        api_comparison["mean_ms_baseline"] / api_comparison["mean_ms_optimized"]
    )
    api_comparison.to_csv(output_dir / "api_comparison.csv", index=False)
    batch_sweep_source = resolve_path(cfg["batch_sweep_source_csv"])
    batch_sweep_path = output_dir / "batch_sweep.csv"
    pd.read_csv(batch_sweep_source).to_csv(batch_sweep_path, index=False)

    summary = {
        "tag": cfg["tag"],
        "config": config_path,
        "pipeline_variants": variants,
        "comparisons": comparisons,
        "fp16_validation": fp16_validation,
        "quantization": quantized,
        "api": {
            "baseline_csv": str(baseline_api_path),
            "optimized_csv": str(output_dir / "api_optimized.csv"),
            "comparison_csv": str(output_dir / "api_comparison.csv"),
            "resources": api_resources,
        },
        "batch_sweep_csv": str(batch_sweep_path),
        "batch_sweep_source_csv": str(batch_sweep_source),
        "batch_sweep_sha256": _file_sha256(batch_sweep_path),
        "batch_sweep_source_sha256": _file_sha256(batch_sweep_source),
        "recommendation": {
            "deploy_fp32_execution_optimizations": (
                fp32_execution_ok
            ),
            "deploy_fp16": fp16_ok,
            "fp16_realtime_gate_passed": fp16_realtime_ok,
            "fp16_supplemental_gate_passed": bool(fp16_validation["passed"]),
            "fp16_speedup": fp16_speedup,
            "fp16_reason": (
                "passed realtime speed/drift and stratified test-sample gates" if fp16_ok
                else "failed realtime or stratified test-sample gate; keep production fp32"
            ),
            "deploy_dynamic_int8": False,
            "dynamic_int8_cpu_fallback_candidate": int8_cpu_fallback_ok,
            "dynamic_int8_reason": (
                (
                    "CPU fallback microbenchmark passed, but current production inference is CUDA; "
                    "do not replace the GPU ensemble without an end-to-end CPU benchmark"
                ) if int8_cpu_fallback_ok else (
                    "CPU dynamic INT8 failed the latency/output gate and current production "
                    "inference is CUDA; do not deploy"
                )
            ),
        },
    }
    pd.DataFrame(variants).to_csv(output_dir / "pipeline_variants.csv", index=False)
    _write_json(output_dir / "summary.json", summary)
    _write_report(output_dir, summary)
    return summary
