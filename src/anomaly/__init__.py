"""Week 5 anomaly detection package."""

from src.anomaly.fusion import (
    confidence_from_threshold,
    fuse_weighted_vote,
    per_node_quantile_threshold,
    scalar_quantile_threshold,
)
from src.anomaly.evaluation import (
    directional_node_mask,
    evaluate_directional_event_metrics,
    evaluate_directional_timestep_metrics,
    event_level_metrics,
    events_from_time_flags,
    summarize_rows,
)
from src.anomaly.statistical import (
    SlotStats,
    StatisticalResult,
    binary_metrics,
    build_events,
    classify_event,
    extract_clusters,
    fit_slot_stats,
    generate_weak_labels,
    score_statistical,
    slot_indices_from_timestamps,
    summarize_event_types,
    time_flags_from_node_mask,
)

__all__ = [
    "SlotStats",
    "StatisticalResult",
    "binary_metrics",
    "build_events",
    "classify_event",
    "confidence_from_threshold",
    "directional_node_mask",
    "evaluate_directional_event_metrics",
    "evaluate_directional_timestep_metrics",
    "event_level_metrics",
    "events_from_time_flags",
    "extract_clusters",
    "fit_slot_stats",
    "fuse_weighted_vote",
    "generate_weak_labels",
    "per_node_quantile_threshold",
    "scalar_quantile_threshold",
    "score_statistical",
    "slot_indices_from_timestamps",
    "summarize_rows",
    "summarize_event_types",
    "time_flags_from_node_mask",
]
