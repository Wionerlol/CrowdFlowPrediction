# Design Specification & Content Outline — Urban Crowd Forecasting and Anomaly Alert Internship Defense

## I. Project Information

| Field | Value |
| --- | --- |
| Project Name | Urban Crowd Forecasting, Anomaly Detection and Tiered Alert System |
| Language | English |
| Audience | University internship defense panel |
| Core Message | During a five-month ride-hailing safety algorithm internship, I built a reproducible end-to-end workflow from literature review and multi-source spatiotemporal data engineering to forecasting, anomaly detection, Web delivery, explainability and performance evaluation. |
| Page Count | 32 |
| Delivery Purpose | presentation |
| Content Integrity | Main slides use “anonymized city-grid mobility data” and “public-benchmark prototype validation.” They do not represent public datasets as proprietary internal data. |

## II. Canvas Specification

- Format: PPT 16:9
- Dimensions / viewBox: `1280 × 720`, `0 0 1280 720`
- Safe margin: 52; footer baseline: 688

## III. Visual Theme

- Mode: narrative
- Style: data-journalism
- Palette: city-safety navy, teal and signal orange
- Tone: evidence-led, technically confident, explicit about weak-label and non-causal limitations
- Motif: orange signal trace over a city sensing grid
- No AI-generated imagery

## IV. Typography System

- Font family: `Arial, "Microsoft YaHei", sans-serif`
- Code: `Consolas, "Courier New", monospace`
- Cover 88, page title 56, hero number 64, lead 36, subtitle 34, body 32, annotation 18, footnote 16

## V. Layout Principles

- Top-left titles, top-right page numbers
- One assertion per page; evidence directly beneath or beside it
- Analytical charts remain no-crop with a concise takeaway
- Dense pages use editorial rules and restrained evidence panels; breathing pages use open text and whitespace
- Placeholders appear only on the system slide for the latest dashboard/alert screenshot and a short Web demo video

## VI. Icon Usage Specification

- Library: `tabler-outline`; stroke width: 2
- Uses the same icon inventory as the Chinese deck; no brand icons

## VII. Visualization Reference List

| Page | Reference | Use |
| --- | --- | --- |
| P04 | `gantt_chart` | Retrospective 20-week allocation |
| P05 | `pipeline_with_stages` | End-to-end technical route |
| P12 | `layered_architecture` | Final AGFormer design |
| P15 | `process_flow` | Three-branch anomaly fusion |
| P17 | `layered_architecture` | Pipeline, API, Web and alert system |

Embedded report charts are used on P10, P11, P13, P14, P16, P18 and P19 without redrawing their data geometry.

## VIII. Image Resource List

- `beijing_landsat.webp`: cover and spatial context, crop permitted
- `fig3_1_baseline_multiseed.png`, `fig3_2_spacetimeformer_ablation.png`, `fig3_3_feature_ablation.png`, `fig6_1_final_model_selection.png`, `prediction_horizon_residual.png`: forecasting evidence, no-crop
- `fig5_1_method_comparison.png`, `anomaly_event_metrics.png`: anomaly evidence, no-crop
- `attention_temporal_heatmap.png`, `shap_group_importance.png`: XAI evidence, no-crop
- `pipeline_time_memory.png`, `api_speedup.png`: performance evidence, no-crop
- `week6_web_screenshot.png`: existing prototype, no-crop
- Placeholders: latest dashboard/alert screenshot and 20–30 second Web demo video only

**Image-as-canvas coverage note:** analytical figures and screenshots must remain no-crop, so they are not full-page canvases. The Landsat asset is the only atmospheric canvas and is used full-bleed on P01.

## IX. Content Outline

1. Cover — Multi-source Spatiotemporal Data for Urban Crowd Forecasting and Anomaly Alerts
2. End-to-end outcome — 1,024 nodes; 24h / 48 steps; three anomaly branches; API/Web/tiered alerts
3. Business chain — Anticipate, Detect, Respond
4. Five-month allocation — 3 + 4 + 7 + 3 + 3 weeks, explicitly retrospective
5. End-to-end technical route — governance, features, forecast, anomaly, delivery
6. Literature route — GCN/STGCN/Transformer/Graph WaveNet/Spacetimeformer/VAE/SHAP
7. Data foundation — dynamic flow, static urban semantics, exogenous context; public prototype boundary
8. Gap-aware, chronological split and leakage prevention; 4,022 test windows
9. Feature and graph audit — 41 candidates to 33 leak-free static; spatial + adaptive graphs
10. Baselines — LSTM 19.4941 ± 0.1373 vs AGFormer Time+Static 16.9376 ± 0.3520 MAE
11. Architecture selection — aligned Spacetimeformer still below task-prior AGFormer
12. Final AGFormer — gated context, temporal attention, spatial/adaptive graph, linear residual, four-seed ensemble
13. Strict ablation — successful and negative results
14. Forecast results — h1 MAE 11.3019; 24h MAE 15.9030; 3.82% improvement; horizon uncertainty
15. Anomaly fusion — statistics, reconstruction, prediction residual; weights .70/.20/.10; threshold .56
16. Anomaly results — Level-A timestep F1 .4912, event F1 .4819, 131 events; weak-label boundary
17. System — batch/realtime pipeline, FastAPI, Streamlit, map, 110/15/6 alert tiers, two allowed placeholders
18. XAI — attention entropy .9956; TreeSHAP surrogate R² .9926; flow importance 97.43%; non-causal caveat
19. Performance — FP32 6.416→4.613s / 1.39×; RSS −53.9%; cache 1.723→.038s; FP16 rejected
20. Highlights, limitations and roadmap
21. Paper and technical references
22. Closing — MAE 15.9030, 131 events, 1.39× speedup

## X. Speaker Notes Requirements

- One natural English note per slide, 35–55 seconds each; cover and closing 20–30 seconds
- Explicitly distinguish sourced results, estimated schedule, weak-label metrics and non-causal explanations
- Each note includes a transition to the next slide

## XI. Technical Constraints Reminder

- SVG `viewBox="0 0 1280 720"`; backgrounds use `<rect>`
- Editable `<text>`/`<tspan>` only; no `foreignObject`, `style`, `class`, `mask`, `script`, animation or iframe
- Use palette colors only; no `rgba()` or `<g opacity>`
- Images reference the declared image list and analytical charts use `preserveAspectRatio="xMidYMid meet"`
- Stable top-level group IDs on every page
