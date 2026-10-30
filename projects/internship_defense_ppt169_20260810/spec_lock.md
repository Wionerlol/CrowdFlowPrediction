# Execution Lock

## canvas
- viewBox: 0 0 1280 720
- format: PPT 16:9

## mode
- mode: narrative

## visual_style
- visual_style: data-journalism

## colors
- bg: #F7FAFC
- secondary_bg: #EAF0F5
- surface: #FFFFFF
- primary: #103B5C
- accent: #F26A2E
- secondary_accent: #2A7F9E
- text: #17212B
- text_secondary: #526372
- border: #C9D5DF
- grid: #DDE5EC
- success: #2E7D5B
- warning: #C97A11
- danger: #C73E3A

## typography
- font_family: "Microsoft YaHei", Arial, sans-serif
- code_family: Consolas, "Courier New", monospace
- body: 32
- title: 56
- cover_title: 88
- hero_number: 64
- lead: 36
- subtitle: 34
- subheading: 34
- annotation: 18
- chart_annotation: 18
- footnote: 16

## icons
- library: tabler-outline
- stroke_width: 2
- inventory: activity, alert-triangle, api, brain, calendar, chart-bar, chart-line, check, cloud, database, device-desktop, file-analytics, filter, git-branch, graph, map, map-pin, network, report-analytics, route, server, shield-check, stack, target, timeline, train, users, world, zoom-check

## images
- beijing_landsat: images/beijing_landsat.webp
- baseline_multiseed: images/fig3_1_baseline_multiseed.png | no-crop
- spacetimeformer_ablation: images/fig3_2_spacetimeformer_ablation.png | no-crop
- feature_ablation: images/fig3_3_feature_ablation.png | no-crop
- final_model_selection: images/fig6_1_final_model_selection.png | no-crop
- prediction_horizon: images/prediction_horizon_residual.png | no-crop
- anomaly_method_comparison: images/fig5_1_method_comparison.png | no-crop
- anomaly_event_metrics: images/anomaly_event_metrics.png | no-crop
- web_screenshot: images/week6_web_screenshot.png | no-crop
- attention_temporal: images/attention_temporal_heatmap.png | no-crop
- shap_group: images/shap_group_importance.png | no-crop
- pipeline_performance: images/pipeline_time_memory.png | no-crop
- api_speedup: images/api_speedup.png | no-crop

## page_rhythm
- P01: anchor
- P02: breathing
- P03: dense
- P04: dense
- P05: dense
- P06: dense
- P07: breathing
- P08: dense
- P09: dense
- P10: dense
- P11: dense
- P12: dense
- P13: dense
- P14: dense
- P15: dense
- P16: dense
- P17: dense
- P18: dense
- P19: dense
- P20: breathing
- P21: dense
- P22: anchor

## page_charts
- P04: gantt_chart
- P05: pipeline_with_stages
- P12: layered_architecture
- P15: process_flow
- P17: layered_architecture

## forbidden
- Mixing icon libraries
- rgba()
- `<style>`, `class`, `<foreignObject>`, `textPath`, `@font-face`, `<animate*>`, `<script>`, `<iframe>`, `<symbol>`+`<use>`
- `<g opacity>`
- HTML named entities; use raw Unicode and XML escaping
