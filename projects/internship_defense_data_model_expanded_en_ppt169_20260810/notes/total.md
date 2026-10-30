## 01_cover

Good morning, professors. My project is titled Multi-Source Spatiotemporal Data for Urban Crowd Forecasting and Anomaly Alerts. I worked in a ride-hailing safety algorithm context, with the goal of turning city-grid mobility signals into forecasts, anomaly events and operational alerts. I will present the work through research, data, modeling, anomaly detection and system delivery.

## 02_closed_loop

This was not only a forecasting-model exercise. I completed a full chain from data to service: 1,024 spatial nodes, a 24-hour forecasting horizon, three anomaly evidence branches, and a FastAPI plus map-based Web system with three alert levels. The next slide translates that technical chain into three connected safety decisions.

## 03_business_questions

The business problem has three stages. Anticipation asks how mobility will change across regions. Detection asks which deviations exceed a normal range. Response requires location, event type and alert severity. A model score is therefore only an intermediate artifact; the final output must be a queryable and explainable event that can trigger an operational action.

## 04_timeline

This 20-week timeline is a retrospective workload estimate. The first seven weeks covered research, cleaning, features and graph construction, with roughly four weeks on the data, feature and graph pipeline itself. Weeks 7 to 14 formed the longest workstream: baseline training, architecture selection, ablation and multi-seed replication. In this defense I use seven data slides for three to four minutes to define nodes and edges, followed by eleven model slides for seven to eight minutes to explain why each module was added and why several reasonable ideas failed.

## 05_pipeline

The end-to-end route has five layers. Data governance produces continuous windows; feature engineering creates aligned model inputs; forecasting generates 48 future half-hour steps; anomaly detection converts scores into event objects; and the API and Web layers expose queries and alerts. The common rules are continuity, leakage control, matched-window comparison, validation-only selection and reproducible outputs.

## 06_research

The literature review created testable hypotheses rather than a list of model names. STGCN motivated explicit graph propagation, Transformer motivated long-context attention, Graph WaveNet motivated learned adjacency, and Spacetimeformer provided a general variable-by-time token formulation. Slides 14 through 23 trace those mechanisms into code, experiments and retention decisions.

## 07_data_pipeline

This slide shows that the model never consumes raw files directly. Multiple flow segments are first consolidated into inflow and outflow channels. Local date-slot strings are converted to real UTC timestamps, followed by duplicate, missingness, range and dtype checks. The standard artifact contains 22,484 half-hour steps over a 32-by-32 anonymous grid. Roads, POIs, transit, earth observation, weather and calendar data must then join this tensor through timestamp or node_id rather than assumed row ordering.

## 08_time_quality

This was the most important data correction. Early windows were based on array position, so adjacent segments could be mistaken for consecutive half hours. A later audit found 73 discontinuities. The Dataset now validates all 95 time differences inside every 48-step history plus 48-step target sample. Train, validation and test remain chronological at seven-to-one-to-two, and scaling statistics are fitted on train only. The test set retains 4,022 valid windows. Early cleaning also interpolated points beyond an absolute z-score of five; that stabilizes forecasting but may suppress genuine anomalies, so a production anomaly system should retain an untouched raw mirror.

## 09_grid_nodes

This slide separates grid topology, node identity and map display coordinates. Rows run north to south, columns west to east, and node_id equals row times 32 plus column. Each node binds a flow series, static features and graph edges. A cell is roughly 1.33 by 1.73 kilometers. Because the explicit graph uses row-column topology, a display-only shift used to align the Web satellite layer does not change node identity or graph convolution. It also cannot repair inaccurate geographic metadata: correct aggregation of roads, POIs and stations still depends on the grid_meta bounds.

## 10_node_features

The full node vector has 41 dimensions. Eleven describe location and accessibility: coordinates, center and ring position, nearest hub and commercial distances, and five road densities. Eleven functional POI categories and five Landsat indicators describe land use. Columns 27 through 34 are flow profiles such as mean, p99, zero rate, peak and weekend ratios, net flow and coefficient of variation. The final six cover subway stations, entrances, bus stops, rail lines, transfer hubs and bus routes. Counts use log-one-plus scaling. Transport POIs were separated because they once dominated the POI inventory and made semantic similarities nearly uniform.

## 11_input_contract

The production model does not use all 41 dimensions. Columns 27 through 34 were computed from the whole flow sequence; keeping them would expose validation and test statistics during training. They are removed at load time, leaving 33 genuinely static node features. Dynamic input contains inflow, outflow and seven real timestamp and calendar fields, giving nine dimensions per node and time. Time and static context are projected separately and injected through a flow-conditioned gate. Twelve weather fields achieved exact 22,484-out-of-22,484 timestamp alignment, but the ablation showed no stable improvement, so they were excluded.

## 12_graph_edges

The four candidate graphs share nodes but encode different relationships. G-spatial connects the eight grid neighbors with Gaussian distance weights and 7,812 directed edges. G-POI applies cosine similarity to eleven functional densities and retains ten neighbors. G-flow keeps fifteen positively correlated inflow neighbors. G-transit matches station names and coordinates to rail lines, connects up to ten station hops and decays weights by hop count. Every artifact stores edge_index, edge_weight and num_nodes. G-flow is data-derived and was not retained; any future reuse should rebuild it strictly from the training period.

## 13_graph_selection

More graphs did not mean better forecasting. The four-graph checkpoint assigned 92.18 percent of its global weight to G-flow, yet its MAE was 16.9569 compared with 16.5350 for the single spatial graph, about 2.55 percent worse. One global set of logits cannot adapt by node and time, G-flow overlaps with the already strong history input, and sparse POI or transit gradients are easily suppressed. The final spatial block therefore combines a DenseGCN over the fixed spatial graph with a dense adaptive adjacency learned from node embeddings E1 and E2. Node features describe what a region is; graph edges determine where messages propagate.

## 14_paper_module_map

This slide maps papers to project mechanisms. STGCN contributed explicit graph convolution and alternating spatiotemporal processing, but I replaced temporal convolution with attention. Graph WaveNet contributed node-embedding adaptive adjacency. Transformer contributed multi-head attention, residual connections, normalization and feed-forward layers. Spacetimeformer acted as a more general challenger with variable-by-time tokens, local and global attention, embeddings and encoder-decoder cross-attention. AGFormer is a project-specific composition, not a claimed reproduction of an external same-name paper.

## 15_experiment_staircase

The experiments progressed through five evidence gates. Gap-aware LSTM reached 19.4941 MAE. Adding explicit and adaptive spatial paths reduced the three-seed AGFormer Time-plus-Static mean to 16.9376. Context gating reached 16.5350 at seed42. A linear residual reduced the matched seed42 result to 16.1132. Finally, prediction-level averaging across four seeds reached 15.9030. The metrics use different seed summaries, so this slide documents the R and D path; strict claims use matched seeds, matched windows or separate multi-seed checks.

## 16_spatial_lineage

The spatial path combines two complementary ideas. STGCN injects a known adjacency matrix into local propagation, while Graph WaveNet learns links that a prior graph may miss through node embeddings. The project therefore keeps a predefined spatial GCN and learns adaptive adjacency. This worked because a 32-by-32 regular grid has stable neighbors: the explicit graph narrows the search space, and the adaptive graph adds non-local dependencies.

## 17_temporal_lineage

The temporal path uses the core Transformer computation without copying the translation architecture. Within each node, 48 half-hour steps form a sequence processed by two layers, four heads and model dimension 32. Residual connections, LayerNorm and feed-forward blocks are retained. A language-style full decoder is intentionally omitted, and temporal and spatial modeling are separated to avoid applying standard global attention to all 1,024 times 48 tokens.

## 18_spacetimeformer_alignment

I evaluated Spacetimeformer through five alignment steps. First, I corrected timestamps, static features and gap-aware windows. Second, I matched AGFormer capacity. Third, I implemented local temporal and linear global attention. Fourth, I added separate value and time embeddings, given or missing embeddings, and context and target tokens. Fifth, I completed the encoder-decoder and local or global cross-attention. The aligned model had about 0.089 million parameters, but its 49,152 context tokens still required approximate attention.

## 19_spacetimeformer_findings

A larger and better-aligned model still did not win. Expanding from 0.030 to 0.082 million parameters changed MAE from 17.9967 to 18.6715, so capacity was not the bottleneck. Disabling future time context improved both MAE and RMSE, suggesting decoder-side future context acted as noise. The explicit regular-grid prior also matched the task better than relearning all relationships as tokens. The useful discovery was the linear output path, which reduced Spacetimeformer MAE from 18.2950 to 17.0688 and was then transferred to AGFormer.

## 20_agformer_architecture

The final AGFormer has four parts. Inputs are 48 flow steps, seven time features and 33 static node features. A context gate makes the contribution of time and static information conditional on the current flow state. The backbone combines a Transformer temporal path, an STGCN-lineage explicit spatial GCN and Graph-WaveNet-style adaptive adjacency. The output adds the linear residual discovered in the Spacetimeformer ablation. Each model has about 0.052 million parameters; the offline path averages four seeds, while real-time inference can fall back to seed42.

## 21_context_ablation

Gating helped, but adding more exogenous features did not guarantee improvement. The gate prevents time and static context from entering every node and period with equal importance. After correcting shifted workdays, Calendar MAE was 17.8542 and the previous apparent gain disappeared. Calendar plus Weather reached 17.8316, only 0.0226 better than calendar alone and still worse than the gated model at 16.5350. Weather effects are sparse and spatially heterogeneous, while strong flow history absorbs part of the signal. The final forecaster keeps time and static context through gates but excludes weather.

## 22_negative_results

Two additional negative results shaped the design. In multi-graph AGFormer, the global softmax assigned 92.18 percent to the flow graph and almost ignored POI and transit, showing that one global scalar cannot express node- or time-specific relevance. For periodic lags, the matched 3,433-window baseline had 16.2443 MAE, while lag 48, lag 336 and delta features produced 17.9614. Daily lags were partly redundant, weekly lags were brittle around holidays and gaps, and deltas amplified shocks. Future work should use node-, head- or layer-level graph routing and a learnable seasonal branch.

## 23_linear_residual

The linear residual was the most reliable structural improvement. The formula is a nonlinear backbone plus a direct linear mapping from 48 past inflow values to 48 future values. It first reduced Spacetimeformer MAE from 18.2950 to 17.0688, then reduced matched seed42 AGFormer MAE from 16.5350 to 16.1132. The linear path handles autoregressive inertia, leaving nonlinear residual structure to the backbone. RMSE moved slightly from 30.1288 to 30.3198, showing that lower average error does not automatically reduce extreme-peak error.

## 24_multiseed_ensemble

To test whether seed42 was accidental, I trained seeds 7, 21 and 123. The four single-model MAEs were 16.1132, 16.9079, 16.4172 and 16.5463, with mean 16.4961 and standard deviation 0.2850. Prediction-level averaging reduced MAE further to 15.9030, with h1 MAE at 11.3019. The offline anomaly pipeline uses all four models to lower the normal residual noise floor, while the real-time API can use seed42 only and trade a small amount of accuracy for roughly four times less forward computation.

## 25_anomaly_fusion

Anomaly detection combines three branches. Three-sigma and IQR define statistical normal ranges. VAE and Transformer autoencoders learn weak-normal reconstruction patterns. The prediction branch uses the residual between observations and the final AGFormer forecast. Weights of 0.70, 0.20 and 0.10, together with decision threshold 0.56, were selected on validation data. Detected timesteps are then merged by temporal and spatial continuity and classified into events.

## 26_anomaly_results

Under the weak-label evaluation, fusion outperformed individual branches and produced 131 events. Level-A timestep F1 was 0.4912 and event-level F1 was 0.4819. These values are not real-world event-detection accuracy because authoritative labels are unavailable and the statistical rule contributes to both weak labels and fusion. A manual sample of 50 events provides a sanity check, not an independent ground truth.

## 27_system

The system represents each result as an event object with time, location, type, severity and explanation fields. Batch and real-time micro-batch modes share preprocessing and inference. FastAPI provides heatmap, forecast and event queries, while Streamlit provides the map, forecast animation and historical statistics. All 131 events are locatable; the alert distribution is 110 general, 15 important and 6 urgent. Two placeholders remain for the latest dashboard screenshot and a short Web demo video.

## 28_xai

The normalized temporal-attention entropy is 0.9956, indicating that aggregated attention is not concentrated on only a few lags. A TreeSHAP surrogate achieved R-squared 0.9926; historical flow contributed 97.43 percent and calendar features 2.33 percent. Among 131 events, 82 matched weather or calendar candidate evidence. Attention, SHAP and external context describe association and contribution, not causality.

## 29_performance

Performance optimization began with measurement. Shared full-sequence inference and batching reduced warm FP32 end-to-end time from 6.416 to 4.613 seconds, a 1.39-times speedup. Process-tree RSS fell by 53.9 percent, and statistical caching reduced 1.723 seconds to 0.038 seconds. FP16 passed quality checks but achieved a speed ratio of only 0.9945, so it was not selected for deployment.

## 30_highlights

The project has three main strengths. The data foundation is trustworthy through gap-aware and leakage-free rules. The model process is evidence-driven through paper-to-module mapping, explicit and adaptive graphs, context gating, residual transfer, strict ablation and multi-seed replication. The system is actionable because anomaly scores become classified, mapped and tiered alerts. Limitations include public-benchmark prototyping, weak labels, offline micro-batches and limited news or event ground truth. Next steps are authorized data, human labels, transfer studies, Optuna and true streaming deployment.

## 31_references

This slide lists the main technical sources. The model section directly uses STGCN Figure 2, Graph WaveNet Figure 3, Transformer Figure 1 and Spacetimeformer Figure 3, with authors and venues or arXiv identifiers shown on the relevant slides. The contribution is not claiming invention of these foundations; it is combining and evaluating them under one controlled data and engineering framework.

## 32_closing

Over five months, I completed an algorithm-engineering loop from research questions and data rules to model training, anomaly detection, system services and performance evaluation. The final 24-hour average MAE is 15.9030, the anomaly framework produced 131 fused events, and the optimized pipeline achieved a 1.39-times end-to-end speedup. My final summary is: turning urban mobility into signals that are predictable, explainable and actionable. Thank you, and I welcome your questions.

