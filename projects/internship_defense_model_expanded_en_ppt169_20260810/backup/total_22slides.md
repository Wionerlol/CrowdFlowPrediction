# 01_cover

Good morning. My project is an urban crowd forecasting and anomaly alert system built during a ride-hailing safety algorithm internship. The objective was to convert city-grid movement signals into forecasts, anomaly events and actionable alerts. I will start with the complete outcome, then move through the five-month development path.

# 02_closed_loop

This was not an isolated forecasting experiment. I built a reproducible chain across 1,024 spatial nodes, a 24-hour forecast horizon, three complementary anomaly branches, and an API and Web layer with tiered alerts. The next slide explains why the business problem naturally becomes three connected decisions.

# 03_business_questions

Ride-hailing safety requires three decisions: anticipate how each region will change, detect deviations beyond normal behavior, and respond with an event type and alert tier. A model score alone is insufficient; it must retain time, space and evidence so it can become an operational event. The next slide shows how the five-month workload was distributed.

# 04_timeline

This 20-week timeline is a retrospective workload estimate for the defense, not a claim that every task started on an exact calendar boundary. The largest block, seven weeks, went to baselines, AGFormer and Spacetimeformer training, ablation and multi-seed validation. Data engineering and system delivery consumed most of the remaining time. Next, I show the shared technical route behind these stages.

# 05_pipeline

The pipeline has five stages: governance produces valid continuous windows; feature engineering produces aligned model inputs; forecasting produces 24-hour sequences; anomaly fusion produces event objects; and delivery exposes them through APIs, maps and alerts. Continuity, leakage prevention, fair windows, validation-based selection and reproducibility apply across every stage. Next is the literature route.

# 06_research

I studied GCN, STGCN, Transformers, adaptive adjacency, Spacetimeformer, VAE-based detection and SHAP. Instead of copying one paper, I combined explicit spatial priors, learned graph relations, temporal attention and complementary anomaly evidence under one evaluation contract. AGFormer here is a project-specific implementation, not a claimed reproduction of an external namesake model. Next, I describe the data foundation.

# 07_data_foundation

All sources are aligned to 1,024 anonymized city-grid nodes and a half-hour clock. Dynamic channels contain inflow and outflow; static channels describe roads, POIs, public transport and remote-sensing context; exogenous channels cover weather and calendars. Prototype validation uses public benchmarks, while the design is intended for compliant, authorized internal deployment. The next slide covers the main data-quality risk.

# 08_gap_aware

The most serious issue was false temporal continuity. If two segments around a missing interval are concatenated, a complete-looking window contains a nonexistent transition. I reconstructed true timestamps, filtered windows that cross gaps, split chronologically and fitted scaling only on training data. This preserved 4,022 valid test windows. Next, I show how feature and graph candidates were audited.

# 09_features_graphs

The initial feature inventory contained 41 static dimensions. Leakage and usefulness audits reduced this to 33 static features, together with seven time and calendar channels and two flow channels. Four graph candidates were tested, but the final predictor uses an explicit spatial graph plus a learned adaptive graph. The next slide begins the model-selection evidence with unified baselines.

# 10_baselines

Longer history clearly improved recurrent baselines, but temporal modeling alone was not enough. The three-seed gap-aware LSTM averaged 19.4941 MAE, while AGFormer with time and static context reached 16.9376. This shifted the main research path toward temporal attention combined with explicit and adaptive spatial propagation. The next slide shows an important negative architecture result.

# 11_model_selection

I expected Spacetimeformer to outperform the task-specific model, so I aligned local and global attention, value and time embedding, encoder-decoder behavior and cross-attention. It still remained behind. With 1,024 nodes and 48 time steps, global attention requires approximation, while a regular city grid strongly rewards explicit neighborhood priors. The next slide presents the final AGFormer design.

# 12_agformer_architecture

The final model fuses 48-step flow history with calendar and static context through gates. Two layers of four-head temporal attention learn long context. Spatial modeling combines a predefined GCN path with adaptive adjacency learned from node embeddings. A linear residual preserves inertial trends while the nonlinear branch learns deviations, and four random-seed predictions are averaged. The next slide tests each choice through ablation.

# 13_ablation

Every ablation keeps the relevant data rules and comparison windows fixed. The 2-by-2 gate study separates flow, time and static context. Calendar baselines were rerun after repair, and periodic lags used the same 3,433 test windows. Gated context and the linear residual helped, while weather, periodic lags and simple global multi-graph weights did not produce stable gains. Next are the final forecast metrics.

# 14_forecast_results

On the same 4,022 test windows, the four-seed ensemble reached 15.9030 average MAE over 24 hours, a 3.82 percent improvement over the gated baseline. The first-step MAE was 11.3019, while error increased with horizon. Selection was made on validation data, and the ensemble optimized MAE although one single model retained a slightly better RMSE. Next, forecasting residuals become one input to anomaly detection.

# 15_anomaly_fusion

The anomaly framework combines statistical ranges from 3-sigma and IQR, reconstruction error from VAE and Transformer autoencoders trained on weak-normal windows, and prediction residuals from the final AGFormer ensemble. Validation selected weights of 0.70, 0.20 and 0.10 and a threshold of 0.56. Detections are merged by temporal and spatial continuity before classification. The next slide reports performance and limitations.

# 16_anomaly_results

The fused detector produced 131 events and outperformed individual branches under the weak-label protocol. Level-A timestep F1 was 0.4912 and event-level F1 was 0.4819. These numbers must not be presented as real-world event accuracy because authoritative labels are unavailable and statistical rules influence both weak labels and fusion. A 50-event manual sample is only a sanity check. Next, I show how the events are operationalized.

# 17_system

The system wraps each event with time, location, class, severity and evidence fields. Batch and realtime micro-batch modes share preprocessing and inference. FastAPI exposes heatmap, forecast and event queries, while Streamlit provides maps, animation and history. The 131 events are all locatable, and the alert distribution is 110 general, 15 important and 6 urgent. Two placeholders are retained for the latest screenshot and a short demo. Next is explainability.

# 18_xai

Normalized temporal-attention entropy is 0.9956, so aggregate attention does not collapse onto a few lags. A TreeSHAP surrogate reaches 0.9926 R-squared; flow history contributes 97.43 percent and calendar context 2.33 percent. Of 131 events, 82 match weather or calendar candidates. Attention, SHAP and context associations describe contribution and correlation, not causation. The next slide covers performance optimization.

# 19_performance

Optimization was driven by measured bottlenecks. Sharing full-sequence inference and batching reduced warm FP32 runtime from 6.416 to 4.613 seconds, a 1.39-times speedup. Process-tree RSS fell 53.9 percent, and cached statistics dropped from 1.723 to 0.038 seconds. FP16 passed quality checks but did not improve speed, so optimized FP32 remains the deployment choice. Next, I summarize the highlights and limits.

# 20_highlights

The project has three main strengths: trusted data rules, a model validated through ablation and multiple seeds, and an anomaly output that becomes a classified, mapped and tiered operational event. Limitations include public-benchmark prototyping, weak labels, offline micro-batch simulation and missing news or activity ground truth. The next steps are authorized data, human labels, cross-city transfer, Optuna and concurrent streaming. The next slide lists the principal research sources.

# 21_references

The spatial foundation comes from GCN, STGCN and Graph WaveNet; long-context modeling comes from Transformers and Spacetimeformer; anomaly and interpretation work builds on VAE, Anomaly Transformer and SHAP, with Optuna planned for automated tuning. The contribution is the engineering combination and empirical selection under one controlled workflow. I will close with the three measurable outcomes.

# 22_closing

Over five months, I completed the full algorithm-engineering loop from research and data contracts to forecasting, anomaly detection, service delivery and performance evaluation. The final system achieved 15.9030 average MAE over 24 hours, produced 131 fused anomaly events and obtained a 1.39-times end-to-end speedup. My summary is simple: turn urban movement into predictable, explainable and actionable safety signals. Thank you, and I welcome your questions.
