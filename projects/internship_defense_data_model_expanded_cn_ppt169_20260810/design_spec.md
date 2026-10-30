# Design Specification & Content Outline — 城市级人流预测与异常预警实习答辩（中文版）

## I. Project Information

| Field | Value |
| --- | --- |
| **Project Name** | 城市级人流预测、异常检测与分级预警系统 |
| **Deck Type** | 五个月算法实习项目答辩 |
| **Language** | 简体中文 |
| **Target Audience** | 学校实习答辩老师；重点判断工作量、技术深度、结果可信度和工程完成度 |
| **Core Message** | 我围绕网约车安全业务场景，完成了从文献调研、多源数据工程、时空预测模型迭代，到异常检测、Web 系统和性能评估的端到端算法项目。 |
| **Page Count** | 32 |
| **Delivery Purpose** | presentation |
| **Content Strategy** | 采用五个月成长叙事重新组织周报；正文称“匿名化城市网格流量数据/公开基准原型验证”，不将公开数据描述为公司内部数据，也不在主体页面反复出现具体基准数据集名称。 |
| **Structure Status** | Strategist-restructured, data-expanded and model-expanded; Slides 7–13 explain cleaning, grid-node mapping, node features, temporal contracts and graph edges; Slides 14–24 trace paper mechanisms, ablations, negative results and final selection. |

## II. Canvas Specification

| Property | Value |
| --- | --- |
| Format | PPT 16:9 |
| Dimensions | 1280 × 720 |
| viewBox | `0 0 1280 720` |
| Safe Margin | 52 |
| Footer Baseline | 688 |

## III. Visual Theme

### Theme Style

- **Mode:** `narrative` — situation → investigation → model tension → engineering resolution → measured outcome.
- **Visual style:** `data-journalism` — clean evidence-led pages, strong chart captions, source lines, restrained panels, and visible uncertainty notes.
- **Tone:** professional, concrete, technically confident, never overstating weak-label anomaly metrics or causal explanations.
- **Recurring motif:** a thin orange “signal trace” crosses key transitions; small grid coordinates and node dots evoke city sensing without decorative clutter.

### Color Scheme

| Role | HEX | Usage |
| --- | --- | --- |
| Background | `#F7FAFC` | Main field |
| Secondary background | `#EAF0F5` | Bands and quiet sections |
| Surface | `#FFFFFF` | Chart and evidence panels |
| Primary | `#103B5C` | Titles, architecture, strong dividers |
| Accent | `#F26A2E` | Key numbers, warnings, selected path |
| Secondary accent | `#2A7F9E` | Data lines and supporting nodes |
| Body text | `#17212B` | Main copy |
| Secondary text | `#526372` | Captions and qualifiers |
| Border | `#C9D5DF` | Panel rules |
| Grid | `#DDE5EC` | Hairlines and chart grid |
| Success | `#2E7D5B` | Passed gates / deployment-ready |
| Warning | `#C97A11` | Evaluation caveats |
| Danger | `#C73E3A` | Urgent alert state |

No AI-generated imagery is used.

## IV. Typography System

### Font Plan

| Role | CJK | Latin / numerals | Character |
| --- | --- | --- | --- |
| Heading | Microsoft YaHei | Arial | Direct, screen-safe |
| Body | Microsoft YaHei | Arial | High legibility in projection |
| Emphasis | Microsoft YaHei | Arial | Bold weight, accent color |
| Code | Consolas | Consolas | Technical snippets only |

- **font_family:** `"Microsoft YaHei", Arial, sans-serif`
- **code_family:** `Consolas, "Courier New", monospace`

### Font Size Hierarchy

All values are unitless px.

| Role | Size |
| --- | ---: |
| Cover title | 88 |
| Page title | 56 |
| Hero number | 64 |
| Lead / core message | 36 |
| Subtitle / subheading | 34 |
| Body | 32 |
| Annotation / chart annotation | 18 |
| Footnote / source | 16 |

## V. Layout Principles

### Page Structure

- Top-left page titles; top-right chapter label and page number.
- One assertion per page; evidence sits beneath or beside it.
- Charts keep their native aspect ratios and include a one-line takeaway, not only a caption.
- Dense pages use rules and whitespace before cards; rounded panels appear only when they group a genuine unit.
- Caveats use a narrow warning band rather than a footnote wall.

### Layout Pattern Library

1. **Evidence hero:** one large number or claim + one supporting chart.
2. **Editorial split:** 38/62 text-to-image or image-to-text, aligned to a 12-column grid.
3. **Process lane:** left-to-right stages with explicit artifacts and validation gates.
4. **Architecture stack:** data → intelligence → service → interface.
5. **Paired evidence:** two comparable plots with a common conclusion strip.
6. **Closing synthesis:** three takeaways + one bounded outlook, no generic “Thank you” page.

### Spacing Specification

| Element | Value |
| --- | ---: |
| Safe margin | 52 |
| Content block gap | 28 |
| Icon-text gap | 12 |
| Card gap | 22 |
| Card padding | 22 |
| Card radius | 10 |

## VI. Icon Usage Specification

### Source

- Library: `tabler-outline`
- Stroke width: `2`
- Brand icons: none

### Recommended Icon List

| Purpose | Icon Path | Pages |
| --- | --- | --- |
| Safety / objective | `tabler-outline/shield-check` | P02–P03 |
| Timeline | `tabler-outline/timeline` | P04 |
| Pipeline | `tabler-outline/route` | P05, P15 |
| Research | `tabler-outline/file-analytics` | P06, P21 |
| Data | `tabler-outline/database` | P07–P09 |
| Graph structure | `tabler-outline/graph` | P09, P12 |
| Model / intelligence | `tabler-outline/brain` | P10–P14 |
| Metrics | `tabler-outline/chart-line` | P10, P14, P16 |
| Alerts | `tabler-outline/alert-triangle` | P15–P17 |
| API / service | `tabler-outline/api` | P17 |
| Web interface | `tabler-outline/device-desktop` | P17 |
| Explainability | `tabler-outline/zoom-check` | P18 |
| Performance | `tabler-outline/activity` | P19 |
| Outlook | `tabler-outline/target` | P20, P22 |

## VII. Visualization Reference List

| Page | Template | Path | Summary-quote | Usage |
| --- | --- | --- | --- | --- |
| P04 | gantt_chart | `templates/charts/gantt_chart.svg` | "Pick for project schedule with 6-12 tasks, durations, and dependencies. Skip for simple milestones without duration (use timeline) or vertical roadmap (use roadmap_vertical)." | Estimated 20-week internship allocation |
| P05 | pipeline_with_stages | `templates/charts/pipeline_with_stages.svg` | "Pick for 3-5 horizontal pipeline stages, each = title + 1-line description + output artifact, connected by arrows (data pipelines, ETL, build pipelines). Skip if any stage lacks an artifact (use process_flow or numbered_steps)." | End-to-end technical route |
| P12 | layered_architecture | `templates/charts/layered_architecture.svg` | "Pick for 3-4 horizontal architecture layers (presentation/service/data), 2-4 module cards per layer, each card = title + 1-line description (description required, even if source brief). Skip if no per-module descriptions (use icon_grid) or no horizontal layering (use module_composition)." | Final AGFormer architecture |
| P15 | process_flow | `templates/charts/process_flow.svg` | "Pick for 3-8 sequential steps connected by simple arrows — approval workflows, customer onboarding, request handling, lifecycle stages. Skip if cyclical (use circular_stages) or stages produce named outputs (use pipeline_with_stages)." | Three-branch anomaly detection and fusion |
| P17 | layered_architecture | `templates/charts/layered_architecture.svg` | "Pick for 3-4 horizontal architecture layers (presentation/service/data), 2-4 module cards per layer, each card = title + 1-line description (description required, even if source brief). Skip if no per-module descriptions (use icon_grid) or no horizontal layering (use module_composition)." | Pipeline, FastAPI, Streamlit and alert UI |

**Runners-up considered**

- `timeline` rejected for P04 because the five-month plan must show durations, not only milestones.
- `numbered_steps` rejected for P05 because every stage produces a named artifact and connector arrows matter.
- `roadmap_vertical` rejected for P04 because horizontal time allocation is the key comparison.
- `client_server_flow` rejected for P17 because the full system contains more than a bilateral client/server exchange.

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Layout pattern | Acquire Via | Status | Reference | text_policy | page_role |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| `beijing_landsat.webp` | 2048×2650 | 0.77 | Cover atmosphere and spatial-data context | Satellite imagery | #1 Full-bleed background with floating title + #30 Flat semi-transparent rectangle overlay + #31 Color-tinted overlay | user | Existing | Project-collected Landsat derivative |  | hero_page |
| `beijing_landsat.webp` | 2048×2650 | 0.77 | Data-source side visual | Satellite imagery | #3 Right-third image + left text body + #21 Rounded rectangle crop + #31 Color-tinted overlay | user | Existing | Same georeferenced image |  | local |
| `fig3_1_baseline_multiseed.png` | 2631×899 | 2.93 | Baseline-to-AGFormer comparison | Data chart | #5 Top-band image + bottom multi-column text + #70 Image with thin colored matte frame | user | Existing | Week4 report figure 3-1 |  | local |
| `fig3_2_spacetimeformer_ablation.png` | 2758×1116 | 2.47 | Spacetimeformer alignment and ablation | Data chart | #5 Top-band image + bottom multi-column text + #70 Image with thin colored matte frame | user | Existing | Week4 report figure 3-2 |  | local |
| `fig3_3_feature_ablation.png` | 2528×986 | 2.56 | Context, graph and lag ablations | Data chart | #5 Top-band image + bottom multi-column text + #70 Image with thin colored matte frame | user | Existing | Week4 report figure 3-3 |  | local |
| `fig6_1_final_model_selection.png` | 2793×1202 | 2.32 | Final model selection | Data chart | #5 Top-band image + bottom multi-column text + #70 Image with thin colored matte frame | user | Existing | Week4 final comparison |  | local |
| `prediction_horizon_residual.png` | 2351×855 | 2.75 | Forecast-horizon and residual evidence | Data chart | #5 Top-band image + bottom multi-column text + #70 Image with thin colored matte frame | user | Existing | Week7 evaluation figure |  | local |
| `fig5_1_method_comparison.png` | 2879×957 | 3.01 | Anomaly detector comparison | Data chart | #5 Top-band image + bottom multi-column text + #70 Image with thin colored matte frame | user | Existing | Week5 report figure 5-1 |  | local |
| `anomaly_event_metrics.png` | 1779×857 | 2.08 | Event-level evaluation | Data chart | #48 Side-by-side comparison (before/after, A/B, then/now) + #70 Image with thin colored matte frame | user | Existing | Week7 event metrics |  | local |
| `week6_web_screenshot.png` | 483×579 | 0.83 | Existing prototype interface evidence | Screenshot | #3 Right-third image + left text body + #21 Rounded rectangle crop + #70 Image with thin colored matte frame | user | Existing | User-provided project screenshot |  | local |
| `attention_temporal_heatmap.png` | 2207×923 | 2.39 | Temporal attention evidence | Data chart | #48 Side-by-side comparison (before/after, A/B, then/now) + #70 Image with thin colored matte frame | user | Existing | Week7 XAI figure |  | local |
| `shap_group_importance.png` | 1601×967 | 1.66 | Feature-group importance evidence | Data chart | #48 Side-by-side comparison (before/after, A/B, then/now) + #70 Image with thin colored matte frame | user | Existing | Week7 XAI figure |  | local |
| `pipeline_time_memory.png` | 2485×882 | 2.82 | Pipeline speed and memory | Data chart | #48 Side-by-side comparison (before/after, A/B, then/now) + #70 Image with thin colored matte frame | user | Existing | Week7 performance figure |  | local |
| `api_speedup.png` | 2046×987 | 2.07 | API endpoint speedup | Data chart | #48 Side-by-side comparison (before/after, A/B, then/now) + #70 Image with thin colored matte frame | user | Existing | Week7 performance figure |  | local |
| `PLACEHOLDER_latest_dashboard.png` | TBD | TBD | Latest system home-page and alert-popup screenshot | Placeholder | #48 Side-by-side comparison (before/after, A/B, then/now) + #70 Image with thin colored matte frame | placeholder | Placeholder | User inserts latest screenshot later |  | local |
| `PLACEHOLDER_web_demo_video.mp4` | TBD | 16:9 | Web interaction demonstration | Placeholder | #17 Picture-in-picture inset + #70 Image with thin colored matte frame | placeholder | Placeholder | User inserts short video later |  | local |

All chart and screenshot images are `no-crop`; the satellite asset may be cropped for composition.

**Image-as-canvas coverage note:** image-bearing rows are predominantly analytical charts and screenshots, which must remain no-crop and therefore are intentionally not used as full-page canvases. The only suitable atmospheric asset, `beijing_landsat.webp`, is already used as a full-bleed canvas on P01 and a spatial-context strip on P07.

## IX. Content Outline

### Part 1: 业务背景与五个月任务

#### Slide 01 - Cover

- **Cover impact:** Use the city satellite grid as the world of the problem; a bright orange signal trace cuts from raw movement to a shield-shaped alert point.
- **Layout:** Full-bleed satellite crop with left opaque navy scrim; title floats in the calm left field.
- **Title:** 多源时空数据驱动的城市人流预测与异常预警
- **Subtitle:** 网约车安全方向算法实习项目答辩
- **Info:** 姓名 / 学号 / 指导老师 / 答辩日期（保留可编辑占位）

#### Slide 02 - 五个月，我完成了一条从数据到预警的闭环

- **Layout:** Four large milestones connected by one signal trace; no card grid.
- **Title:** 从数据到预警：完整闭环，而非单点模型
- **Core message:** 项目最终交付了可复现的数据、预测、检测、服务与评估链路。
- **Content:**
  - 1,024 个城市空间节点
  - 未来 24 小时、48 个半小时预测步
  - 统计 + 重构 + 预测误差三路异常融合
  - FastAPI + Streamlit + 三级预警与性能门禁

#### Slide 03 - 业务问题被拆成三个连续决策

- **Layout:** Left role statement, right three vertically linked questions.
- **Title:** 网约车安全场景需要“预知—识别—处置”
- **Core message:** 人流安全不是只预测一个数，而是要把区域、时间、异常类型和预警动作串起来。
- **Content:**
  - 岗位背景：网约车安全方向算法实习生
  - 预知：各区域未来 24 小时人流如何变化？
  - 识别：哪些波动超出正常范围？
  - 处置：异常属于哪一类，应触发一般、重要还是紧急预警？

#### Slide 04 - 20 周工作量主要投入模型迭代与异常检测

- **Layout:** Adapt `gantt_chart`; bottom band shows estimated week allocation.
- **Title:** 五个月实习工作周期与投入分配
- **Core message:** 约一半周期用于模型训练、选型、消融和复验，其余时间完成数据底座与系统落地。
- **Visualization:** gantt_chart
- **Content:**
  - W1–W3：需求分析、论文调研、技术选型（3 周）
  - W3–W7：数据清洗、特征与图结构、pipeline（4 周）
  - W7–W14：基线、AGFormer/Spacetimeformer、消融、多 seed 与优化设计（7 周）
  - W14–W17：三类异常检测、融合、分类、事件级评估（3 周）
  - W17–W20：API/Web/预警、XAI、性能优化、报告与答辩（3 周）
  - 注：该时间轴是按项目工作量回溯估算的五个月答辩口径。

#### Slide 05 - 技术路线围绕同一份时空事实逐层增值

- **Layout:** Adapt `pipeline_with_stages`; each stage names its artifact.
- **Title:** 端到端技术路线
- **Core message:** 统一时间、空间和数据规则，使模型结果能直接进入异常检测和系统服务。
- **Visualization:** pipeline_with_stages
- **Content:**
  - 数据治理 → gap-aware 连续窗口、时序划分、无泄漏缩放
  - 特征工程 → 流量、时间日历、33 维静态节点特征、候选图结构
  - 预测建模 → 基线、AGFormer、Spacetimeformer、残差头与 ensemble
  - 异常检测 → 三路得分、验证集定权、时空事件合并与分类
  - 系统落地 → batch/realtime pipeline、REST API、地图 Web 与三级预警

### Part 2: 调研、数据与图结构

#### Slide 06 - 论文调研决定了“显式图先验 + 全局时间建模”主线

- **Layout:** Horizontal research lineage with six paper anchors and one project decision band.
- **Title:** 从经典时空图到长序列 Transformer
- **Core message:** 文献并未直接给出答案，但帮助我把空间先验、长时依赖和无监督异常检测组合为可验证路线。
- **Content:**
  - STGCN：显式图卷积 + 时间卷积
  - Transformer：全局时序依赖
  - Graph WaveNet：学习自适应邻接
  - Spacetimeformer：将变量×时间展平为 token
  - VAE / Anomaly Transformer：以正常模式或关联差异检测异常
  - SHAP：将黑盒预测拆解为特征贡献
  - 项目 AGFormer 为自主组合实现，不冒充同名外部论文复现。

#### Slide 07 - 多源数据被统一到 1,024 个匿名化城市网格

- **Layout:** Left 62% source constellation, right satellite strip; image is contextual, not proof of exact administrative bounds.
- **Title:** 数据底座：动态流量 + 城市静态语义 + 外生上下文
- **Core message:** 统一网格和半小时时钟，让流量、路网、POI、交通、遥感、天气和日历能够进入同一训练样本。
- **Content:**
  - 动态标签：匿名化网约车流入/流出网格序列
  - 静态空间：路网密度、功能性 POI、公交/地铁、距离与地理位置
  - 遥感：NDVI / NDBI / MNDWI 等土地覆盖指标
  - 外生变量：天气、工作日/周末、节假日与调休
  - 原型验证使用公开基准数据；方案面向合规内部环境迁移。

#### Slide 08 - 最大的数据难点不是缺值，而是时间连续性与泄漏

- **Layout:** Before/after time strip; three validation gates under it.
- **Title:** 数据处理：gap-aware、时序划分、无泄漏
- **Core message:** 不连续历史段若被直接拼接，会制造虚假的连续窗口并污染训练和评估。
- **Content:**
  - 先构建真实时间戳，再过滤跨时间缺口窗口
  - 训练/验证/测试按时间 7:1:2 划分，不随机打乱
  - scaler 只在训练段拟合；日历和天气按真实时间对齐
  - 48 步历史输入预测未来 48 步；测试集保留 4,022 个有效连续窗口
  - 训练 checkpoint 保存配置、数据规则和缩放信息，避免误用旧模型。

#### Slide 09 - 41 维候选静态特征，经审计后保留 33 维进入最终模型

- **Layout:** Feature funnel on left; four candidate graphs on right with final path highlighted.
- **Title:** 特征与图结构：先充分构建，再严格消融
- **Core message:** 多源不等于全部硬塞进模型；最终输入由消融和无泄漏规则共同决定。
- **Content:**
  - 候选静态特征：地理、路网、POI、遥感、公共交通、历史统计，共 41 维
  - 最终模型：33 维无泄漏静态节点特征 + 7 维时间/日历 + 2 维流量
  - 构建四类候选图：空间邻接、POI 相似、历史流量相关、轨道拓扑
  - 多图融合没有超过单图方案；最终预测采用显式空间图 + 学习自适应邻接
  - 经验：外部特征更适合作为门控上下文，而非无条件相加。

### Part 3: 预测模型的训练、选型与创新

#### Slide 10 - 基线揭示：长输入有效，但仅做时序或仅做空间都不够

- **Layout:** Full-width existing chart with three annotated callouts below.
- **Title:** 从 ARIMA、LSTM 到图时空模型
- **Core message:** 48 步历史显著改善 RNN，而真正的提升来自同时建模长时依赖和空间传播。
- **Visualization:** Existing chart image
- **Content:**
  - 统一比较 ARIMA、Prophet、LSTM、GRU、GCN、GAT、STGCN 等基线
  - gap-aware LSTM 三 seed：24h 平均 MAE 19.4941 ± 0.1373
  - AGFormer Time+Static 三 seed：MAE 16.9376 ± 0.3520
  - 由此确定后续主线：Transformer 时间建模 + 显式/自适应图空间建模。

#### Slide 11 - 更通用的 Spacetimeformer 未必更适合规则城市网格

- **Layout:** Existing ablation chart above; bottom two-column “expected / observed” explanation.
- **Title:** 架构选型：通用全局注意力输给了任务先验
- **Core message:** 经过原版式 local/global attention 对齐后，Spacetimeformer 仍未超过带显式图先验的 AGFormer。
- **Visualization:** Existing chart image
- **Content:**
  - 对齐 encoder-decoder、local/global attention、value/time embedding 与 cross-attention
  - 1024×48 token 使全局注意力必须采用线性近似，优化难度更高
  - 规则网格邻域关系强，AGFormer 的空间图先验更稳定
  - 结论不是“Transformer 不好”，而是归纳偏置与数据规模必须匹配。

#### Slide 12 - 最终模型把趋势交给线性头，把复杂残差交给图 Transformer

- **Layout:** Adapt `layered_architecture`; center equation-like flow is editable text.
- **Title:** 最终 AGFormer：门控上下文 + 双图路径 + 线性残差
- **Core message:** 平行分工比单一复杂分支更有效：线性路径守住惯性，非线性路径学习时空偏差。
- **Visualization:** layered_architecture
- **Content:**
  - 输入层：48 步 inflow/outflow + 7 维时间日历 + 33 维静态特征
  - 门控融合：根据当前流量状态调节时间与静态上下文贡献
  - 时间层：2 层、4 头自注意力，`d_model=32`
  - 空间层：预定义空间图 GCN + 节点嵌入学习的 adaptive adjacency
  - 输出层：`prediction = nonlinear_AGFormer + Linear(history_inflow)`
  - 稳定性：seed 42/7/21/123 预测均值 ensemble。

#### Slide 13 - 真正有效的改动来自消融，而不是盲目加特征

- **Layout:** Existing feature ablation image with four numbered findings around it.
- **Title:** 严格消融：哪些特征和结构真正有用？
- **Core message:** 门控上下文与线性残差稳定改善 MAE；天气、周期滞后和多图方案在当前规则下未形成增益。
- **Visualization:** Existing chart image
- **Content:**
  - 2×2 门控消融拆分 flow / time / static 上下文
  - 日历修复后重跑基线，避免“错误时间编码带来虚假提升”
  - 周期滞后实验使用相同 3,433 测试窗口做严格对照
  - 多图权重被 flow graph 主导，说明简单全局图权重不足以利用所有图
  - 负结果同样记录，避免只展示成功实验。

#### Slide 14 - 四 seed 集成把 24h MAE 降至 15.9030

- **Layout:** Left final-selection chart, right KPI stack; bottom horizon uncertainty strip.
- **Title:** 最终预测性能与误差边界
- **Core message:** 最终模型相对原 AGFormer gated 基线 MAE 改善 3.82%，但误差随预测时域扩大而上升。
- **Visualization:** Existing chart images
- **Content:**
  - h1：MAE 11.3019 / RMSE 19.8621 / MAPE 23.4913
  - 24h 平均：MAE 15.9030 / RMSE 30.2250 / MAPE 35.4350
  - 同一 4,022 个测试窗口，模型选择在验证集完成
  - ensemble MAE 最低；原单模型 RMSE 略低，指标间存在权衡
  - Web 端展示 24h 预测时应提示远期不确定性更大。

### Part 4: 异常检测、分类与预警

#### Slide 15 - 三类证据经验证集定权后形成异常事件

- **Layout:** Adapt `process_flow`; three detector branches converge, then spatial-temporal event builder.
- **Title:** 异常检测：统计、重构、预测误差三路融合
- **Core message:** 单一检测器各有偏差，融合框架用互补证据降低误报并保留召回。
- **Visualization:** process_flow
- **Content:**
  - 统计分支：3σ + IQR，按网格与时段建立正常范围
  - 重构分支：VAE + Transformer Autoencoder，仅用 weak-normal 窗口训练
  - 预测分支：最终 AGFormer ensemble 的观测—预测残差
  - 验证集选择权重 0.70 / 0.20 / 0.10 与阈值 0.56
  - 时间连续 + 空间 IoU/质心距离合并事件，再做类型与严重度分类。

#### Slide 16 - 融合提升弱标签一致性，但没有真实标签就不能夸大 F1

- **Layout:** Method comparison chart left, event metrics right, warning band below.
- **Title:** 异常检测结果：事件级比 timestep 更接近业务语义
- **Core message:** 融合优于单分支，并识别 131 个事件；所有精度指标只代表与弱标签的一致性。
- **Visualization:** Existing chart images
- **Content:**
  - Level-A timestep：Precision 0.4322 / Recall 0.5688 / F1 0.4912
  - Level-A event：Precision 0.4580 / Recall 0.5085 / F1 0.4819
  - Level-B event Recall 0.7931，但 Precision 0.1756
  - 50 个事件人工抽样表用于 sanity check，不替代权威真值
  - 统计规则参与弱标签和融合，存在规则相关性；结果不能解释为现实事件检测准确率。

#### Slide 17 - 检测结果被转化为可查询、可解释、可分级的系统事件

- **Layout:** Adapt `layered_architecture`; right side uses existing screenshot plus two dashed placeholders.
- **Title:** 系统落地：Pipeline → API → 地图 → 三级预警
- **Core message:** 算法输出不止一个分数，而是带时间、空间、类型、解释字段和处置行为的事件对象。
- **Visualization:** layered_architecture
- **Content:**
  - Batch / realtime 微批：预处理、特征、ensemble 推理、异常融合一次完成
  - FastAPI：热力图、24h 预测、异常事件、区域/时间查询、统计接口
  - Streamlit + PyDeck/Plotly：实时热力图、预测动画、事件详情与历史统计
  - 一般 110 / 重要 15 / 紧急 6；重要弹窗，紧急弹窗 + 声音
  - 地图事件 131/131 可定位：108 主中心，23 最高分网格补位
  - 占位：最新首页/预警弹窗截图；Web 操作短视频。

### Part 5: 解释、优化与结论

#### Slide 18 - 可解释性回答“模型看了什么”，但不冒充因果

- **Layout:** Two existing figures side-by-side; one evidence strip beneath each.
- **Title:** 注意力、TreeSHAP 与异常上下文证据
- **Core message:** 最终模型主要依赖历史流量，时间注意力较分散；天气和日历只能提供候选解释。
- **Visualization:** Existing chart images
- **Content:**
  - 时间注意力归一化熵 0.9956：聚合后没有集中到少数 lag
  - TreeSHAP 代理 R² 0.9926；历史 inflow/outflow 贡献合计 97.43%
  - 日历上下文贡献 2.33%；静态特征全局占比较小但存在门控交互
  - 131 个事件中 82 个匹配天气/日历候选证据，49 个无外部上下文
  - Attention、SHAP 和天气关联均不是因果证明。

#### Slide 19 - 性能优化由瓶颈测量驱动，而不是默认使用 FP16

- **Layout:** Paired performance figures above; decision gate below.
- **Title:** 系统性能：更快、更省内存，同时守住数值一致性
- **Core message:** 正式部署保留 FP32 执行优化，因为它真实加速；FP16 虽省显存却没有达到速度门槛。
- **Visualization:** Existing chart images
- **Content:**
  - 暖启动 FP32：6.416s → 4.613s，端到端 1.39×
  - 进程树 RSS 下降 53.9%，CUDA 峰值分配下降 14.8%
  - 共享 full-sequence 推理，消除 h1 与 24h forecast 重复前向
  - 统计量指纹缓存：1.723s → 0.038s
  - batch=8 延迟最低；FP16 质量通过但速度比仅 0.9945×，不部署。

#### Slide 20 - 项目亮点来自“可验证的完整性”

- **Layout:** Three naked-text pillars and a bottom outlook arrow; no cards.
- **Title:** 项目亮点、局限与下一步
- **Core message:** 最重要的成果不是某个单一模型，而是把数据规则、实验、公平对照和工程部署连成可复现系统。
- **Content:**
  - 亮点 1：多源时空数据进入统一无泄漏 pipeline，候选 41 维经审计收敛到 33 维
  - 亮点 2：自研 AGFormer 结合显式图、自适应图、门控上下文与线性残差，并做严格多 seed/消融
  - 亮点 3：三路异常融合落到事件分类、解释字段、三级预警和地图服务
  - 技术难点：时间缺口、空间映射、1024×48 注意力规模、无真异常标签、GPU/系统资源约束
  - 局限：公开基准原型、弱标签评估、离线微批模拟、未接入新闻/活动真值
  - 展望：经授权的真实业务数据、人工事件标签、跨城市迁移、Optuna 自动优化、并发与流式部署。

#### Slide 21 - 主要论文与技术参考

- **Layout:** Two-column bibliography with a thin method-to-paper mapping bar.
- **Title:** 论文与技术参考
- **Core message:** 项目模型是对经典图学习、Transformer、异常检测和可解释性工作的工程组合与实证选择。
- **Content:**
  - Yu et al., STGCN, IJCAI 2018, arXiv:1709.04875
  - Vaswani et al., Attention Is All You Need, NeurIPS 2017
  - Kipf & Welling, GCN, ICLR 2017, arXiv:1609.02907
  - Wu et al., Graph WaveNet, IJCAI 2019, arXiv:1906.00121
  - Grigsby et al., Spacetimeformer, arXiv:2109.12218
  - Kingma & Welling, VAE, ICLR 2014, arXiv:1312.6114
  - Xu et al., Anomaly Transformer, ICLR 2022, arXiv:2110.02642
  - Lundberg & Lee, SHAP, NeurIPS 2017
  - Akiba et al., Optuna, KDD 2019, arXiv:1907.10902

#### Slide 22 - Closing

- **Closing impact:** Restate the transformation as one line—“把城市流动变成可预测、可解释、可处置的安全信号”—with a single orange signal terminating in a shield marker.
- **Layout:** Large closing statement on the left; right side three outcome numbers and a restrained Q&A label.
- **Title:** 从城市流量到安全信号
- **Core message:** 五个月实习让我完成了从研究问题到可运行系统的完整算法工程闭环。
- **Content:**
  - 15.9030 — 24h 平均 MAE
  - 131 — 融合异常事件
  - 1.39× — 优化后端到端加速
  - Q&A

## X. Speaker Notes Requirements

- One note per slide, written first in `notes/total.md` and split later.
- Chinese notes target 35–55 seconds per page; P01/P22 20–30 seconds; total target 12–16 minutes.
- Notes explicitly distinguish sourced facts, estimated five-month scheduling, weak-label metrics, and non-causal explanations.
- Each note includes one transition sentence to the next slide.
- English version mirrors the same facts and page order but uses natural spoken English rather than literal word-for-word translation.

## XI. Technical Constraints Reminder

1. SVG viewBox: `0 0 1280 720`.
2. Background uses `<rect>`.
3. Text wrapping uses `<tspan>`; `<foreignObject>` is forbidden.
4. Use `fill-opacity` / `stroke-opacity`; `rgba()` is forbidden.
5. Forbidden: `<style>`, `class`, `mask`, `textPath`, `animate*`, `script`, `iframe`.
6. `<g opacity>` is forbidden; set opacity per child.
7. Raw Unicode is required; XML-reserved characters must be escaped.
8. Data charts and screenshots use `preserveAspectRatio="xMidYMid meet"` and are never cropped.
9. Every slide uses top-level groups with stable IDs for editable export.
10. All text remains editable; figures remain source images; placeholders are native dashed SVG regions.
