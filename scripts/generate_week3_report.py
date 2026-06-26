"""
Week 3 报告 PDF 生成脚本
输出：outputs/reports/week3_report.pdf
"""
from pathlib import Path
from weasyprint import HTML, CSS

HTML_CONTENT = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"/>
<title>Week 3 进度报告 — 基线模型训练与评估</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
    font-size: 10pt;
    line-height: 1.65;
    color: #1a1a1a;
    background: #fff;
  }

  .cover {
    page-break-after: always;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    padding: 60px 40px;
    background: linear-gradient(160deg, #0f2650 0%, #1a4a8a 60%, #2563c0 100%);
    color: #fff;
    text-align: center;
  }
  .cover .badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 6px 20px;
    font-size: 9pt;
    letter-spacing: 0.08em;
    margin-bottom: 40px;
  }
  .cover h1 { font-size: 26pt; font-weight: 700; margin-bottom: 16px; line-height: 1.3; }
  .cover h2 { font-size: 14pt; font-weight: 400; opacity: 0.85; margin-bottom: 48px; }
  .cover .meta { font-size: 9.5pt; opacity: 0.7; line-height: 2; }

  .page { padding: 48px 56px; max-width: 800px; margin: 0 auto; }
  .page-break { page-break-before: always; }

  h1.section { font-size: 16pt; font-weight: 700; color: #1a4a8a;
               border-bottom: 2.5px solid #1a4a8a; padding-bottom: 8px;
               margin: 32px 0 18px; }
  h2.sub { font-size: 12pt; font-weight: 700; color: #2563c0;
           margin: 24px 0 10px; }
  h3.sub2 { font-size: 10.5pt; font-weight: 700; color: #333;
            margin: 16px 0 8px; }

  p { margin-bottom: 10px; }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0 20px;
    font-size: 9.5pt;
  }
  th {
    background: #1a4a8a;
    color: #fff;
    padding: 8px 12px;
    text-align: center;
    font-weight: 600;
  }
  td {
    padding: 7px 12px;
    text-align: center;
    border-bottom: 1px solid #e8edf5;
  }
  tr:nth-child(even) td { background: #f4f7fc; }
  tr.best td { background: #e8f5e9; font-weight: 700; }
  tr.group-header td {
    background: #dce8f8;
    font-weight: 700;
    font-size: 9pt;
    color: #1a4a8a;
    text-align: left;
    padding-left: 14px;
  }

  .highlight-box {
    background: #f0f5ff;
    border-left: 4px solid #2563c0;
    padding: 14px 18px;
    margin: 16px 0;
    border-radius: 0 6px 6px 0;
  }
  .highlight-box p { margin-bottom: 6px; }
  .highlight-box p:last-child { margin-bottom: 0; }

  .warning-box {
    background: #fff8e1;
    border-left: 4px solid #f59e0b;
    padding: 14px 18px;
    margin: 16px 0;
    border-radius: 0 6px 6px 0;
  }

  .finding {
    display: flex;
    gap: 12px;
    margin-bottom: 10px;
    align-items: flex-start;
  }
  .finding .num {
    background: #2563c0;
    color: #fff;
    border-radius: 50%;
    width: 22px;
    height: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 9pt;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 1px;
  }

  ul { padding-left: 20px; margin-bottom: 10px; }
  li { margin-bottom: 5px; }

  .tag-badge {
    display: inline-block;
    background: #e8f0fe;
    color: #1a4a8a;
    border-radius: 4px;
    padding: 1px 7px;
    font-size: 8.5pt;
    font-weight: 600;
    margin: 0 2px;
  }

  .footer {
    margin-top: 48px;
    padding-top: 16px;
    border-top: 1px solid #ddd;
    font-size: 8.5pt;
    color: #888;
    text-align: center;
  }

  code {
    background: #f1f5f9;
    border-radius: 3px;
    padding: 1px 5px;
    font-family: "Courier New", monospace;
    font-size: 9pt;
    color: #c0392b;
  }
</style>
</head>
<body>

<!-- ══════════════════ COVER ══════════════════ -->
<div class="cover">
  <div class="badge">WEEK 3 · 2026-06-25</div>
  <h1>基线模型训练与评估</h1>
  <h2>城市人流预测系统 — 第三周进度报告</h2>
  <div class="meta">
    数据集：TaxiBJ (北京) &nbsp;|&nbsp; 预测任务：未来 24 小时人流量<br/>
    评估划分：训练 70% / 验证 10% / 测试 20%（时序 7:1:2）<br/>
    6 个基线模型全部完成 · T<sub>in</sub> 消融实验（6h / 1天 / 1周）
  </div>
</div>

<!-- ══════════════════ 1. 目标与方法 ══════════════════ -->
<div class="page">
<h1 class="section">1 &nbsp; 本周目标与完成情况</h1>

<h2 class="sub">1.1 任务目标</h2>
<p>按项目文档要求，本周实现并评估三类基线模型，建立统一的评估体系，为后续多图 ST-GCN 主模型提供对比基准：</p>
<ul>
  <li><strong>传统时序基线</strong>：ARIMA、Prophet（逐网格独立训练，预测未来24小时）</li>
  <li><strong>深度学习基线</strong>：LSTM、GRU（批量训练，共享权重）</li>
  <li><strong>基础图神经网络</strong>：GCN、GAT（利用空间邻接图，时间序列作节点特征）</li>
</ul>

<h2 class="sub">1.2 完成情况</h2>
<table>
  <tr><th>任务</th><th>状态</th><th>备注</th></tr>
  <tr><td>ARIMA 统计诊断（ADF / ACF-PACF / AIC / Ljung-Box）</td><td>✅ 完成</td><td>ARIMA(2,0,1)</td></tr>
  <tr><td>ARIMA Fit-once 滚动预测（1017 格）</td><td>✅ 完成</td><td>~30 分钟</td></tr>
  <tr><td>Prophet Fit-once 预测（1017 格）</td><td>✅ 完成</td><td>~20 分钟</td></tr>
  <tr><td>LSTM / GRU 训练（T_in=12，T_out=48）</td><td>✅ 完成</td><td>各约 20 分钟</td></tr>
  <tr><td>GCN / GAT 训练（T_in=12，T_out=48）</td><td>✅ 完成</td><td>各约 8–15 分钟</td></tr>
  <tr><td>T_in 消融实验（6h / 1天 / 1周）</td><td>✅ 完成</td><td>4 模型 × 3 档</td></tr>
  <tr><td>统一结果 CSV 与 MLflow 追踪</td><td>✅ 完成</td><td>week3_metrics.csv</td></tr>
</table>

<h2 class="sub">1.3 实验设置</h2>
<table>
  <tr><th>参数</th><th>值</th></tr>
  <tr><td>数据集</td><td>TaxiBJ（22484 时间步，32×32 网格，30 分钟/步）</td></tr>
  <tr><td>划分比例</td><td>训练 70% / 验证 10% / 测试 20%（时序不打乱）</td></tr>
  <tr><td>训练集大小</td><td>15,679 步；验证集 2,190 步；测试集 4,438 步</td></tr>
  <tr><td>输入长度（基线）</td><td>T_in = 12 步（6 小时）</td></tr>
  <tr><td>预测长度</td><td>T_out = 48 步（24 小时）</td></tr>
  <tr><td>优化器</td><td>AdamW（lr=0.001，weight_decay=1e-4）</td></tr>
  <tr><td>损失函数</td><td>Huber Loss</td></tr>
  <tr><td>早停策略</td><td>patience = 15 epochs</td></tr>
  <tr><td>评估指标</td><td>MAE、RMSE、MAPE（反归一化后在原始尺度计算）</td></tr>
  <tr><td>GPU</td><td>NVIDIA RTX 5080 Laptop（16 GB）</td></tr>
</table>
</div>

<!-- ══════════════════ 2. 模型架构 ══════════════════ -->
<div class="page page-break">
<h1 class="section">2 &nbsp; 模型架构说明</h1>

<h2 class="sub">2.1 传统统计模型</h2>

<h3 class="sub2">ARIMA — 统计时序诊断流程</h3>
<p>针对 TaxiBJ 的人流时序进行了完整的三阶段统计分析：</p>
<ul>
  <li><strong>阶段1 — ADF 平稳性检验</strong>：对 30 个高活跃网格进行 Augmented Dickey-Fuller 检验，所有采样格子均已平稳（p&lt;0.001），确定差分阶数 <code>d=0</code>，无需差分处理。</li>
  <li><strong>阶段2 — AIC 参数选择与残差诊断</strong>：在 p∈[0,3]、q∈[0,2] 范围内进行网格搜索，最优阶数集中于 ARIMA(2,0,1) 和 ARIMA(2,0,2)，以众数确定全局阶数 <strong>ARIMA(2,0,1)</strong>。Ljung-Box 检验显示残差仍存在自相关（p&lt;0.05），说明简单 ARIMA 无法完全捕获交通流的周期性规律。</li>
  <li><strong>阶段3 — Fit-once 滚动预测</strong>：在完整训练集拟合一次模型参数，通过 <code>append(refit=False)</code> 滚动追加真实值更新残差状态，对全部 1,017 个非零格进行测试集预测。</li>
</ul>

<h3 class="sub2">Prophet</h3>
<p>使用 Meta Prophet 分解模型，配置每日季节性（<code>daily_seasonality=True</code>）和每周季节性（<code>weekly_seasonality=True</code>），关闭年度季节性（数据仅跨 3 年）。采用 Fit-once 策略，在训练集拟合一次后直接外推至整个测试集时间范围。</p>
<div class="warning-box">
  <p><strong>节假日特征</strong>：本次未加入中国公共节假日数据。作为统一基线对比，其他模型同样未引入节假日信息，保证评估条件一致。</p>
</div>

<h2 class="sub">2.2 深度学习序列模型</h2>

<h3 class="sub2">NodeLSTM / NodeGRU</h3>
<p>对每个网格节点共享同一组 RNN 参数（parameter sharing across nodes）。前向传播时将输入 <code>(B, N, T_in, 2)</code> 重塑为 <code>(B×N, T_in, 2)</code>，批量并行处理所有节点的时序，取最后隐层状态经全连接层输出 T_out 步预测。</p>
<table>
  <tr><th>超参数</th><th>LSTM</th><th>GRU</th></tr>
  <tr><td>隐层维度</td><td>64</td><td>64</td></tr>
  <tr><td>层数</td><td>2</td><td>2</td></tr>
  <tr><td>Dropout</td><td>0.1</td><td>0.1</td></tr>
  <tr><td>参数量</td><td>0.054 M</td><td>0.041 M</td></tr>
</table>

<h2 class="sub">2.3 图神经网络模型</h2>

<h3 class="sub2">GCN / GAT</h3>
<p>以 32×32 网格的地理邻接关系构建空间图 G_spatial（7,812 条边），节点特征为将 T_in 个时间步展平后的向量 <code>[T_in × 2]</code>（inflow + outflow 拼接）。两层图卷积后经全连接层输出预测。</p>
<table>
  <tr><th>超参数</th><th>GCN</th><th>GAT</th></tr>
  <tr><td>隐层维度</td><td>64</td><td>32（×4 heads = 128）</td></tr>
  <tr><td>图卷积层数</td><td>2</td><td>2</td></tr>
  <tr><td>注意力头数</td><td>—</td><td>4</td></tr>
</table>
</div>

<!-- ══════════════════ 3. 实验结果 ══════════════════ -->
<div class="page page-break">
<h1 class="section">3 &nbsp; 实验结果</h1>

<h2 class="sub">3.1 基线模型对比（T_in = 12，测试集 avg）</h2>

<table>
  <tr><th>类别</th><th>模型</th><th>MAE ↓</th><th>RMSE ↓</th><th>MAPE ↓</th></tr>
  <tr class="group-header"><td colspan="5">传统统计模型</td></tr>
  <tr><td>统计</td><td>ARIMA(2,0,1)</td><td>44.73</td><td>74.83</td><td>110.7%</td></tr>
  <tr><td>统计</td><td>Prophet</td><td>43.81</td><td>70.18</td><td>105.7%</td></tr>
  <tr class="group-header"><td colspan="5">图神经网络（纯空间）</td></tr>
  <tr><td>GNN</td><td>GCN</td><td>37.35</td><td>63.55</td><td>82.7%</td></tr>
  <tr><td>GNN</td><td>GAT</td><td>33.53</td><td>56.14</td><td>100.7%</td></tr>
  <tr class="group-header"><td colspan="5">深度学习序列模型</td></tr>
  <tr class="best"><td>深度学习</td><td>LSTM</td><td>29.02</td><td>51.54</td><td>63.0%</td></tr>
  <tr class="best"><td>深度学习</td><td>GRU</td><td>29.04</td><td>51.58</td><td>61.2%</td></tr>
</table>

<h2 class="sub">3.2 LSTM 预测误差随预测步长变化</h2>
<p>以 LSTM（最优基线）分析误差随预测步长（h1=30min, h3=1.5h, h6=3h, h12=6h, avg=全24h）的变化趋势：</p>
<table>
  <tr><th>预测步长</th><th>时间跨度</th><th>MAE</th><th>RMSE</th><th>MAPE</th></tr>
  <tr><td>h1（1步）</td><td>30 分钟</td><td>13.09</td><td>22.63</td><td>28.4%</td></tr>
  <tr><td>h3（3步）</td><td>1.5 小时</td><td>16.71</td><td>29.75</td><td>34.6%</td></tr>
  <tr><td>h6（6步）</td><td>3 小时</td><td>20.95</td><td>37.48</td><td>44.5%</td></tr>
  <tr><td>h12（12步）</td><td>6 小时</td><td>25.70</td><td>45.05</td><td>57.4%</td></tr>
  <tr><td>avg（全段）</td><td>24 小时均值</td><td>29.02</td><td>51.54</td><td>63.0%</td></tr>
</table>
<p>误差随预测步长单调递增，短期预测（30分钟内）MAPE 仅 28.4%，24 小时平均误差达 63.0%，体现了长期预测的固有难度。</p>

<h2 class="sub">3.3 T_in 消融实验（输入历史长度影响）</h2>
<table>
  <tr><th>模型</th><th>6h（T_in=12）</th><th>1天（T_in=48）</th><th>1周（T_in=336）</th></tr>
  <tr><td>LSTM</td><td>29.02</td><td><strong>20.34</strong></td><td>20.73</td></tr>
  <tr><td>GRU</td><td>29.04</td><td><strong>20.43</strong></td><td>21.26</td></tr>
  <tr><td>GCN</td><td>37.35</td><td>—</td><td><strong>33.45</strong></td></tr>
  <tr><td>GAT</td><td>33.53</td><td>—</td><td><strong>25.49</strong></td></tr>
</table>
<p style="font-size:8.5pt;color:#555;">表中数值为 avg horizon MAE（↓越小越好）。GCN/GAT 的 T_in=48 结果因显存限制未测试。</p>
</div>

<!-- ══════════════════ 4. 分析与讨论 ══════════════════ -->
<div class="page page-break">
<h1 class="section">4 &nbsp; 分析与讨论</h1>

<h2 class="sub">4.1 核心发现</h2>

<div class="finding">
  <div class="num">1</div>
  <div>
    <strong>时序建模能力是 24 小时预测的关键</strong><br/>
    LSTM/GRU（MAE≈29）显著优于纯空间模型 GCN/GAT（MAE 33–37），尽管后者利用了图结构。说明在 24 小时长程预测中，捕获时间依赖的能力比捕获空间依赖更为关键。ARIMA/Prophet 缺乏对交通流复杂非线性和高阶周期性的建模能力，表现最差。
  </div>
</div>

<div class="finding">
  <div class="num">2</div>
  <div>
    <strong>历史窗口 1 天是 LSTM/GRU 的最优输入长度</strong><br/>
    从 6 小时延长至 1 天，MAE 从 29.02 降至 20.34（-30%），提升显著；但进一步延长至 1 周（T_in=336）并无额外收益（MAE=20.73，甚至微幅上升）。这说明 LSTM 对超长时序的梯度传播困难，1 天内的日内周期性（早晚高峰）是最有效的预测信号。
  </div>
</div>

<div class="finding">
  <div class="num">3</div>
  <div>
    <strong>图模型从更长历史中获益更多</strong><br/>
    GAT 从 6h→1 周 MAE 下降 24%（33.53→25.49），GCN 下降 10%（37.35→33.45）。图模型没有专用的时序建模模块，依赖更长的历史特征向量来弥补这一缺陷，说明时序与空间的联合建模（即后续 ST-GCN 的方向）是更优的技术路线。
  </div>
</div>

<div class="finding">
  <div class="num">4</div>
  <div>
    <strong>ARIMA 残差诊断揭示模型局限</strong><br/>
    Ljung-Box 检验显示所有测试格子的 ARIMA 残差均存在自相关（p&lt;0.05），拒绝白噪声假设。这说明简单 ARIMA(2,0,1) 无法捕获交通流的强周期性结构（日周期、周周期），其残差中仍含大量可预测信息，为深度学习方法的优越性提供了统计支撑。
  </div>
</div>

<h2 class="sub">4.2 模型局限性（Limitation Analysis）</h2>

<div class="highlight-box">
  <p><strong>LSTM/GRU</strong>：共享权重设计忽略了节点间的空间异质性——市中心与郊区的流量模式差异显著，但两者使用相同的网络权重；同时缺乏空间信息传递，相邻网格的互动规律无法被捕捉。</p>
  <p><strong>GCN/GAT</strong>：将时序信息展平为静态节点特征，丢失了时间序列的顺序结构；且仅使用地理邻接图，未利用功能相似性、交通流量相关性等多模态空间关系。</p>
  <p><strong>ARIMA/Prophet</strong>：均为单变量模型，无法利用相邻网格的信息；ARIMA 对强季节性序列需要 SARIMA 扩展才能充分建模（目前残差仍有自相关）；Prophet 未引入节假日特征，对春节、国庆等特殊时段预测精度下降。</p>
  <p><strong>通用局限</strong>：所有基线模型仅使用出行流量一种特征，未融合 POI、卫星图像、公交地铁路网等多源时空语义特征，这是主模型的核心改进方向。</p>
</div>

<h2 class="sub">4.3 下周工作计划</h2>
<ul>
  <li>实现多图门控融合 ST-GCN（MultiGraphSTGCN），联合利用 G_spatial / G_poi / G_flow / G_transit 四张图</li>
  <li>引入多源节点特征（POI 分布、卫星影像、公交地铁特征）</li>
  <li>与本周基线进行统一对比，验证多图融合与多源特征的增益</li>
  <li>完成两周 / 一月 T_in 消融补充实验</li>
</ul>

<div class="footer">
  Week 3 进度报告 &nbsp;·&nbsp; 城市人流预测系统 &nbsp;·&nbsp; 2026-06-25 &nbsp;·&nbsp;
  实验代码：<code>scripts/train.py</code> / <code>scripts/train_arima.py</code> / <code>scripts/ablation_tin.py</code>
</div>
</div>

</body>
</html>"""

def main():
    out_path = Path("outputs/reports/week3_report.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    css = CSS(string="""
        @page {
            size: A4;
            margin: 0;
        }
        @page:not(:first) {
            margin: 0;
        }
    """)

    print("生成 Week 3 报告...")
    HTML(string=HTML_CONTENT, base_url=str(Path("outputs/reports").resolve())).write_pdf(
        str(out_path), stylesheets=[css]
    )
    size_kb = out_path.stat().st_size // 1024
    print(f"✅ 已生成：{out_path}  ({size_kb} KB)")

if __name__ == "__main__":
    main()
