# 弱标签生成规则

**版本**: v1.0  
**日期**: 2026-06-13  
**依赖**: `docs/anomaly_detection_spec.md`（Z-score 阈值定义）

---

## 一、为什么需要弱标签

TaxiBJ/TaxiNYC 均无人工标注的异常标签，无监督方法（VAE、AnomalyTransformer）不需要标签，但**评估**需要。弱标签的作用：

1. 构建验证集，计算 Precision/Recall/F1
2. 作为半监督信号，辅助模型收敛
3. 可解释性分析的事件锚点

---

## 二、三级标签体系

### Level-A：统计规则标签（自动生成，覆盖广，噪声高）

触发条件与 `anomaly_detection_spec.md` Section 3 的 L2+ 告警对齐：

```python
def generate_level_a_labels(inflow, slot_mu, slot_sig):
    """
    输入: inflow (T, 32, 32), slot_mu/sig (48, 32, 32)
    输出: labels (T,)  0=正常 1=异常
    """
    labels = np.zeros(len(inflow), dtype=int)
    for t in range(len(inflow)):
        s = t % 48
        z = (inflow[t] - slot_mu[s]) / slot_sig[s]
        # L2 条件：z>3 且空间≥12格
        n_exceed = (z > 3.0).sum()
        if n_exceed >= 12:
            labels[t] = 1
    return labels   # 预期约 3.78% 为正例
```

**预期正例率**：~3.78%（约 850 步），负例约 21,634 步。

### Level-B：多规则交叉验证标签（精度更高）

同时满足以下**两个独立规则**才标为异常，降低噪声：

```python
def generate_level_b_labels(inflow, slot_mu, slot_sig, outflow=None):
    """规则1: z-score 空间聚集 AND 规则2: inflow+outflow 同步异常"""
    labels = np.zeros(len(inflow), dtype=int)
    for t in range(len(inflow)):
        s = t % 48
        z_in = (inflow[t] - slot_mu[s]) / slot_sig[s]

        rule1 = (z_in > 3.0).sum() >= 12      # 聚集条件

        # 规则2：outflow 同步异常（两者均超阈值 → 排除单向流）
        if outflow is not None:
            z_out = (outflow[t] - slot_mu[s]) / slot_sig[s]  # 共享统计量近似
            rule2 = (z_out > 2.5).sum() >= 6
        else:
            rule2 = True   # 无 outflow 时退化为单规则

        if rule1 and rule2:
            labels[t] = 1

    return labels   # 预期正例率 ~1.5~2%
```

### Level-C：节假日校验标签（最高可信度，数量少）

利用 `BJ_Holiday.txt` 节假日信息，在已知大节假日（春节、国庆、五一）前后 ±2 天的 L2+ 异常，直接标记为 **高可信正例**（label=2）：

```python
import pandas as pd

def load_holidays(holiday_file='data/raw/taxibj/BJ_Holiday.txt'):
    # 格式：每行一个日期或节假日名称，需解析
    with open(holiday_file, encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()
    # TODO: 解析具体格式（Week 2 确认）
    return set(lines)

def generate_level_c_labels(level_a_labels, timestamps, holiday_set):
    """在 Level-A 标签基础上，节假日附近的正例升级为 label=2"""
    labels = level_a_labels.copy()
    major_holidays = {'0101', '0501', '1001', '0214', '1231'}  # 元旦/五一/国庆/情人节/跨年
    for t, ts in enumerate(timestamps):
        if labels[t] == 1:
            mmdd = ts.strftime('%m%d')
            if mmdd in major_holidays:
                labels[t] = 2   # 高可信正例
    return labels
```

---

## 三、数据集划分策略

避免时间泄露：按时间顺序切分，不随机 shuffle。

```
BJ13 (4888步) + BJ14 (4780步) + BJ15 前75% (4197步) → 训练集  ~13865步
BJ15 后25%   (1399步)                                  → 验证集  ~1399步
BJ16 (7220步)                                          → 测试集  ~7220步
```

- **slot_mu / slot_sig 仅在训练集上计算**，不使用验证/测试集信息
- 正例比例（测试集）：Level-A ~3.78%，约 273 步

---

## 四、类别不平衡处理

正例约 3~4%，不平衡比约 25:1。应对策略（按优先级）：

| 策略 | 适用场景 | 实现 |
|------|---------|------|
| 加权损失函数 | 深度模型训练 | `pos_weight = (1-pos_rate)/pos_rate ≈ 25` in BCEWithLogitsLoss |
| 过采样（正例重复） | 小规模实验 | `class_weight='balanced'` in sklearn |
| 阈值后移 | 评估时调整决策阈值 | 默认 0.5 → 建议 0.2~0.3（提升召回） |
| 无监督方法不需要 | AnomalyTransformer/VAE | 以异常分数排序，阈值由验证集 F1 确定 |

---

## 五、标签质量评估方法

弱标签本身有噪声，用以下方式估计质量：

1. **噪声上界估计**：人工抽查 50 个正例步骤，计算人工标注与弱标签的一致率（目标 >70%）
2. **时间平滑一致性**：异常通常持续 >1 步，统计连续正例比例（预期 >60%）
3. **节假日召回率**：在已知重大节假日（春节/国庆高峰日）上，Level-A 标签的命中率（目标 >80%）

```python
# 时间平滑一致性检验
def check_temporal_consistency(labels):
    consecutive = np.sum((labels[:-1] == 1) & (labels[1:] == 1))
    positive_total = labels.sum()
    print(f"连续正例比: {consecutive / max(positive_total,1) * 100:.1f}%")
```

---

## 六、执行计划

| 周次 | 任务 |
|------|------|
| Week 2 | 实现 Level-A 标签生成，输出 `data/processed/labels_level_a.npy` |
| Week 2 | 解析 `BJ_Holiday.txt` 格式，实现 Level-C 升级 |
| Week 5 | 训练 VAE/AnomalyTransformer，以 Level-B 标签为验证集评估 |
| Week 5 | 网格搜索最优决策阈值，输出 Precision-Recall 曲线 |
