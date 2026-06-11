# Spacetimeformer 论文导读

**论文**：Spacetimeformer: Long-Range Transformers for Dynamic Spatiotemporal Forecasting
**来源**：ICLR 2022 | arXiv:2109.12218
**阅读时长**：约 70 分钟

---

## 读之前：一句话定位

> Spacetimeformer 的核心思想极简单但极深刻：**把每个（变量, 时间步）组合单独变成一个 token**，让标准 Transformer 的注意力自然地同时处理时间和空间关系，不再需要任何预定义图。

---

## 一、它解决了什么问题？

### 标准 Transformer 做时序预测的方式

假设有 N=207 个传感器，T=12 个时间步。

```
标准方式：每个时间步 → 一个 token

Token 1: [sensor_1_speed, sensor_2_speed, ..., sensor_207_speed] @ t=1
Token 2: [sensor_1_speed, sensor_2_speed, ..., sensor_207_speed] @ t=2
...
Token 12: [...] @ t=12

注意力矩阵：12 × 12
只能捕捉时间关系（哪个时间步最重要），完全看不到传感器之间的关系。
```

**问题**：207 个传感器被强行塞进一个 token 里，Transformer 看不到它们各自的"个性"，也看不到它们之间的互动。

### Spacetimeformer 的方式

```
新方式：每个（传感器, 时间步）→ 一个 token

Token (1,1): sensor_1 @ t=1
Token (2,1): sensor_2 @ t=1
...
Token (207,1): sensor_207 @ t=1
Token (1,2): sensor_1 @ t=2
...
Token (207,12): sensor_207 @ t=12

注意力矩阵：(207×12) × (207×12) = 2484 × 2484
每个传感器在每个时间步都可以"关注"其他任意传感器在任意时间步。
```

**好处**：注意力矩阵自然地就是一个动态的空间-时间关联图。无需预定义图，无需特殊的图卷积层。

---

## 二、输入序列变换（最关键的一步）

### 原始输入

```
shape: (T_context, N)
       历史时间步数  变量数（传感器数）

例如：(12, 207) —— 12个时间步，207个传感器
```

### 变换后的序列

```
shape: (T_context × N, 1)
       = (12 × 207, 1)
       = (2484, 1)

原来：每行是所有传感器在某时刻的值
现在：每行是某传感器在某时刻的值（单个数值）
```

这就是论文里说的"flattening"（展平），把 (时间, 空间) 的二维矩阵，变成一维的 token 序列。

### 直觉

```
原来的矩阵（时间 × 空间）：
         传感器1  传感器2  传感器3
t=1  [   80,      90,      70   ]   → 一个 token
t=2  [   85,      88,      72   ]   → 一个 token
t=3  [   78,      92,      68   ]   → 一个 token

Spacetimeformer 的序列（展平）：
(t=1, sensor=1): 80    → token 1
(t=1, sensor=2): 90    → token 2
(t=1, sensor=3): 70    → token 3
(t=2, sensor=1): 85    → token 4
(t=2, sensor=2): 88    → token 5
...
```

---

## 三、嵌入层（Embedding Layer）

每个 token 的表示由四个部分相加：

$$\text{token}_{i} = \text{Value\_Embed}(y_i) + \text{Time\_Embed}(t_i) + \text{Position\_Embed}(pos_i) + \text{Variable\_Embed}(var_i)$$

| 嵌入 | 输入 | 作用 |
|------|------|------|
| **Value Embedding** | 当前值 y（单个数字） | 告诉模型"这里的流量/速度是多少" |
| **Time Embedding** | 时间戳（小时、星期等） | 告诉模型"这是哪个时刻"（用 Time2Vec 学习周期性） |
| **Position Embedding** | 在序列中的位置 | 告诉模型"这是第几步" |
| **Variable Embedding** | 传感器编号 | 告诉模型"这是哪个传感器"（每个传感器有独立的嵌入向量） |

**Time2Vec**（时间嵌入的关键组件）：

$$\text{Time2Vec}(\tau)[i] = \begin{cases} w_0 \tau + b_0 & i=0 \\ \sin(w_i \tau + b_i) & i>0 \end{cases}$$

- 第一维：线性趋势（捕捉时间线性增长）
- 其余维：正弦函数（捕捉不同频率的周期性，如日周期、周周期）
- `w_i, b_i` 都是可学习参数（模型自己找最合适的频率）

---

## 四、注意力机制（与标准 Transformer 的关系）

### 标准 Self-Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

这个公式没有变，变的是**谁作为 Q、K、V**。

在 Spacetimeformer 里：
- Q = 某个 (sensor_i, t_j) 的嵌入，在"问"：哪些其他 token 和我最相关？
- K = 所有其他 (sensor, t) 的嵌入，在"说"：我的特征是什么
- V = 实际传递的信息

**结果**：注意力权重矩阵 A 的 `A[i][j]` = token (sensor_i, t_j) 关注 token (sensor_k, t_l) 的程度

这就相当于一个**动态的时空关联矩阵**，无需人工定义，随时间步变化。

### 本地 + 全局混合注意力

全局注意力计算量是 O((N×T)²)，当 N=207, T=48 时，是 9936² ≈ 1亿次运算，很慢。

**解决方案：两层注意力**

```
Local Attention（本地注意力）：
  每个 token 只关注同一变量的其他时间步
  sensor_1 @ t=1 → sensor_1 @ t=2, t=3, ..., t=12
  复杂度：O(N × T²)

Global Attention（全局注意力）：
  每个 token 关注所有 token
  sensor_1 @ t=1 → sensor_2 @ t=5, sensor_50 @ t=8, ...
  复杂度：O((N×T)²)  ← 用 Performer 近似降到线性
```

**Performer（快速注意力近似）**：

用随机特征（random features）近似 softmax 注意力，把 O((NT)²) 降为 O(NT × r)，其中 r 是随机特征维度，通常远小于 NT。

**直觉**：不是精确计算每对 token 之间的相关度，而是用一个随机投影近似，速度大幅提升，精度损失可接受。

---

## 五、编码器-解码器架构

```
Context 序列（历史）:          Target 序列（预测区间）:
(sensor, t=1..12) → 展平       (sensor, t=13..24) → 展平（值用0占位）
        │                               │
    编码器                          解码器
    (多层注意力)                   (多层注意力 + Cross-Attention)
        │                               │
        └──────── Cross-Attention ──────┘
                        │
                   预测输出（全部时间步一次性输出）
```

**Cross-Attention**：解码器的每个 token 通过注意力机制，从编码器的历史信息中"提取"相关内容。

**一次性输出所有时间步**（非自回归）：
- 自回归：逐步生成（t+1 → 用 t+1 生成 t+2 → ...），误差累积
- 非自回归：一次前向传播输出全部 T_future 步，快且误差不累积

---

## 六、Spacetimeformer 的独特优势

| 能力 | 机制 |
|------|------|
| 无需图结构 | 自适应注意力替代图卷积 |
| 动态时空关联 | 注意力权重随时间步变化 |
| 长程依赖 | Transformer 天然擅长长序列 |
| 外生变量融合 | Variable Embedding 区分不同类型的输入 |
| 缺失值处理 | Given Embedding 标记缺失位置 |

---

## 七、实验结果解读

**METR-LA（207传感器，交通速度预测）**：

| 方法 | MAE |
|------|-----|
| STGCN | 4.59（60分钟） |
| Graph WaveNet | 3.53（60分钟） |
| **Spacetimeformer** | **2.86（平均）** |

- 没有使用预定义路网图，比使用图的方法效果更好或持平
- 在变量数多（空间范围大）的数据集上优势更明显

**AL Solar（137个太阳能传感器）**：
- Spacetimeformer MSE: 7.75 vs 时序基线 9.94（提升 22%）
- 说明当空间关联强时，Spacetimeformer 增益显著

---

## 八、计算量问题（实际使用要注意）

序列长度 = N × T（传感器数 × 时间步数）

本项目场景：N=1024（32×32网格），T=48（24小时/30分钟步）
序列长度 = 1024 × 48 = **49152**

这对标准 Attention 来说太长了。解决方案：
1. 使用 Performer 近似注意力
2. 降低网格分辨率（16×16 = 256节点）
3. 使用窗口注意力（只在时间窗口内做全局注意力）

实际上 Week 4 实现时会先在 **METR-LA（N=207）** 上实验验证效果，再考虑迁移到 TaxiBJ 的大网格。

---

## 九、与本项目的关联

| Spacetimeformer | 本项目对应 |
|-----------------|-----------|
| Variable Embedding | 每个网格有自己的嵌入，区分不同区域特征 |
| Time Embedding (Time2Vec) | 捕捉 30分钟/日/周/节假日周期 |
| 无需预定义图 | 不依赖 OSM 路网即可建模区域关联 |
| 动态注意力 | 早高峰 vs 深夜的区域关联自动切换 |
| 外生变量 | 气象特征可通过多变量输入融合 |

---

## 十、自测问题

1. Spacetimeformer 的 token 序列长度是多少？（给定 N=207 传感器，T=12 历史步，T_future=3 预测步）

2. Variable Embedding 和 Position Embedding 各自解决什么问题？如果只用其中一个会怎样？

3. Local Attention 和 Global Attention 各自的计算复杂度是多少？为什么要混合使用？

4. 论文说"注意力矩阵可以自然地解释为动态图的邻接矩阵"，这句话怎么理解？

5. 对比 STGCN、Graph WaveNet、Spacetimeformer：它们在建模空间关联时分别用了什么机制？各有什么权衡？

6. 在本项目（TaxiBJ 32×32 网格预测）中，你更倾向用哪个模型作为 Week 4 的主模型？理由是什么？
