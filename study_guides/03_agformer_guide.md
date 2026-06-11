# Adaptive Graph Transformer 导读

**主论文**：STAEformer — Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting
**来源**：CIKM 2023 | 官方代码：https://github.com/XDZhelheim/STAEformer
**代码参考**：patrickfan/AGFormer（自适应图+Transformer流水线实现）
**阅读时长**：约 55 分钟

---

## 读之前：一句话定位

> STAEformer 证明了：**给普通 Transformer 加上一个可学习的"时空自适应嵌入"，不需要任何图卷积，就能在 METR-LA 等数据集上超越所有 GNN 方法**。核心问题不是"用什么图"，而是"怎么让模型记住每个节点的独特时空身份"。

---

## 一、它解决了什么问题？

### Transformer 用于交通预测的困境

标准 Transformer 做交通预测时，每个节点每个时间步的表示只来自**当前观测值** + **时间戳**。

**问题**：传感器 A（高速公路入口）和传感器 B（居民区路口）在早上8点的流量模式完全不同——一个总是早高峰拥堵，另一个总是平缓。但如果两个传感器在同一时刻观测值碰巧相同，模型根本分不清它们。

**本质缺陷**：模型没有"记忆"——它不知道每个节点历史上在这个时间点长什么样。

### 之前的解法（及其问题）

| 方法 | 解法 | 问题 |
|------|------|------|
| GCN 类（STGCN）| 预定义邻接矩阵 | 图固定，不能学习隐性关联 |
| Graph WaveNet | 自学习邻接矩阵 E₁E₂ᵀ | 图是静态的（不随时间步变化） |
| Spacetimeformer | 每个(节点,时间步)做 token | 序列太长，计算量 O((NT)²) |

**STAEformer 的解法**：给每个 (节点, 时间步) 对分配一个可学习的向量——**自适应嵌入**——让模型主动记住"这个节点在这个时间段的典型行为"。

---

## 二、核心创新：自适应时空嵌入（Adaptive Embedding）

### 定义

```python
# 形状: (in_steps=12, num_nodes=207, adaptive_embedding_dim=80)
self.adaptive_embedding = nn.Parameter(
    torch.empty(in_steps, num_nodes, adaptive_embedding_dim)
)
nn.init.xavier_uniform_(self.adaptive_embedding)
```

这是一个 **12 × 207 × 80** 的可学习张量：
- 12 = 历史时间步数
- 207 = 传感器数（每个传感器有自己的嵌入）
- 80 = 嵌入维度

**直觉**：想象一个 207×12 的"身份证矩阵"，每个 (传感器, 时间步) 有一张独特的 80 维身份证。训练过程中，模型学会了让身份证自动反映该节点在该时刻的典型行为模式。

### 为什么这能替代图？

图卷积的作用是：让每个节点获取邻居的信息。

自适应嵌入的作用是：让每个节点在 **Transformer 的注意力** 里自动找到相似模式的其他节点——不需要显式定义"谁是邻居"，模型通过注意力权重学习出来。

**对比**：
```
Graph WaveNet: 显式学习邻接矩阵 A[i][j]（静态）
STAEformer:   每个节点有自己的嵌入 Ea[t][i]，
              注意力机制让相似嵌入的节点自动相互关注（动态）
```

---

## 三、完整嵌入层（5 种嵌入拼接）

```python
# 前向传播中的嵌入拼接（model_dim = 24+24+24+0+80 = 152）
features = []
features.append(self.input_proj(x))        # 1. 输入特征嵌入: (B, T, N, 24)
features.append(self.tod_embedding(tod))   # 2. 一天中第几步:  (B, T, N, 24)
features.append(self.dow_embedding(dow))   # 3. 星期几:        (B, T, N, 24)
# features.append(spatial_emb)            # 4. 静态节点嵌入:  (B, T, N,  0) ← 默认关闭
features.append(adp_emb)                  # 5. 自适应嵌入:    (B, T, N, 80)
x = torch.cat(features, dim=-1)           # → (B, T, N, 152)
```

| 嵌入 | 维度 | 编码什么 |
|------|------|---------|
| 输入特征嵌入 | 24 | 当前观测值（速度/流量）|
| 时间步嵌入（ToD） | 24 | 一天中的第几个30分钟（0-287）|
| 星期嵌入（DoW） | 24 | 周一到周日 |
| 静态节点嵌入 | 0（可选）| 节点的固定空间位置 |
| **自适应嵌入** | **80** | **每个(节点,时间步)的时空身份**（核心贡献）|

**注意**：自适应嵌入维度（80）占总 model_dim（152）的 **52%**，权重最大，是真正起作用的部分。

---

## 四、时间注意力 + 空间注意力（双轴 Transformer）

```python
# 先在时间轴做注意力
for attn in self.attn_layers_t:
    x = attn(x, dim=1)   # dim=1 是时间维度

# 再在空间轴做注意力
for attn in self.attn_layers_s:
    x = attn(x, dim=2)   # dim=2 是节点维度
```

### 时间注意力（dim=1）

**输入**：`(batch, T=12, N=207, D=152)`
**操作**：对每个节点，在 12 个时间步之间做 Self-Attention
**效果**：学习"这个节点在第 t 步的状态，主要受过去哪几步影响"

```
对 node_i 来说，12步序列做注意力:
[t=1, t=2, ..., t=12] → 哪些历史时刻最相关？
```

### 空间注意力（dim=2）

**输入**：`(batch, T=12, N=207, D=152)`
**操作**：对每个时间步，在 207 个节点之间做 Self-Attention
**效果**：学习"在时刻 t，哪些节点的状态对当前节点最有参考价值"

```
对 t=8(早8点) 来说，207个节点做注意力:
[node_1, node_2, ..., node_207] → 哪些节点行为最相关？
```

### SelfAttentionLayer 完整结构

```python
class SelfAttentionLayer(nn.Module):
    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)     # 把目标维度移到倒数第2位
        residual = x
        out = self.attn(x, x, x)     # Multi-head Self-Attention
        out = self.ln1(residual + out)  # 残差 + LayerNorm
        residual = out
        out = self.feed_forward(out)    # FFN: Linear → ReLU → Linear
        out = self.ln2(residual + out)  # 残差 + LayerNorm
        out = out.transpose(dim, -2)  # 还原维度
        return out
```

这就是标准的 Transformer Encoder Block：Attention → 残差 → LayerNorm → FFN → 残差 → LayerNorm。

---

## 五、输出投影

```python
# use_mixed_proj=True（默认）
out = x.transpose(1, 2)                    # (B, N, T=12, D=152)
out = out.reshape(B, N, T * D)             # (B, N, 12*152=1824)
out = self.output_proj(out)                # Linear(1824 → out_steps*output_dim)
out = out.view(B, N, out_steps, out_dim)   # (B, N, 12, 1)
out = out.transpose(1, 2)                  # (B, out_steps=12, N=207, 1)
```

把所有时间步的特征拼成一个大向量，一次性线性变换输出所有预测步——非自回归，速度快。

---

## 六、实验结果

**数据集**：METR-LA（207传感器），PEMS-BAY（325传感器）

**METR-LA（MAE，越低越好）**：

| 方法 | 15分钟 | 30分钟 | 60分钟 |
|------|--------|--------|--------|
| STGCN | 2.88 | 3.47 | 4.59 |
| Graph WaveNet | 2.69 | 3.07 | 3.53 |
| DCRNN | 2.77 | 3.15 | 3.60 |
| **STAEformer** | **2.59** | **3.03** | **3.34** |

**结论**：不用任何图卷积，仅靠自适应嵌入+标准Transformer，超越所有 GNN 方法。

---

## 七、patrickfan/AGFormer 的架构对照

patrickfan 的实现展示了完整的自适应图 + Transformer 流水线：

```
原始输入 (温度/降水/流量)
        │
  特征提取器（共享编码器）
        │ → 节点嵌入（类似 STAEformer 的自适应嵌入）
        │
  自适应图模块（Adaptive Graph Module）
        │ → 基于 k-NN 图 + 阈值剪枝，动态学习节点间连接
        │ → 消息传递：融合邻居信息
        │
  Transformer 时序模块
        │ → 对空间增强后的序列做时序注意力
        │
  预测头（Forecasting Head）
        │
  多步预测输出
```

**对应关系**：

| patrickfan/AGFormer | STAEformer |
|---------------------|-----------|
| 特征提取器 | input_proj |
| 自适应图模块 | adaptive_embedding（隐式图）|
| Transformer时序模块 | attn_layers_t |
| 预测头 | output_proj |

patrickfan 是显式图（学出邻接矩阵再做消息传递），STAEformer 是隐式图（用嵌入+注意力隐式建模）。两者思想一脉相承。

---

## 八、与本项目的关联

| STAEformer 组件 | 本项目对应 |
|----------------|-----------|
| tod_embedding（288步/天）| TaxiBJ 30分钟间隔，48步/天 |
| dow_embedding（周一到日）| BJ_Holiday.txt 节假日标记可加入 |
| adaptive_embedding（12×207×80）| 本项目：48×1024×80（48步，32×32=1024网格）|
| 空间注意力（207节点间）| 1024个网格之间的空间注意力（需稀疏化）|
| 输出（12步）| 本项目目标：预测未来48步 |

**重点注意**：本项目网格数 N=1024，空间注意力复杂度 O(N²)=O(1M)，需要稀疏注意力或分块处理。可先在 METR-LA（N=207）上验证效果，再迁移到 TaxiBJ。

---

## 九、自测问题

1. 自适应嵌入的形状是 `(in_steps, num_nodes, dim)`，为什么需要 `in_steps` 这个维度？如果去掉 `in_steps` 变成 `(num_nodes, dim)` 会有什么影响？

2. 时间注意力和空间注意力是顺序执行的（先时间后空间），能否反过来（先空间后时间）？论文里有实验回答这个问题吗？

3. 对比 Graph WaveNet 的显式自适应邻接矩阵 `softmax(ReLU(E₁E₂ᵀ))`，STAEformer 的自适应嵌入是如何隐式建模节点间关系的？

4. STAEformer 在消融实验中去掉自适应嵌入后性能下降多少？这说明什么？

5. 在 TaxiBJ 的 32×32 网格上用 STAEformer，adaptive_embedding 的参数量是多少？（提示：48步 × 1024网格 × 80维）这对训练有什么影响？
