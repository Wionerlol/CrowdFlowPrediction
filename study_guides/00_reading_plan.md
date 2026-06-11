# 论文阅读计划

---

## 推荐阅读顺序

```
前置知识速查 → STGCN → AGFormer(STAEformer) → Spacetimeformer
     15分钟       60分钟        55分钟               70分钟
```

### 为什么这个顺序？

每篇论文都在解决前一篇留下的核心问题：

```
STGCN（IJCAI 2018）
│  创新：图卷积 + 时序卷积联合建模，第一次抛弃 RNN
│  遗留问题：图结构必须人工预先定义，学不到隐性关联
▼
AGFormer / STAEformer（CIKM 2023）
│  创新：用"自适应嵌入"让 Transformer 自己学出每个节点的时空身份
│         不需要显式图结构，METR-LA 上超越所有 GNN 方法
│  遗留问题：嵌入是固定大小的参数，不能直接处理
│           可变数量的节点或超长序列
▼
Spacetimeformer（ICLR 2022）
   创新：每个（节点, 时间步）单独作为 token，
         注意力矩阵天然就是动态时空关联图，
         无需任何预定义结构，支持任意节点数
```

---

## 每篇预计用时与重点章节

| 论文 | 预计用时 | 必读部分 | 可略读 |
|------|---------|---------|-------|
| 前置知识 | 15分钟 | 全部 | — |
| STGCN（arXiv:1709.04875）| 60分钟 | Abstract, Sec 3.1-3.3, Fig 2 | 附录、证明 |
| AGFormer/STAEformer（CIKM 2023）| 55分钟 | 本导读 + 模型代码 | 消融实验细节 |
| Spacetimeformer（arXiv:2109.12218）| 70分钟 | Abstract, Sec 3.1-3.4, Fig 1 | Sec 5 消融细节 |

---

## 导读文件索引

| 文件 | 对应论文 | 核心概念 |
|------|---------|---------|
| `01_prerequisites.md` | — | 图、GCN、Attention、空洞卷积 |
| `02_stgcn_guide.md` | STGCN | 三明治架构、Chebyshev近似、门控时序卷积 |
| `03_agformer_guide.md` | STAEformer + patrickfan/AGFormer | 自适应嵌入、双轴注意力、代码逐行解读 |
| `04_spacetimeformer_guide.md` | Spacetimeformer | Token展平、Time2Vec、Local+Global注意力 |

---

## 阅读策略建议

**第一轮（快读）**：只读 Abstract + Introduction，10分钟/篇，建立整体印象

**第二轮（精读）**：对照导读逐节理解架构，遇到公式先读左边是什么再看右边怎么算

**STAEformer 特别建议**：导读里有完整模型代码，建议**边读导读边看代码**（代码只有 180 行，非常清晰），对照 `forward()` 函数理解数据流

**遇到卡壳**：直接发消息告诉我哪里不懂，我来拆解
