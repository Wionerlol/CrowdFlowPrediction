# 模型量化与 TensorRT 加速预研报告

**项目**：城市人流异常检测  
**日期**：2026-06-17  
**阶段**：Week 6/7 预研 — 系统性能优化  
**适用模型**：STGCN（基线）/ STAEformer（主模型）/ TESTAM（进阶）  
**目标硬件**：RTX 5080 Laptop 16GB VRAM，推理延迟目标 < 100ms，RAM < 8GB

---

## 一、为什么需要量化与加速

本项目在推理阶段的性能瓶颈来自两处：

| 瓶颈 | 描述 | 影响 |
|------|------|------|
| 图卷积（ChebConv/GCNConv） | 稀疏矩阵乘法，GPU 利用率低 | 延迟 |
| Transformer 注意力（STAEformer） | N×T 维度的 QKV 计算，1024节点×12步=12,288 token | 显存 |
| 实时推理需求 | 每 30 分钟触发一次全城 1024 节点预测 | 响应时间 |

**量化目标**：在 MAE 损失 ≤ 1% 的前提下，推理速度提升 2~4×，显存降低 50%。

---

## 二、量化基础

### 2.1 数值格式对比

| 格式 | 位宽 | 精度 | 显存（1M参数） | 典型计算峰值（RTX 5080） |
|------|------|------|--------------|----------------------|
| FP32 | 32 bit | 高（基准） | 4 MB | ~80 TFLOPS |
| FP16 | 16 bit | 中（误差 <0.1%） | 2 MB | ~160 TFLOPS（2×） |
| BF16 | 16 bit | 中（动态范围同FP32） | 2 MB | ~160 TFLOPS |
| INT8 | 8 bit | 低（误差 0.5~2%） | 1 MB | ~320 TOPS（4×） |
| INT4 | 4 bit | 很低（需 QAT） | 0.5 MB | ~640 TOPS（8×） |

RTX 50xx Blackwell 架构原生支持 FP8 推理，理论可达 1.2 PFLOPS。

### 2.2 量化的核心思想

将浮点权重 $w \in [w_{min}, w_{max}]$ 映射到整数范围：

$$w_q = \text{round}\left(\frac{w}{s}\right) + z, \quad s = \frac{w_{max} - w_{min}}{2^b - 1}$$

其中 $s$ 为 scale（缩放因子），$z$ 为 zero point，$b$ 为量化位宽。

反量化：$\hat{w} = s \cdot (w_q - z)$

**量化误差来源**：
1. **截断误差（Clipping error）**：超出 $[w_{min}, w_{max}]$ 的值被截断
2. **舍入误差（Rounding error）**：离散化引入的近似误差
3. **累积误差**：多层量化误差叠加（深层网络更敏感）

### 2.3 量化粒度

| 粒度 | 描述 | 精度 | 适用场景 |
|------|------|------|---------|
| Per-tensor | 整个张量共享一个 scale | 低 | 简单快速 |
| Per-channel | 每个输出通道独立 scale | 中 | 卷积权重（推荐） |
| Per-token（激活） | 每个 token 独立 scale（动态） | 高 | LLM 推理 |
| Per-group | 每 G 个参数共享 scale | 高 | INT4 权重量化 |

---

## 三、PyTorch 原生量化方案

### 3.1 方案一：动态量化（Dynamic Quantization）

**原理**：权重在模型加载时量化为 INT8，激活值在推理时动态量化（运行时确定 scale）。

**适用层**：`nn.Linear`、`nn.Embedding`（对 LSTM/GRU 非常有效）

**不适用**：卷积层（ChebConv 本质是矩阵乘，但权重维度特殊）

```python
import torch.quantization

# 动态量化（仅量化 Linear 层）
model_dynamic = torch.quantization.quantize_dynamic(
    model,
    qconfig_spec={torch.nn.Linear},
    dtype=torch.qint8
)

# 测试推理速度
import time
with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(100):
        _ = model_dynamic(x)
    print(f"Dynamic INT8: {(time.perf_counter()-t0)/100*1000:.2f} ms/step")
```

**本项目适用性**：STAEformer 中 QKV 投影、FFN 层可量化，预期加速 1.5~2×。

---

### 3.2 方案二：静态量化（Post-Training Static Quantization, PTQ）

**原理**：用校准集（calibration data）统计激活值分布，离线确定 scale/zero_point，推理时全程 INT8。

**流程**：

```python
from torch.quantization import get_default_qconfig, prepare, convert

# 1. 配置量化方案
model.qconfig = get_default_qconfig('fbgemm')  # x86 CPU
# model.qconfig = get_default_qconfig('qnnpack')  # ARM 移动端

# 2. 插入量化/反量化观察器
model_prepared = prepare(model)

# 3. 校准（用训练集子集跑前向）
with torch.no_grad():
    for x_cal, _ in calibration_loader:  # 通常 100~500 批次即可
        model_prepared(x_cal)

# 4. 量化转换
model_int8 = convert(model_prepared)
torch.save(model_int8.state_dict(), 'model_int8.pt')
```

**本项目校准集配置**：
- 从 TaxiBJ 训练集中随机抽取 500 个时步
- 覆盖早高峰、晚高峰、凌晨低谷三种流量模式（保证激活值分布全面）

**限制**：`torch.quantization` 不支持 PyG 的 ChebConv/GCNConv——这些层使用 `torch_sparse` 后端，无法直接量化。需要单独处理（见第五章）。

---

### 3.3 方案三：混合精度（FP16/BF16）

最简单、风险最低的加速方案，**推荐作为第一步**：

```python
# 训练时自动混合精度
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast(dtype=torch.float16):
    pred = model(x)
    loss = criterion(pred, y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 推理时 FP16
model.half()  # 将所有参数转 FP16
with torch.no_grad():
    with autocast():
        pred = model(x.half())
```

**RTX 5080 FP16 vs FP32**：
- 计算峰值 2× 提升（Tensor Core 利用率更高）
- 显存减少 50%
- STAEformer 在 TaxiBJ 1024 节点上 FP16 推理约 8~15ms

**注意事项**：
- 注意力 softmax 建议保持 FP32（数值稳定性）
- Layer Norm 的 reduce 操作建议 FP32
- 用 `autocast` 会自动处理这些边界情况

---

### 3.4 方案四：量化感知训练（QAT）

**原理**：在训练中模拟量化误差（插入 fake quantize 节点），让模型学会在量化约束下仍保持精度。

**适用场景**：PTQ 精度损失 > 1% 时使用，或需要 INT4 量化时。

```python
from torch.quantization import prepare_qat

model.train()
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_qat = prepare_qat(model)

# 正常训练（最后 5~10 个 epoch）
for epoch in range(fine_tune_epochs):
    for x, y in train_loader:
        pred = model_qat(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

# 转为推理模型
model_qat.eval()
model_int8 = convert(model_qat)
```

**预估：** STAEformer QAT INT8 相比 FP32 基线，MAE 损失 < 0.5%，速度提升 3~4×。

---

## 四、TensorRT 加速

### 4.1 TensorRT 工作原理

TensorRT 是 NVIDIA 的高性能推理优化引擎，主要做三件事：

```
ONNX / TorchScript 模型
        ↓
  [图优化 Layer Fusion]
  Conv+BN+ReLU → 单一 kernel
  Multi-head Attention 融合
        ↓
  [精度校准 Calibration]
  FP32 → FP16 / INT8
        ↓
  [Kernel Auto-tuning]
  针对目标 GPU 选择最快 CUDA kernel
        ↓
  TensorRT Engine (.engine / .plan)
```

**关键优化技术**：
| 技术 | 效果 |
|------|------|
| Layer Fusion | 减少内存读写（Conv+BN+Activation合并） |
| Kernel Auto-tuning | 针对具体张量形状选最优 CUDA 实现 |
| Dynamic Shape | 支持 batch size / 序列长度动态变化 |
| FP16/INT8 Calibration | 自动确定量化参数 |
| Multi-Stream | 并发执行多个推理请求 |

### 4.2 导出路径：PyTorch → ONNX → TensorRT

```
PyTorch (.pt)
    ↓ torch.onnx.export()
ONNX (.onnx)
    ↓ trtexec / tensorrt Python API
TensorRT Engine (.engine)
    ↓ trt.IRuntime / polygraphy
推理
```

**步骤一：导出 ONNX**

```python
import torch
import torch.onnx

model.eval()
dummy_input = {
    'x': torch.randn(1, 12, 1024, 2),          # (B, T_in, N, F)
    'edge_index': graph['edge_index'],           # (2, E)
    'edge_weight': graph['edge_weight'],         # (E,)
    'node_feat': node_features,                  # (N, 38)
}

torch.onnx.export(
    model,
    (dummy_input,),
    'staeformer.onnx',
    input_names=['x', 'edge_index', 'edge_weight', 'node_feat'],
    output_names=['pred'],
    dynamic_axes={
        'x': {0: 'batch_size'},
        'pred': {0: 'batch_size'},
    },
    opset_version=17,
    do_constant_folding=True,
)
```

**步骤二：验证 ONNX**

```bash
pip install onnxruntime-gpu onnx
python -c "import onnx; m = onnx.load('staeformer.onnx'); onnx.checker.check_model(m); print('ONNX OK')"
```

**步骤三：转换为 TensorRT Engine**

```bash
# 方法1：命令行工具
trtexec \
  --onnx=staeformer.onnx \
  --saveEngine=staeformer_fp16.engine \
  --fp16 \
  --workspace=4096 \
  --minShapes=x:1x12x1024x2 \
  --optShapes=x:4x12x1024x2 \
  --maxShapes=x:8x12x1024x2

# 方法2：Python API（更灵活）
```

```python
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, fp16=True):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # INT8 校准（可选）
    # config.set_flag(trt.BuilderFlag.INT8)
    # config.int8_calibrator = MyCalibrator(calibration_loader)

    engine = builder.build_serialized_network(network, config)
    with open('staeformer_fp16.engine', 'wb') as f:
        f.write(engine)
    return engine
```

**步骤四：TensorRT 推理**

```python
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

class TRTInferencer:
    def __init__(self, engine_path):
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def infer(self, x_np):
        # 分配 GPU 缓冲区
        bindings = []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            buf = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            bindings.append(int(buf))

        cuda.memcpy_htod(bindings[0], np.ascontiguousarray(x_np, dtype=np.float16))
        self.context.execute_v2(bindings)

        out = np.empty(self.engine.get_binding_shape(1), dtype=np.float16)
        cuda.memcpy_dtoh(out, bindings[-1])
        return out
```

### 4.3 TensorRT 对 GNN 的支持情况

**关键挑战**：PyG 的 ChebConv/GCNConv 底层使用 `scatter_add`（稀疏聚合），TensorRT 不直接支持此类动态图操作。

| 操作 | TensorRT 支持 | 解决方案 |
|------|-------------|---------|
| `nn.Linear` | ✅ 原生支持 | 直接导出 |
| `nn.MultiheadAttention` | ✅ 原生支持 | 直接导出 |
| `ChebConv`（稀疏图乘） | ⚠️ 部分支持 | 转稠密矩阵乘（见下） |
| `scatter_add`（邻居聚合） | ❌ 不支持 | 用稠密 adj 矩阵替换 |
| Dynamic graph structure | ❌ 不支持 | 固定图结构（推理时图不变） |

**解决方案：稠密化图运算**

将 `edge_index` 形式的稀疏图转为稠密邻接矩阵，图卷积变为标准矩阵乘法：

```python
class DenseChebConv(nn.Module):
    """TensorRT 兼容的稠密 ChebConv 实现"""
    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        self.K = K
        self.lins = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False) for _ in range(K)
        ])
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, L_norm):
        """
        x: (N, in_channels)
        L_norm: (N, N) 归一化图拉普拉斯矩阵（稠密，推理前预计算）
        """
        out = self.lins[0](x)                 # T_0(L) * x = x
        if self.K > 1:
            x1 = L_norm @ x                   # T_1(L) * x = Lx
            out = out + self.lins[1](x1)
        for k in range(2, self.K):
            x2 = 2 * L_norm @ x1 - x          # Chebyshev 递推
            out = out + self.lins[k](x2)
            x, x1 = x1, x2
        return out + self.bias
```

对于 1024 节点，稠密邻接矩阵 (1024, 1024) 仅 4MB（FP32），稠密矩阵乘可被 TensorRT 完整优化。

---

## 五、本项目量化实施方案

### 5.1 分阶段策略

```
阶段1（Week 6，2天）: 混合精度 FP16
  - model.half() + autocast
  - 验证 MAE 损失 < 0.1%
  - 预期加速: 1.8~2.2×

阶段2（Week 7，2天）: ONNX 导出 + TensorRT FP16
  - 将 GNN 层替换为稠密实现
  - trtexec FP16 模式构建 engine
  - 预期加速: 3~4×（含 kernel fusion）

阶段3（Week 7，选做）: INT8 PTQ
  - 用 500 步校准集校准
  - 如果 MAE 损失 < 1% 则启用
  - 预期加速: 5~6×
```

### 5.2 各模型量化兼容性

| 模型 | FP16 | PTQ INT8 | TensorRT | 主要障碍 |
|------|------|---------|---------|---------|
| STGCN | ✅ 简单 | ✅ 可行 | ⚠️ 需稠密化ChebConv | ChebConv 稀疏操作 |
| STAEformer | ✅ 简单 | ✅ 可行 | ✅ 较好支持 | 无图操作，纯 Transformer |
| TESTAM | ✅ 简单 | ⚠️ 需测试 | ⚠️ 部分支持 | 混合图+注意力 |

**STAEformer 是 TensorRT 加速的最佳候选**（无 GNN 稀疏操作，纯 Transformer 架构）。

### 5.3 量化敏感层识别

以下层量化后精度损失最大，建议保持 FP32：

```python
# 对 STAEformer 不量化的层
SKIP_QUANTIZE = [
    'norm',           # Layer Normalization（数值稳定性敏感）
    'embed',          # 节点 ID 嵌入（稀疏查表）
    'output_proj',    # 最后输出层（精度敏感）
]
```

### 5.4 性能基准测试代码

```python
import torch
import time

def benchmark(model, x, n_warmup=20, n_runs=100, device='cuda'):
    model.eval()
    x = x.to(device)
    torch.cuda.synchronize()

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(x)
            torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_runs * 1000  # ms

    mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    return elapsed, mem_mb

# 对比实验
x = torch.randn(1, 12, 1024, 2)   # TaxiBJ: 1 batch, 12步, 1024节点, 2通道

results = {}
results['fp32'] = benchmark(model_fp32, x)
results['fp16'] = benchmark(model_fp16, x.half())
# results['trt_fp16'] = benchmark_trt(trt_inferencer, x)

for name, (ms, mem) in results.items():
    print(f"{name}: {ms:.2f} ms/step, {mem:.1f} MB")
```

### 5.5 预期性能指标

基于文献和同类项目经验，在 RTX 5080 上预期：

| 方案 | 推理延迟 | 显存 | MAE 损失 | 实现难度 |
|------|---------|------|---------|---------|
| FP32 基线 | ~40ms | ~800MB | 基准 | — |
| FP16 | ~18ms | ~400MB | <0.1% | 低 |
| TRT FP16 | ~12ms | ~350MB | <0.1% | 中 |
| TRT INT8 | ~8ms | ~220MB | <1% | 高 |

均满足 < 100ms 目标，**推荐实施 FP16 + TRT FP16 两阶段方案**。

---

## 六、torch.compile 快速加速（PyTorch 2.x）

`torch.compile` 是比 TensorRT 更轻量的替代方案，对代码改动极小：

```python
import torch

# 一行代码加速
model_compiled = torch.compile(model, mode='reduce-overhead')
# mode 选项:
#   'default'         — 平衡编译速度和推理速度
#   'reduce-overhead' — 减少 Python/CUDA 调用开销（推荐批量推理）
#   'max-autotune'    — 最大化推理速度（编译时间长）

# 首次推理有编译开销（10~60s），后续推理速度 1.5~3×
with torch.no_grad():
    pred = model_compiled(x)
```

**对 PyG 的支持**：`torch.compile` 对 PyG 2.3+ 版本有良好支持，GNN 层可直接编译加速，无需手动稠密化。

**建议**：在尝试 TensorRT 之前，先用 `torch.compile` 验证加速效果，通常能以最低代码成本获得 1.5~2× 加速。

---

## 七、INT8 校准策略（PTQ 进阶）

TensorRT INT8 校准有三种模式：

| 校准器 | 算法 | 精度 | 适用场景 |
|--------|------|------|---------|
| `IInt8MinMaxCalibrator` | Min-Max | 低 | 快速验证 |
| `IInt8EntropyCalibrator2` | KL 散度 | 中 | **推荐**（默认） |
| `IInt8LegacyCalibrator` | 百分位截断 | 高 | 分布异常时 |

```python
import tensorrt as trt
import numpy as np

class TaxiBJCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data, cache_file='calibration.cache'):
        super().__init__()
        self.data = calibration_data   # numpy array (N_cal, T, nodes, F)
        self.idx = 0
        self.cache_file = cache_file
        self.device_input = cuda.mem_alloc(calibration_data[0].nbytes)

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.idx >= len(self.data):
            return None
        batch = self.data[self.idx:self.idx+1].astype(np.float32)
        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(batch))
        self.idx += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
```

---

## 八、部署架构建议

```
推理请求 (FastAPI)
    │
    ▼
预处理 (numpy/pandas, CPU)
    │
    ▼
TensorRT Engine (GPU FP16)
  ├── STAEformer 预测分支  → 未来 12 步流量
  └── 异常检测分支 (Z-score, CPU) → 告警级别
    │
    ▼
后处理 + 缓存 (Redis)
    │
    ▼
API 响应 (JSON)
```

**关键优化**：
- 预计算并缓存 TensorRT Engine（构建一次，使用多次）
- 节点特征矩阵（38维静态）预传至 GPU，每次推理仅传动态时序
- 使用 CUDA Stream 并发处理批量请求
- 结果缓存 5 分钟（相同时槽的预测结果复用）

---

## 九、工具链与依赖

```bash
# 安装 TensorRT（CUDA 12.8，RTX 50xx）
pip install tensorrt==10.x.x --index-url https://pypi.ngc.nvidia.com

# ONNX 工具链
pip install onnx==1.16 onnxruntime-gpu==1.18 onnxsim

# 性能分析
pip install nvitop            # GPU 实时监控
pip install torch-tb-profiler # PyTorch Profiler + TensorBoard

# 可选：Polygraphy（TensorRT 调试神器）
pip install polygraphy --extra-index-url https://pypi.ngc.nvidia.com
```

---

## 十、参考文献与资源

| 资源 | 链接 / 说明 |
|------|------------|
| TensorRT 官方文档 | developer.nvidia.com/tensorrt |
| torch.compile 文档 | pytorch.org/docs/stable/torch.compiler |
| PyTorch 量化文档 | pytorch.org/docs/stable/quantization.html |
| Polygraphy 工具 | github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy |
| ONNX Opset 参考 | onnx.ai/onnx/operators |
| 同类项目参考 | TrafficTransformer TensorRT 部署案例（GitHub） |
