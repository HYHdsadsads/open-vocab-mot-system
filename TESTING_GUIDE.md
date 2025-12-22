# 🧪 项目测试指南

## 测试概览

本项目提供了多层次的测试方法，从快速功能验证到完整性能评估。

---

## ✅ 测试结果（刚刚运行）

```
🧪 测试优化功能 - 增强版
================================================================================
📋 环境信息:
   Python: 3.10.8
   PyTorch: 2.9.1+cpu
   CUDA: ❌ 不可用

1️⃣ 测试 YOLO-RD (检索增强检测)...
   ✅ YOLO-RD 模块正常 (6.55s)

2️⃣ 测试 Knowledge DeepSORT (知识增强跟踪)...
   ✅ Knowledge DeepSORT 模块正常 (3.46s)

3️⃣ 测试边缘计算优化...
   ✅ 边缘优化模块正常 (0.01s)

4️⃣ 测试可视化工具...
   ✅ 可视化工具正常 (0.36s)

📊 测试总结
================================================================================
✅ 通过: 4
❌ 失败: 0

🎉 所有测试通过!
```

---

## 🎯 测试方法

### 1️⃣ 快速功能测试（推荐）

**用途**: 验证所有核心模块是否正常工作

```bash
python test_optimization.py
```

**测试内容**:
- ✅ YOLO-RD 检索增强检测
- ✅ Knowledge DeepSORT 知识增强跟踪
- ✅ 边缘计算优化分析
- ✅ 可视化工具

**预期时间**: 10-15 秒

**成功标志**: 
```
🎉 所有测试通过!
✅ 通过: 4
❌ 失败: 0
```

---

### 2️⃣ 完整实验测试

**用途**: 运行三个完整的研究实验

```bash
python experiments/run_optimization_experiments.py
```

**测试内容**:

#### 实验 1: 检索增强检测
- 构建领域字典（工业、交通场景）
- 测试检索增强检测
- 验证缓存机制

#### 实验 2: 知识增强跟踪
- 初始化跟踪器
- 模拟多帧跟踪
- 统计跟踪性能

#### 实验 3: 边缘计算优化
- 算子融合分析
- 量化策略对比
- 推理延迟分解
- 生成可视化图表

**预期时间**: 2-5 分钟

**输出文件**:
```
optimization_results.json          # 优化分析结果
./optimization_plots/              # 可视化图表目录
├── latency_breakdown.png          # 延迟分解饼图
├── quantization_comparison.png    # 量化对比图
├── fusion_analysis.png            # 算子融合分析
└── speedup_comparison.png         # 加速比对比
```

---

### 3️⃣ 端到端视频测试

**用途**: 测试完整的视频处理流程

```bash
# 基础测试（自动生成测试视频）
python main.py

# 使用自己的视频
python main.py --video your_video.mp4 --output result.mp4

# 完整功能测试
python main.py \
  --video your_video.mp4 \
  --use-yolo-rd \
  --use-knowledge-tracker \
  --experiment
```

**测试内容**:
- 视频读取和处理
- 目标检测
- 目标跟踪
- 结果可视化
- 视频输出

**预期时间**: 取决于视频长度（约 1-2 分钟/150帧）

**输出**:
- `output_video.mp4`: 处理后的视频
- 控制台统计信息

---

### 4️⃣ 模块单元测试

**用途**: 测试单个模块的功能

#### 测试 YOLO-RD

```python
from models.yolo_rd import RetrievalAugmentedDetector
from ultralytics import YOLO
import clip
import numpy as np

# 加载模型
yolo_model = YOLO("yolov8x-worldv2.pt")
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")

# 初始化 YOLO-RD
yolo_rd = RetrievalAugmentedDetector(yolo_model, clip_model, clip_preprocess)

# 构建字典
yolo_rd.build_domain_dictionary("test", ["person", "car", "bicycle"])

# 测试检测
test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
detections = yolo_rd.forward(test_image, domain="test")

print(f"检测到 {len(detections)} 个目标")
```

#### 测试 Knowledge DeepSORT

```python
from tracking.knowledge_deepsort import KnowledgeDeepSORT

# 初始化跟踪器
tracker = KnowledgeDeepSORT(dictionary=yolo_rd.domain_dict)

# 模拟跟踪
for frame_id in range(10):
    detections = yolo_rd.forward(test_image, domain="test")
    tracks = tracker.update(detections, frame_id)
    print(f"帧 {frame_id}: {len(tracks)} 条轨迹")

# 获取统计
stats = tracker.get_statistics()
print(f"总帧数: {stats['total_frames']}")
print(f"ID 切换: {stats['id_switches']}")
```

#### 测试边缘优化

```python
from optimization.edge_optimization import EdgeOptimizationPipeline
import torch

# 创建测试模型
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 64, 3),
    torch.nn.BatchNorm2d(64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 10)
)

# 初始化优化器
optimizer = EdgeOptimizationPipeline(model)

# 运行分析
dummy_input = torch.randn(1, 3, 224, 224)
results = optimizer.run_full_optimization_analysis(dummy_input)

print(f"融合机会: {len(results['fusion_analysis']['opportunities'])}")
print(f"优化建议: {len(results['recommendations'])}")
```

---

### 5️⃣ 性能基准测试

**用途**: 测量系统性能指标

```bash
# 创建性能测试脚本
python -c "
import time
import numpy as np
from models.yolo_rd import RetrievalAugmentedDetector
from ultralytics import YOLO
import clip

# 加载模型
yolo_model = YOLO('yolov8x-worldv2.pt')
clip_model, clip_preprocess = clip.load('ViT-B/32', device='cpu')
yolo_rd = RetrievalAugmentedDetector(yolo_model, clip_model, clip_preprocess)

# 构建字典
yolo_rd.build_domain_dictionary('test', ['person', 'car'])

# 性能测试
test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
times = []

for i in range(10):
    start = time.time()
    detections = yolo_rd.forward(test_image, domain='test')
    times.append(time.time() - start)

print(f'平均检测时间: {np.mean(times):.4f}s')
print(f'FPS: {1/np.mean(times):.2f}')
"
```

---

## 📊 测试指标

### 功能测试指标

| 测试项 | 状态 | 耗时 |
|--------|------|------|
| YOLO-RD | ✅ 通过 | 6.55s |
| Knowledge DeepSORT | ✅ 通过 | 3.46s |
| 边缘优化 | ✅ 通过 | 0.01s |
| 可视化 | ✅ 通过 | 0.36s |

### 性能测试指标

| 指标 | CPU 模式 | GPU 模式（预期） |
|------|----------|------------------|
| 检测速度 | ~1.2 FPS | ~30-60 FPS |
| 内存占用 | ~2.1 GB | ~3-4 GB |
| 模型加载 | ~10s | ~5s |
| 字典构建 | ~6s | ~3s |

---

## 🐛 故障排查

### 问题 1: 测试失败

**症状**: 某个测试模块失败

**解决方案**:
```bash
# 1. 检查依赖
python check_dependencies.py

# 2. 验证安装
python verify_installation.py

# 3. 查看详细错误
python test_optimization.py --verbose
```

### 问题 2: 内存不足

**症状**: `RuntimeError: out of memory`

**解决方案**:
```bash
# 减小批处理大小
python main.py --batch-size 1

# 使用更小的模型
# 修改 config.py 中的模型路径
```

### 问题 3: 速度太慢

**症状**: 测试运行时间过长

**解决方案**:
```bash
# 1. 确认使用 CPU 模式（正常较慢）
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 2. 如果有 GPU，安装 CUDA 版 PyTorch
# 3. 减少测试数据量
```

---

## ✅ 测试检查清单

运行测试前确认：

- [ ] Python 3.10+ 已安装
- [ ] 所有依赖已安装 (`python check_dependencies.py`)
- [ ] 模型文件存在 (`yolov8x-worldv2.pt`)
- [ ] 有足够的磁盘空间（至少 2GB）
- [ ] 有足够的内存（至少 4GB）

运行测试后确认：

- [ ] 所有功能测试通过
- [ ] 生成了输出文件
- [ ] 没有错误或警告
- [ ] 性能指标在预期范围内

---

## 📈 测试报告示例

### 快速测试报告

```
测试日期: 2024-12-19
测试环境: Windows, Python 3.10.8, CPU
测试结果: ✅ 全部通过 (4/4)

详细结果:
- YOLO-RD: ✅ 通过 (6.55s)
- Knowledge DeepSORT: ✅ 通过 (3.46s)
- 边缘优化: ✅ 通过 (0.01s)
- 可视化: ✅ 通过 (0.36s)

总耗时: 10.38s
```

### 完整实验报告

```
实验日期: 2024-12-19
实验环境: Windows, Python 3.10.8, CPU

实验 1: 检索增强检测
- 状态: ✅ 完成
- 领域字典: 2 个（工业、交通）
- 检测目标: 测试通过

实验 2: 知识增强跟踪
- 状态: ✅ 完成
- 跟踪帧数: 10 帧
- 轨迹数: 测试通过

实验 3: 边缘计算优化
- 状态: ✅ 完成
- 融合机会: 已识别
- 量化对比: 已完成
- 可视化: 已生成

输出文件:
- optimization_results.json
- ./optimization_plots/ (4 个图表)
```

---

## 🚀 下一步

测试通过后，你可以：

1. **查看结果**: 检查生成的图表和报告
2. **运行实际视频**: 使用真实视频测试
3. **调整参数**: 优化性能和准确率
4. **部署应用**: 集成到实际项目中

---

## 📚 相关文档

- [快速开始](QUICKSTART.md) - 基础使用指南
- [优化文档](OPTIMIZATION_README.md) - 详细技术文档
- [实现总结](IMPLEMENTATION_SUMMARY.md) - 功能说明

---

## 💡 测试建议

### 开发阶段
```bash
# 频繁运行快速测试
python test_optimization.py
```

### 功能验证
```bash
# 运行完整实验
python experiments/run_optimization_experiments.py
```

### 性能评估
```bash
# 使用真实视频测试
python main.py --video real_video.mp4 --experiment
```

### 部署前
```bash
# 全面测试
python test_optimization.py
python experiments/run_optimization_experiments.py
python main.py --video test_video.mp4
```

---

**测试愉快！** 🧪
