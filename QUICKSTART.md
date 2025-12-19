# 🚀 快速开始指南

## 📋 目录
1. [环境准备](#环境准备)
2. [快速测试](#快速测试)
3. [基础使用](#基础使用)
4. [高级功能](#高级功能)
5. [常见问题](#常见问题)

---

## 环境准备

### 1. 安装依赖

```bash
# 基础依赖
pip install torch torchvision ultralytics

# CLIP 模型
pip install git+https://github.com/openai/CLIP.git

# 其他依赖
pip install opencv-python numpy scipy scikit-learn matplotlib tqdm
```

### 2. 下载模型

```bash
# YOLO 模型会自动下载
# CLIP 模型会在首次运行时自动下载
```

### 3. 验证安装

```bash
python test_optimization.py
```

预期输出：
```
🧪 测试优化功能 - 增强版
================================================================================
📋 环境信息:
   Python: 3.10.x
   PyTorch: 2.0.1
   CUDA: ✅ 可用
   GPU: NVIDIA GeForce RTX 3090

1️⃣ 测试 YOLO-RD (检索增强检测)...
   ✅ YOLO-RD 模块正常 (2.34s)

2️⃣ 测试 Knowledge DeepSORT (知识增强跟踪)...
   ✅ Knowledge DeepSORT 模块正常 (1.87s)

3️⃣ 测试边缘计算优化...
   ✅ 边缘优化模块正常 (3.21s)

4️⃣ 测试可视化工具...
   ✅ 可视化工具正常 (0.45s)

📊 测试总结
================================================================================
✅ 通过: 4
❌ 失败: 0

🎉 所有测试通过!
```

---

## 快速测试

### 方式 1: 使用默认配置

```bash
python main.py
```

这将：
- 自动创建测试视频
- 使用默认类别（person, car, bicycle, motorcycle）
- 输出到 `output_video.mp4`

### 方式 2: 指定视频文件

```bash
python main.py --video your_video.mp4 --output result.mp4
```

### 方式 3: 完整实验

```bash
python experiments/run_optimization_experiments.py
```

这将运行三个完整实验并生成所有可视化图表。

---

## 基础使用

### 1. 视频跟踪

```bash
# 基础跟踪
python main.py --video input.mp4 --output output.mp4

# 指定类别
python main.py --video input.mp4 --classes person car bicycle

# 使用 GPU
python main.py --video input.mp4 --device cuda

# 调整批处理大小
python main.py --video input.mp4 --batch-size 4
```

### 2. 启用实验模式（详细日志）

```bash
python main.py --video input.mp4 --experiment
```

输出示例：
```
🚀 初始化跟踪系统
================================================================================
目标类别: person, car, bicycle
批处理大小: 2
实验模式: 启用
YOLO-RD: 禁用
知识增强跟踪: 禁用
================================================================================

🎬 开始处理视频...
处理视频: 100%|████████████████████| 150/150 [00:45<00:00,  3.31it/s]

📊 处理统计
================================================================================
总帧数: 150
总轨迹数: 12

类别统计:
  person: 856
  car: 423
  bicycle: 127

性能指标:
  detection_time: 0.0234s (平均)
  tracking_time: 0.0089s (平均)
================================================================================
✅ 处理完成! 输出: output.mp4
```

### 3. 使用高级功能

```bash
# 启用 YOLO-RD 检索增强检测
python main.py --video input.mp4 --use-yolo-rd

# 启用知识增强跟踪
python main.py --video input.mp4 --use-knowledge-tracker

# 同时启用所有功能
python main.py \
  --video input.mp4 \
  --output output.mp4 \
  --use-yolo-rd \
  --use-knowledge-tracker \
  --experiment \
  --device cuda \
  --batch-size 4
```

---

## 高级功能

### 1. YOLO-RD 检索增强检测

```python
from models.yolo_rd import RetrievalAugmentedDetector
from ultralytics import YOLO
import clip

# 加载模型
yolo_model = YOLO("yolov8x-worldv2.pt")
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")

# 初始化 YOLO-RD
yolo_rd = RetrievalAugmentedDetector(
    yolo_model, 
    clip_model, 
    clip_preprocess, 
    device="cuda"
)

# 构建领域字典
industrial_classes = ["person", "helmet", "vest", "machine", "vehicle"]
yolo_rd.build_domain_dictionary("industrial", industrial_classes)

# 检测
import cv2
image = cv2.imread("test.jpg")
detections = yolo_rd.forward(image, domain="industrial")

# 查看结果
for det in detections:
    print(f"类别: {det['class_name']}, 置信度: {det['confidence']:.3f}")
    print(f"检索结果: {det['retrieved_classes'][:3]}")
```

### 2. 知识增强跟踪

```python
from tracking.knowledge_deepsort import KnowledgeDeepSORT

# 初始化跟踪器
tracker = KnowledgeDeepSORT(
    dictionary=yolo_rd.domain_dict,
    device="cuda"
)

# 多帧跟踪
for frame_id, detections in enumerate(detection_results):
    tracks = tracker.update(detections, frame_id)
    
    # 查看轨迹
    for track in tracks:
        print(f"ID: {track.id}, 类别: {track.class_name}")
        print(f"位置: {track.bbox}")
        print(f"命中次数: {track.hits}, 丢失次数: {track.misses}")

# 获取统计信息
stats = tracker.get_statistics()
print(f"总帧数: {stats['total_frames']}")
print(f"总检测数: {stats['total_detections']}")
print(f"ID 切换: {stats['id_switches']}")
```

### 3. 边缘计算优化

```python
from optimization.edge_optimization import EdgeOptimizationPipeline
import torch

# 创建模型
model = YourModel()

# 初始化优化流程
optimizer = EdgeOptimizationPipeline(model, device="cuda")

# 运行完整分析
dummy_input = torch.randn(1, 3, 224, 224)
results = optimizer.run_full_optimization_analysis(dummy_input)

# 保存结果
optimizer.save_results(results, "optimization_results.json")

# 生成可视化
from visualization.optimization_plots import OptimizationVisualizer
visualizer = OptimizationVisualizer(results)
visualizer.generate_all_plots("./plots")
```

### 4. 自定义配置

编辑 `config.py`:

```python
# 修改跟踪器配置
TRACKER_CONFIG = {
    "max_age": 50,  # 增加最大年龄
    "min_hits": 5,  # 提高最小命中次数
    "match_threshold": 0.6  # 降低匹配阈值
}

# 修改 YOLO-RD 配置
YOLO_RD_CONFIG = {
    "retrieval_top_k": 10,  # 增加检索数量
    "retrieval_threshold": 0.8,  # 提高检索阈值
    "enable_cache": True,
    "cache_size": 2000  # 增加缓存大小
}
```

---

## 常见问题

### Q1: CUDA 内存不足

**问题**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 减小批处理大小
python main.py --batch-size 1

# 或使用 CPU
python main.py --device cpu
```

### Q2: 模型文件未找到

**问题**: `FileNotFoundError: yolov8x-worldv2.pt not found`

**解决方案**:
```bash
# 下载模型到项目根目录
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-worldv2.pt

# 或修改 config.py 中的路径
MODEL_CONFIG = {
    "yolo_model_path": "/path/to/your/yolov8x-worldv2.pt"
}
```

### Q3: CLIP 模型下载慢

**问题**: CLIP 模型下载速度慢或失败

**解决方案**:
```python
# 使用国内镜像或手动下载
# 1. 手动下载 CLIP 模型
# 2. 设置环境变量
export CLIP_MODEL_PATH=/path/to/clip/models

# 或使用代理
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port
```

### Q4: 可视化图表不显示中文

**问题**: 图表中文显示为方块

**解决方案**:
```python
# 在 visualization/optimization_plots.py 开头添加
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
# 或
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
```

### Q5: 性能不如预期

**问题**: 处理速度慢

**解决方案**:
```bash
# 1. 确保使用 GPU
python main.py --device cuda

# 2. 增加批处理大小
python main.py --batch-size 8

# 3. 启用缓存
# 在代码中设置 enable_cache=True（默认已启用）

# 4. 使用量化模型
# 参考 optimization/edge_optimization.py 中的量化方法
```

### Q6: 跟踪 ID 频繁切换

**问题**: 目标 ID 不稳定

**解决方案**:
```python
# 调整跟踪器配置
TRACKER_CONFIG = {
    "max_age": 50,  # 增加
    "min_hits": 2,  # 减少
    "match_threshold": 0.8  # 增加
}

# 或启用知识增强跟踪
python main.py --use-knowledge-tracker
```

---

## 📚 更多资源

- **详细文档**: [OPTIMIZATION_README.md](OPTIMIZATION_README.md)
- **优化改进**: [OPTIMIZATION_IMPROVEMENTS.md](OPTIMIZATION_IMPROVEMENTS.md)
- **实现总结**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **原始文档**: [readme](readme)

---

## 🆘 获取帮助

### 命令行帮助

```bash
python main.py --help
```

### 查看示例

```bash
# 查看测试代码
cat test_optimization.py

# 查看实验脚本
cat experiments/run_optimization_experiments.py
```

### 调试模式

```bash
# 启用详细日志
python main.py --experiment --video input.mp4

# 查看性能统计
# 结果会在控制台输出
```

---

## ✅ 检查清单

在开始使用前，确保：

- [ ] Python 3.8+ 已安装
- [ ] PyTorch 已安装（建议 2.0+）
- [ ] CUDA 可用（可选，但推荐）
- [ ] 所有依赖已安装
- [ ] 测试通过 (`python test_optimization.py`)
- [ ] 模型文件可访问

---

## 🎯 下一步

1. ✅ 运行快速测试验证安装
2. ✅ 尝试基础视频跟踪
3. ✅ 探索高级功能（YOLO-RD、知识增强跟踪）
4. ✅ 运行完整实验查看优化效果
5. ✅ 根据需求调整配置

祝使用愉快！🚀
