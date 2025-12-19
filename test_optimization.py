"""
快速测试优化功能 - 增强版
"""
import torch
import numpy as np
import sys
import time

print("=" * 80)
print("🧪 测试优化功能 - 增强版")
print("=" * 80)

# 测试环境
print(f"\n📋 环境信息:")
print(f"   Python: {sys.version.split()[0]}")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA: {'✅ 可用' if torch.cuda.is_available() else '❌ 不可用'}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

test_results = {
    "passed": 0,
    "failed": 0,
    "errors": []
}

# 测试 1: YOLO-RD
print("\n1️⃣ 测试 YOLO-RD (检索增强检测)...")
test_start = time.time()
try:
    from models.yolo_rd import DomainDictionary
    import clip
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    domain_dict = DomainDictionary(clip_model, device, enable_cache=True)
    domain_dict.build_from_classes("test", ["person", "car", "bicycle"])
    
    # 测试检索功能
    test_embedding = np.random.randn(512)
    test_embedding = test_embedding / np.linalg.norm(test_embedding)
    results = domain_dict.retrieve_similar(test_embedding, "test", top_k=2)
    
    assert len(results) <= 2, "检索结果数量错误"
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results), "检索结果格式错误"
    
    test_time = time.time() - test_start
    print(f"   ✅ YOLO-RD 模块正常 ({test_time:.2f}s)")
    test_results["passed"] += 1
except Exception as e:
    print(f"   ❌ YOLO-RD 测试失败: {e}")
    test_results["failed"] += 1
    test_results["errors"].append(f"YOLO-RD: {str(e)}")

# 测试 2: Knowledge DeepSORT
print("\n2️⃣ 测试 Knowledge DeepSORT (知识增强跟踪)...")
test_start = time.time()
try:
    from tracking.knowledge_deepsort import KnowledgeEnhancedFeatureExtractor, KnowledgeDeepSORT
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor = KnowledgeEnhancedFeatureExtractor(device=device)
    
    # 测试前向传播
    visual_feat = torch.randn(2, 512).to(device)
    semantic_feat = torch.randn(2, 512).to(device)
    
    with torch.no_grad():
        output = feature_extractor(visual_feat, semantic_feat)
    
    assert output.shape == (2, 256), f"输出形状错误: {output.shape}"
    
    # 测试跟踪器
    from models.yolo_rd import DomainDictionary
    import clip
    clip_model, _ = clip.load("ViT-B/32", device=device)
    domain_dict = DomainDictionary(clip_model, device)
    
    tracker = KnowledgeDeepSORT(domain_dict, device=device)
    assert hasattr(tracker, 'stats'), "跟踪器缺少统计功能"
    
    test_time = time.time() - test_start
    print(f"   ✅ Knowledge DeepSORT 模块正常 ({test_time:.2f}s)")
    test_results["passed"] += 1
except Exception as e:
    print(f"   ❌ Knowledge DeepSORT 测试失败: {e}")
    test_results["failed"] += 1
    test_results["errors"].append(f"Knowledge DeepSORT: {str(e)}")

# 测试 3: 边缘优化
print("\n3️⃣ 测试边缘计算优化...")
test_start = time.time()
try:
    from optimization.edge_optimization import OperatorFusionAnalyzer, QuantizationStrategy, EdgeOptimizationPipeline
    
    # 创建简单测试模型
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 64, 3)
            self.bn = torch.nn.BatchNorm2d(64)
            self.relu = torch.nn.ReLU()
            self.fc = torch.nn.Linear(64, 10)
        
        def forward(self, x):
            x = self.relu(self.bn(self.conv(x)))
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = TestModel()
    
    # 测试算子融合分析
    fusion_analyzer = OperatorFusionAnalyzer(model)
    fusion_report = fusion_analyzer.generate_fusion_report()
    
    assert "total_fusion_opportunities" in fusion_report, "融合报告格式错误"
    print(f"   ✅ 发现 {fusion_report['total_fusion_opportunities']} 个融合机会")
    
    # 测试量化策略
    device = "cuda" if torch.cuda.is_available() else "cpu"
    quant_strategy = QuantizationStrategy(model, device)
    dummy_input = torch.randn(1, 3, 32, 32)
    
    fp32_result = quant_strategy.benchmark_fp32(dummy_input, num_runs=10)
    assert "avg_latency_ms" in fp32_result, "基准测试结果格式错误"
    print(f"   ✅ FP32 延迟: {fp32_result['avg_latency_ms']:.2f} ms")
    
    # 测试完整流程
    pipeline = EdgeOptimizationPipeline(model, device)
    assert hasattr(pipeline, 'run_full_optimization_analysis'), "缺少完整分析方法"
    
    test_time = time.time() - test_start
    print(f"   ✅ 边缘优化模块正常 ({test_time:.2f}s)")
    test_results["passed"] += 1
except Exception as e:
    print(f"   ❌ 边缘优化测试失败: {e}")
    test_results["failed"] += 1
    test_results["errors"].append(f"边缘优化: {str(e)}")

# 测试 4: 可视化
print("\n4️⃣ 测试可视化工具...")
test_start = time.time()
try:
    from visualization.optimization_plots import OptimizationVisualizer
    
    # 创建测试数据
    test_data = {
        "breakdown_chart": {
            "labels": ["Convolution", "Linear", "Activation"],
            "values": [45.0, 30.0, 25.0],
            "percentages": [45.0, 30.0, 25.0]
        },
        "quantization_comparison": {
            "fp32": {"avg_latency_ms": 100.0, "model_size_mb": 400.0},
            "fp16": {"avg_latency_ms": 55.0, "model_size_mb": 200.0, "speedup": 1.8},
            "int8": {"avg_latency_ms": 35.0, "model_size_mb": 100.0, "speedup": 2.9}
        },
        "fusion_analysis": {
            "fusion_stats": {"Conv+BN+ReLU": 5, "Linear+ReLU": 3},
            "total_fusion_opportunities": 8,
            "memory_saved_mb": 15.5
        }
    }
    
    visualizer = OptimizationVisualizer(test_data)
    
    # 测试各个绘图方法存在
    assert hasattr(visualizer, 'plot_latency_breakdown_pie'), "缺少饼图方法"
    assert hasattr(visualizer, 'plot_quantization_comparison'), "缺少量化对比方法"
    assert hasattr(visualizer, 'plot_fusion_analysis'), "缺少融合分析方法"
    assert hasattr(visualizer, 'generate_all_plots'), "缺少批量生成方法"
    
    test_time = time.time() - test_start
    print(f"   ✅ 可视化工具正常 ({test_time:.2f}s)")
    test_results["passed"] += 1
except Exception as e:
    print(f"   ❌ 可视化测试失败: {e}")
    test_results["failed"] += 1
    test_results["errors"].append(f"可视化: {str(e)}")

print("\n" + "=" * 80)
print("📊 测试总结")
print("=" * 80)
print(f"✅ 通过: {test_results['passed']}")
print(f"❌ 失败: {test_results['failed']}")

if test_results["errors"]:
    print("\n错误详情:")
    for error in test_results["errors"]:
        print(f"  - {error}")

if test_results["failed"] == 0:
    print("\n🎉 所有测试通过!")
else:
    print(f"\n⚠️ {test_results['failed']} 个测试失败")

print("=" * 80)
print("\n💡 下一步:")
print("  1. 运行完整实验: python experiments/run_optimization_experiments.py")
print("  2. 查看文档: cat OPTIMIZATION_README.md")
print("  3. 运行主程序: python main.py --help")
print("=" * 80 + "\n")

sys.exit(0 if test_results["failed"] == 0 else 1)
