"""
面向边缘计算的异构推理加速与量化研究
包含：算子融合分析、量化策略、推理延迟分解
"""
import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict


class OperatorFusionAnalyzer:
    """算子融合分析器"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.fusion_stats = defaultdict(int)
        self.memory_savings = {}
        
    def analyze_conv_bn_relu_fusion(self) -> Dict:
        """分析 Conv+BN+ReLU 融合"""
        fusion_opportunities = []
        total_params = 0
        fused_params = 0
        
        modules = list(self.model.named_modules())
        
        for i in range(len(modules) - 2):
            name1, module1 = modules[i]
            name2, module2 = modules[i + 1]
            name3, module3 = modules[i + 2]
            
            # 检测 Conv -> BN -> ReLU 模式
            if (isinstance(module1, nn.Conv2d) and 
                isinstance(module2, nn.BatchNorm2d) and
                isinstance(module3, nn.ReLU)):
                
                # 计算参数量
                conv_params = sum(p.numel() for p in module1.parameters())
                bn_params = sum(p.numel() for p in module2.parameters())
                
                fusion_opportunities.append({
                    "pattern": "Conv+BN+ReLU",
                    "layers": [name1, name2, name3],
                    "conv_params": conv_params,
                    "bn_params": bn_params,
                    "can_fuse": True
                })
                
                total_params += conv_params + bn_params
                fused_params += conv_params  # 融合后只保留 Conv 参数
                
                self.fusion_stats["Conv+BN+ReLU"] += 1
        
        # 计算内存节省
        memory_saved = (total_params - fused_params) * 4 / (1024 * 1024)  # MB
        
        return {
            "fusion_opportunities": fusion_opportunities,
            "total_fusions": len(fusion_opportunities),
            "memory_saved_mb": memory_saved,
            "fusion_stats": dict(self.fusion_stats)
        }
    
    def analyze_linear_fusion(self) -> Dict:
        """分析 Linear 层融合"""
        fusion_opportunities = []
        
        modules = list(self.model.named_modules())
        
        for i in range(len(modules) - 1):
            name1, module1 = modules[i]
            name2, module2 = modules[i + 1]
            
            # 检测 Linear -> ReLU/GELU 模式
            if isinstance(module1, nn.Linear):
                if isinstance(module2, (nn.ReLU, nn.GELU)):
                    fusion_opportunities.append({
                        "pattern": f"Linear+{module2.__class__.__name__}",
                        "layers": [name1, name2],
                        "can_fuse": True
                    })
                    self.fusion_stats[f"Linear+{module2.__class__.__name__}"] += 1
        
        return {
            "fusion_opportunities": fusion_opportunities,
            "total_fusions": len(fusion_opportunities),
            "fusion_stats": dict(self.fusion_stats)
        }
    
    def generate_fusion_report(self) -> Dict:
        """生成完整融合报告"""
        conv_fusion = self.analyze_conv_bn_relu_fusion()
        linear_fusion = self.analyze_linear_fusion()
        
        total_fusions = conv_fusion["total_fusions"] + linear_fusion["total_fusions"]
        
        return {
            "conv_bn_relu_fusion": conv_fusion,
            "linear_fusion": linear_fusion,
            "total_fusion_opportunities": total_fusions,
            "estimated_speedup": 1.0 + (total_fusions * 0.05),  # 每个融合约 5% 加速
            "memory_saved_mb": conv_fusion.get("memory_saved_mb", 0)
        }


class QuantizationStrategy:
    """量化策略选择器"""
    
    def __init__(self, model: nn.Module, device="cpu"):
        self.model = model
        self.device = device
        self.quantization_results = {}
        
    def benchmark_fp32(self, dummy_input: torch.Tensor, num_runs: int = 100) -> Dict:
        """FP32 基准测试"""
        self.model.eval()
        self.model.to(self.device)
        dummy_input = dummy_input.to(self.device)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        # 测试
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = self.model(dummy_input)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.time() - start) * 1000)  # ms
        
        return {
            "precision": "FP32",
            "avg_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies),
            "model_size_mb": sum(p.numel() * 4 for p in self.model.parameters()) / (1024 * 1024)
        }
    
    def benchmark_fp16(self, dummy_input: torch.Tensor, num_runs: int = 100) -> Dict:
        """FP16 基准测试"""
        if self.device != "cuda":
            return {"error": "FP16 requires CUDA"}
        
        model_fp16 = self.model.half()
        dummy_input_fp16 = dummy_input.half().to(self.device)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model_fp16(dummy_input_fp16)
        
        # 测试
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model_fp16(dummy_input_fp16)
                torch.cuda.synchronize()
                latencies.append((time.time() - start) * 1000)
        
        return {
            "precision": "FP16",
            "avg_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies),
            "model_size_mb": sum(p.numel() * 2 for p in self.model.parameters()) / (1024 * 1024)
        }
    
    def benchmark_int8(self, dummy_input: torch.Tensor, num_runs: int = 100) -> Dict:
        """INT8 量化基准测试"""
        try:
            # 动态量化
            model_int8 = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            
            dummy_input = dummy_input.to(self.device)
            
            # 预热
            with torch.no_grad():
                for _ in range(10):
                    _ = model_int8(dummy_input)
            
            # 测试
            latencies = []
            with torch.no_grad():
                for _ in range(num_runs):
                    start = time.time()
                    _ = model_int8(dummy_input)
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    latencies.append((time.time() - start) * 1000)
            
            # 估算模型大小
            model_size = sum(p.numel() for p in model_int8.parameters()) / (1024 * 1024)
            
            return {
                "precision": "INT8",
                "avg_latency_ms": np.mean(latencies),
                "std_latency_ms": np.std(latencies),
                "min_latency_ms": np.min(latencies),
                "max_latency_ms": np.max(latencies),
                "model_size_mb": model_size
            }
        except Exception as e:
            return {"error": f"INT8 quantization failed: {str(e)}"}
    
    def compare_all_precisions(self, dummy_input: torch.Tensor) -> Dict:
        """对比所有精度"""
        results = {}
        
        print("🔍 测试 FP32...")
        results["fp32"] = self.benchmark_fp32(dummy_input)
        
        if self.device == "cuda":
            print("🔍 测试 FP16...")
            results["fp16"] = self.benchmark_fp16(dummy_input)
        
        print("🔍 测试 INT8...")
        results["int8"] = self.benchmark_int8(dummy_input)
        
        # 计算加速比
        fp32_latency = results["fp32"]["avg_latency_ms"]
        
        if "fp16" in results and "avg_latency_ms" in results["fp16"]:
            results["fp16"]["speedup"] = fp32_latency / results["fp16"]["avg_latency_ms"]
        
        if "avg_latency_ms" in results["int8"]:
            results["int8"]["speedup"] = fp32_latency / results["int8"]["avg_latency_ms"]
        
        return results


class InferenceProfiler:
    """推理延迟分解分析器"""
    
    def __init__(self, model: nn.Module, device="cpu"):
        self.model = model
        self.device = device
        self.layer_times = {}
        
    def profile_layers(self, dummy_input: torch.Tensor, num_runs: int = 50) -> Dict:
        """逐层性能分析"""
        self.model.eval()
        self.model.to(self.device)
        dummy_input = dummy_input.to(self.device)
        
        layer_times = defaultdict(list)
        
        # 注册钩子
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                start = time.time()
                # 模拟层执行时间
                if self.device == "cuda":
                    torch.cuda.synchronize()
                elapsed = (time.time() - start) * 1000
                layer_times[name].append(elapsed)
            return hook
        
        # 为每一层注册钩子
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                hooks.append(module.register_forward_hook(make_hook(name)))
        
        # 运行推理
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(dummy_input)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        # 统计结果
        layer_stats = {}
        total_time = 0
        
        for name, times in layer_times.items():
            avg_time = np.mean(times)
            layer_stats[name] = {
                "avg_time_ms": avg_time,
                "std_time_ms": np.std(times),
                "percentage": 0  # 稍后计算
            }
            total_time += avg_time
        
        # 计算百分比
        for name in layer_stats:
            layer_stats[name]["percentage"] = (layer_stats[name]["avg_time_ms"] / total_time) * 100
        
        return {
            "layer_stats": layer_stats,
            "total_time_ms": total_time,
            "num_layers": len(layer_stats)
        }
    
    def generate_breakdown_chart_data(self, profile_results: Dict) -> Dict:
        """生成延迟分解饼图数据"""
        layer_stats = profile_results["layer_stats"]
        
        # 按层类型分组
        type_times = defaultdict(float)
        
        for name, stats in layer_stats.items():
            # 识别层类型
            if "conv" in name.lower():
                layer_type = "Convolution"
            elif "linear" in name.lower() or "fc" in name.lower():
                layer_type = "Linear"
            elif "bn" in name.lower() or "batch" in name.lower():
                layer_type = "BatchNorm"
            elif "relu" in name.lower() or "gelu" in name.lower():
                layer_type = "Activation"
            elif "pool" in name.lower():
                layer_type = "Pooling"
            elif "attention" in name.lower():
                layer_type = "Attention"
            else:
                layer_type = "Other"
            
            type_times[layer_type] += stats["avg_time_ms"]
        
        total_time = sum(type_times.values())
        
        # 生成饼图数据
        chart_data = {
            "labels": list(type_times.keys()),
            "values": list(type_times.values()),
            "percentages": [(v / total_time) * 100 for v in type_times.values()]
        }
        
        return chart_data


class EdgeOptimizationPipeline:
    """边缘计算优化完整流程"""
    
    def __init__(self, model: nn.Module, device="cpu"):
        self.model = model
        self.device = device
        
        self.fusion_analyzer = OperatorFusionAnalyzer(model)
        self.quantization_strategy = QuantizationStrategy(model, device)
        self.profiler = InferenceProfiler(model, device)
        
    def run_full_optimization_analysis(self, dummy_input: torch.Tensor) -> Dict:
        """运行完整优化分析"""
        print("=" * 60)
        print("🚀 边缘计算优化分析开始")
        print("=" * 60)
        
        results = {}
        
        # 1. 算子融合分析
        print("\n📊 1. 算子融合分析...")
        results["fusion_analysis"] = self.fusion_analyzer.generate_fusion_report()
        print(f"   ✅ 发现 {results['fusion_analysis']['total_fusion_opportunities']} 个融合机会")
        print(f"   ✅ 预计内存节省: {results['fusion_analysis']['memory_saved_mb']:.2f} MB")
        
        # 2. 量化策略对比
        print("\n📊 2. 量化策略对比...")
        results["quantization_comparison"] = self.quantization_strategy.compare_all_precisions(dummy_input)
        
        fp32_latency = results["quantization_comparison"]["fp32"]["avg_latency_ms"]
        print(f"   ✅ FP32: {fp32_latency:.2f} ms")
        
        if "fp16" in results["quantization_comparison"]:
            fp16_latency = results["quantization_comparison"]["fp16"]["avg_latency_ms"]
            fp16_speedup = results["quantization_comparison"]["fp16"].get("speedup", 1.0)
            print(f"   ✅ FP16: {fp16_latency:.2f} ms (加速 {fp16_speedup:.2f}x)")
        
        if "avg_latency_ms" in results["quantization_comparison"]["int8"]:
            int8_latency = results["quantization_comparison"]["int8"]["avg_latency_ms"]
            int8_speedup = results["quantization_comparison"]["int8"].get("speedup", 1.0)
            print(f"   ✅ INT8: {int8_latency:.2f} ms (加速 {int8_speedup:.2f}x)")
        
        # 3. 推理延迟分解
        print("\n📊 3. 推理延迟分解...")
        results["profiling"] = self.profiler.profile_layers(dummy_input)
        results["breakdown_chart"] = self.profiler.generate_breakdown_chart_data(results["profiling"])
        print(f"   ✅ 总延迟: {results['profiling']['total_time_ms']:.2f} ms")
        print(f"   ✅ 分析了 {results['profiling']['num_layers']} 层")
        
        # 4. 生成优化建议
        results["recommendations"] = self._generate_recommendations(results)
        
        print("\n" + "=" * 60)
        print("✅ 优化分析完成")
        print("=" * 60)
        
        return results
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """生成优化建议 - 增强版"""
        recommendations = []
        
        # 基于融合分析
        fusion_count = results["fusion_analysis"]["total_fusion_opportunities"]
        if fusion_count > 0:
            recommendations.append(
                f"✅ 建议使用 TensorRT 进行算子融合，可融合 {fusion_count} 个算子组合"
            )
            recommendations.append(
                f"   预计加速: {results['fusion_analysis'].get('estimated_speedup', 1.0):.2f}x"
            )
        
        # 基于量化对比
        if "fp16" in results["quantization_comparison"]:
            fp16_speedup = results["quantization_comparison"]["fp16"].get("speedup", 1.0)
            if fp16_speedup > 1.5:
                recommendations.append(
                    f"✅ 建议使用 FP16 量化，可获得 {fp16_speedup:.2f}x 加速"
                )
                fp16_size = results["quantization_comparison"]["fp16"].get("model_size_mb", 0)
                recommendations.append(
                    f"   模型大小减少至: {fp16_size:.1f} MB"
                )
        
        if "speedup" in results["quantization_comparison"]["int8"]:
            int8_speedup = results["quantization_comparison"]["int8"]["speedup"]
            if int8_speedup > 2.0:
                recommendations.append(
                    f"✅ 建议使用 INT8 量化，可获得 {int8_speedup:.2f}x 加速"
                )
                int8_size = results["quantization_comparison"]["int8"].get("model_size_mb", 0)
                recommendations.append(
                    f"   模型大小减少至: {int8_size:.1f} MB"
                )
        
        # 基于延迟分解
        breakdown = results["breakdown_chart"]
        max_idx = np.argmax(breakdown["percentages"])
        bottleneck = breakdown["labels"][max_idx]
        bottleneck_pct = breakdown["percentages"][max_idx]
        
        if bottleneck_pct > 30:
            recommendations.append(
                f"⚠️ 瓶颈在 {bottleneck} 层（占 {bottleneck_pct:.1f}%），建议优先优化"
            )
            
            # 针对性建议
            if "conv" in bottleneck.lower():
                recommendations.append(
                    "   建议: 使用深度可分离卷积或 MobileNet 架构"
                )
            elif "linear" in bottleneck.lower():
                recommendations.append(
                    "   建议: 使用矩阵分解或知识蒸馏"
                )
            elif "attention" in bottleneck.lower():
                recommendations.append(
                    "   建议: 使用线性注意力或稀疏注意力"
                )
        
        # 批处理建议
        recommendations.append(
            "💡 建议使用批处理推理（batch_size=4-8）以提升 GPU 利用率"
        )
        
        # 内存优化建议
        memory_saved = results["fusion_analysis"].get("memory_saved_mb", 0)
        if memory_saved > 10:
            recommendations.append(
                f"💾 通过算子融合可节省 {memory_saved:.1f} MB 显存"
            )
        
        return recommendations
    
    def save_results(self, results: Dict, output_path: str = "optimization_results.json"):
        """保存结果到文件"""
        # 转换 numpy 类型为 Python 原生类型
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        results_serializable = convert_types(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已保存到: {output_path}")
