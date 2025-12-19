"""
优化结果可视化工具
生成饼图、柱状图等
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import numpy as np
from typing import Dict, List
import json


class OptimizationVisualizer:
    """优化结果可视化器"""
    
    def __init__(self, results: Dict):
        self.results = results
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_latency_breakdown_pie(self, save_path: str = "latency_breakdown.png"):
        """绘制延迟分解饼图 - 优化版"""
        breakdown = self.results.get("breakdown_chart", {})
        
        if not breakdown:
            print("⚠️ 没有延迟分解数据")
            return
        
        labels = breakdown["labels"]
        sizes = breakdown["percentages"]
        
        # 优化的颜色方案
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        colors = colors[:len(labels)]
        
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # 突出显示最大的部分
        explode = [0.1 if s == max(sizes) else 0 for s in sizes]
        
        # 绘制饼图
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            explode=explode,
            shadow=True,
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        
        # 美化
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(13)
        
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        # 添加图例
        ax.legend(wedges, [f'{l}: {s:.1f}%' for l, s in zip(labels, sizes)],
                 title="Layer Types",
                 loc="center left",
                 bbox_to_anchor=(1, 0, 0.5, 1),
                 fontsize=11)
        
        ax.set_title('Inference Latency Breakdown', 
                    fontsize=18, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 延迟分解饼图已保存: {save_path}")
    
    def plot_quantization_comparison(self, save_path: str = "quantization_comparison.png"):
        """绘制量化策略对比柱状图"""
        quant_results = self.results.get("quantization_comparison", {})
        
        if not quant_results:
            print("⚠️ 没有量化对比数据")
            return
        
        precisions = []
        latencies = []
        model_sizes = []
        
        for precision in ["fp32", "fp16", "int8"]:
            if precision in quant_results and "avg_latency_ms" in quant_results[precision]:
                precisions.append(precision.upper())
                latencies.append(quant_results[precision]["avg_latency_ms"])
                model_sizes.append(quant_results[precision].get("model_size_mb", 0))
        
        if not precisions:
            print("⚠️ 没有有效的量化数据")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 延迟对比
        x = np.arange(len(precisions))
        bars1 = ax1.bar(x, latencies, color=['#3498db', '#2ecc71', '#e74c3c'])
        ax1.set_xlabel('Precision', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
        ax1.set_title('Inference Latency Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(precisions)
        ax1.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontweight='bold')
        
        # 模型大小对比
        bars2 = ax2.bar(x, model_sizes, color=['#3498db', '#2ecc71', '#e74c3c'])
        ax2.set_xlabel('Precision', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Model Size (MB)', fontsize=12, fontweight='bold')
        ax2.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(precisions)
        ax2.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 量化对比图已保存: {save_path}")
    
    def plot_fusion_analysis(self, save_path: str = "fusion_analysis.png"):
        """绘制算子融合分析图"""
        fusion_results = self.results.get("fusion_analysis", {})
        
        if not fusion_results:
            print("⚠️ 没有融合分析数据")
            return
        
        fusion_stats = fusion_results.get("fusion_stats", {})
        
        if not fusion_stats:
            print("⚠️ 没有融合统计数据")
            return
        
        patterns = list(fusion_stats.keys())
        counts = list(fusion_stats.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(patterns, counts, color='#9b59b6')
        ax.set_xlabel('Number of Fusion Opportunities', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fusion Pattern', fontsize=12, fontweight='bold')
        ax.set_title('Operator Fusion Analysis', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{int(width)}',
                   ha='left', va='center', fontweight='bold', fontsize=10)
        
        # 添加总结信息
        total_fusions = fusion_results.get("total_fusion_opportunities", 0)
        memory_saved = fusion_results.get("memory_saved_mb", 0)
        
        info_text = f"Total Fusions: {total_fusions}\nMemory Saved: {memory_saved:.2f} MB"
        ax.text(0.98, 0.02, info_text,
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='bottom',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 融合分析图已保存: {save_path}")
    
    def plot_speedup_comparison(self, save_path: str = "speedup_comparison.png"):
        """绘制加速比对比图"""
        quant_results = self.results.get("quantization_comparison", {})
        
        if not quant_results:
            print("⚠️ 没有量化对比数据")
            return
        
        precisions = []
        speedups = []
        
        for precision in ["fp16", "int8"]:
            if precision in quant_results and "speedup" in quant_results[precision]:
                precisions.append(precision.upper())
                speedups.append(quant_results[precision]["speedup"])
        
        if not precisions:
            print("⚠️ 没有加速比数据")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x = np.arange(len(precisions))
        bars = ax.bar(x, speedups, color=['#2ecc71', '#e74c3c'], width=0.6)
        
        ax.set_xlabel('Precision', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speedup (vs FP32)', fontsize=12, fontweight='bold')
        ax.set_title('Quantization Speedup Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(precisions)
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='FP32 Baseline')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}x',
                   ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 加速比对比图已保存: {save_path}")
    
    def generate_all_plots(self, output_dir: str = "./optimization_plots"):
        """生成所有可视化图表"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n📊 生成可视化图表...")
        print("=" * 60)
        
        self.plot_latency_breakdown_pie(f"{output_dir}/latency_breakdown.png")
        self.plot_quantization_comparison(f"{output_dir}/quantization_comparison.png")
        self.plot_fusion_analysis(f"{output_dir}/fusion_analysis.png")
        self.plot_speedup_comparison(f"{output_dir}/speedup_comparison.png")
        
        print("=" * 60)
        print(f"✅ 所有图表已保存到: {output_dir}")


def visualize_from_json(json_path: str, output_dir: str = "./optimization_plots"):
    """从 JSON 文件加载结果并可视化"""
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    visualizer = OptimizationVisualizer(results)
    visualizer.generate_all_plots(output_dir)


if __name__ == "__main__":
    # 示例用法
    import sys
    
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./optimization_plots"
        visualize_from_json(json_path, output_dir)
    else:
        print("用法: python optimization_plots.py <results.json> [output_dir]")
