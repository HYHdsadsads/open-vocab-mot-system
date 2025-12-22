import json

# 读取实验结果
with open('optimization_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

print('=' * 80)
print('实验指标详细报告')
print('=' * 80)
print()

# 1. 算子融合分析
print('1. 算子融合分析')
print('-' * 80)
fusion = results['fusion_analysis']
print(f'发现融合机会: {fusion["total_fusion_opportunities"]} 个')
# 从linear_fusion中获取fusion_stats
fusion_stats = fusion.get("linear_fusion", {}).get("fusion_stats", {})
conv_bn_relu = fusion_stats.get("Conv+BN+ReLU", 0)
linear_relu = fusion_stats.get("Linear+ReLU", 0)
print(f'   - Conv+BN+ReLU 融合: {conv_bn_relu} 个')
print(f'   - Linear+ReLU 融合: {linear_relu} 个')
print(f'内存节省: {fusion["memory_saved_mb"]:.4f} MB')
print(f'预计加速比: {fusion["estimated_speedup"]}x')
print()

# 2. 量化策略对比
print('2. 量化策略对比')
print('-' * 80)
quant = results['quantization_comparison']
fp32 = quant['fp32']
int8 = quant['int8']
print(f'FP32 (基准):')
print(f'   - 平均延迟: {fp32["avg_latency_ms"]:.2f} ms')
print(f'   - 标准差: {fp32["std_latency_ms"]:.2f} ms')
print(f'   - 最小延迟: {fp32["min_latency_ms"]:.2f} ms')
print(f'   - 最大延迟: {fp32["max_latency_ms"]:.2f} ms')
print(f'   - 模型大小: {fp32["model_size_mb"]:.2f} MB')
print()
print(f'INT8 (量化):')
print(f'   - 平均延迟: {int8["avg_latency_ms"]:.2f} ms')
print(f'   - 标准差: {int8["std_latency_ms"]:.2f} ms')
print(f'   - 最小延迟: {int8["min_latency_ms"]:.2f} ms')
print(f'   - 最大延迟: {int8["max_latency_ms"]:.2f} ms')
print(f'   - 模型大小: {int8["model_size_mb"]:.2f} MB')
print(f'   - 加速比: {int8["speedup"]:.2f}x')
print(f'   - 模型压缩: {(1 - int8["model_size_mb"]/fp32["model_size_mb"])*100:.1f}%')
print()

# 3. 推理延迟分解
print('3. 推理延迟分解')
print('-' * 80)
profiling = results['profiling']
breakdown = results['breakdown_chart']
print(f'总延迟: {profiling["total_time_ms"]:.4f} ms')
print(f'分析层数: {profiling["num_layers"]} 层')
print()
print('各层类型占比:')
for label, percentage in zip(breakdown['labels'], breakdown['percentages']):
    if percentage > 0:
        print(f'   - {label}: {percentage:.1f}%')
print()

# 4. 优化建议
print('4. 优化建议')
print('-' * 80)
for i, rec in enumerate(results['recommendations'], 1):
    print(f'{i}. {rec}')
print()

print('=' * 80)
print('报告生成完成')
print('=' * 80)
