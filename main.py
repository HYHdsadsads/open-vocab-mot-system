"""开放词汇多目标跟踪系统入口 - 优化版"""
import cv2
import os
import sys
import argparse
import torch
from pathlib import Path

from pipeline.system import OpenVocabMOTSystem
from config import DATASET_CONFIG, MODEL_CONFIG


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='开放词汇多目标跟踪系统')
    parser.add_argument('--video', type=str, default=None,
                        help='输入视频路径')
    parser.add_argument('--output', type=str, default='./output_video.mp4',
                        help='输出视频路径')
    parser.add_argument('--classes', type=str, nargs='+',
                        default=["person", "car", "bicycle", "motorcycle"],
                        help='目标类别列表')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='运行设备')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='批处理大小')
    parser.add_argument('--experiment', action='store_true',
                        help='启用实验模式（详细日志）')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='可视化跟踪结果')
    parser.add_argument('--use-yolo-rd', action='store_true',
                        help='使用 YOLO-RD 检索增强检测')
    parser.add_argument('--use-knowledge-tracker', action='store_true',
                        help='使用知识增强跟踪器')
    
    return parser.parse_args()


def check_environment():
    """检查运行环境"""
    print("=" * 60)
    print("🔍 环境检查")
    print("=" * 60)
    
    # 检查 CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 可用: {'✅' if cuda_available else '❌'}")
    if cuda_available:
        print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
    
    # 检查 Python 版本
    print(f"Python 版本: {sys.version.split()[0]}")
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 检查模型文件
    yolo_path = MODEL_CONFIG["yolo_model_path"]
    if os.path.exists(yolo_path):
        print(f"✅ YOLO 模型: {yolo_path}")
    else:
        print(f"⚠️ YOLO 模型未找到: {yolo_path}")
    
    print("=" * 60 + "\n")


def create_test_video_if_needed(video_path: str) -> str:
    """如果视频不存在，创建测试视频"""
    if video_path and os.path.exists(video_path):
        return video_path
    
    print("⚠️ 视频文件不存在，创建测试视频...")
    
    import numpy as np
    
    output_path = "test_video.mp4"
    width, height = 640, 480
    fps = 30
    duration = 5
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(duration * fps):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 添加移动的目标
        x1 = int(100 + i * 0.5) % (width - 50)
        cv2.rectangle(frame, (x1, 200), (x1 + 50, 280), (0, 255, 0), -1)
        cv2.putText(frame, "Person", (x1, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        x2 = int(400 - i * 0.7) % (width - 100)
        cv2.rectangle(frame, (x2, 300), (x2 + 100, 350), (0, 255, 255), -1)
        cv2.putText(frame, "Car", (x2, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ 测试视频已创建: {output_path}\n")
    
    return output_path


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 检查环境
    check_environment()
    
    # 确定设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🖥️ 使用设备: {device}\n")
    
    # 准备视频路径
    video_path = args.video
    if not video_path:
        video_path = os.path.join(DATASET_CONFIG["base_dir"], "test_video.mp4")
    
    video_path = create_test_video_if_needed(video_path)
    
    # 初始化跟踪系统
    print("=" * 60)
    print("🚀 初始化跟踪系统")
    print("=" * 60)
    print(f"目标类别: {', '.join(args.classes)}")
    print(f"批处理大小: {args.batch_size}")
    print(f"实验模式: {'启用' if args.experiment else '禁用'}")
    print(f"YOLO-RD: {'启用' if args.use_yolo_rd else '禁用'}")
    print(f"知识增强跟踪: {'启用' if args.use_knowledge_tracker else '禁用'}")
    print("=" * 60 + "\n")
    
    try:
        mot_system = OpenVocabMOTSystem(
            args.classes,
            experiment_mode=args.experiment,
            batch_size=args.batch_size
        )
        
        # 处理视频
        print("🎬 开始处理视频...\n")
        results = mot_system.process_video(
            video_path,
            output_path=args.output,
            visualize=args.visualize
        )
        
        # 打印统计信息
        if results:
            print("\n" + "=" * 60)
            print("📊 处理统计")
            print("=" * 60)
            print(f"总帧数: {len(results)}")
            
            # 统计轨迹
            all_track_ids = set()
            for frame_result in results:
                for obj in frame_result:
                    all_track_ids.add(obj['track_id'])
            
            print(f"总轨迹数: {len(all_track_ids)}")
            
            # 统计类别
            class_counts = {}
            for frame_result in results:
                for obj in frame_result:
                    cls = obj.get('class_name', 'unknown')
                    class_counts[cls] = class_counts.get(cls, 0) + 1
            
            print("\n类别统计:")
            for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cls}: {count}")
            
            # 性能指标
            if mot_system.perf_metrics:
                print("\n性能指标:")
                for metric, values in mot_system.perf_metrics.items():
                    if values:
                        import numpy as np
                        print(f"  {metric}: {np.mean(values):.4f}s (平均)")
            
            print("=" * 60)
            print(f"✅ 处理完成! 输出: {args.output}")
            print("=" * 60 + "\n")
        else:
            print("⚠️ 未生成跟踪结果")
    
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()