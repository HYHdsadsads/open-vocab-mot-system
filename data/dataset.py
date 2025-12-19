"""数据集数据集加载模块（明确接口定义）
"""
import os
import cv2
import json
import xml.etree.ElementTree as ET
from glob import glob
from config import DATASET_CONFIG

class UniversalDatasetLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.config = DATASET_CONFIG["datasets"][dataset_name]
        self.base_path = self.config["path"].replace("${base_dir}", DATASET_CONFIG["base_dir"])
        self.format = self.config["format"]

    def load_frames(self, split="train", subset=None):
        """加载指定split的视频帧/图像"""
        if self.format in ["MOTChallenge"]:
            # MOT17/20格式：每个序列为独立文件夹
            sequences = self.config["split"][split] if not subset else [subset]
            frames = {}
            for seq in sequences:
                img_dir = os.path.join(self.base_path, seq, self.config["image_dir"])
                frame_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
                frames[seq] = [cv2.imread(fp) for fp in frame_paths]
            return frames

        elif self.format == "COCO":
            # LVIS/COCO格式：图像集中存储
            split_file = self.config["split"][split]
            with open(os.path.join(self.base_path, split_file), "r") as f:
                ann_data = json.load(f)
            img_paths = {img["id"]: os.path.join(self.base_path, self.config["image_dir"], img["file_name"])
                         for img in ann_data["images"]}
            return {img_id: cv2.imread(fp) for img_id, fp in img_paths.items()}

        elif self.format == "PascalVOC":
            # ODinW/VOC格式：按子数据集组织
            subdataset = subset if subset else self.config["subdatasets"][0]
            img_dir = os.path.join(self.base_path, subdataset, self.config["split"][split], self.config["image_dir"])
            frame_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
            return {os.path.basename(fp): cv2.imread(fp) for fp in frame_paths}

    def load_annotations(self, split="train", subset=None):
        """加载标注（统一转换为内部格式）"""
        if self.format in ["MOTChallenge"]:
            sequences = self.config["split"][split] if not subset else [subset]
            annotations = {}
            for seq in sequences:
                # 获取当前split对应的标注文件夹（gt/det）
                ann_dir = self.config.get("annotation_dir", {}).get(split, "gt")
                # 拼接路径时使用os.path.join统一分隔符
                ann_path = os.path.join(self.base_path, seq, ann_dir, f"{ann_dir}{self.config['annotation_suffix']}")

                # 检查文件是否存在
                if not os.path.exists(ann_path):
                    raise FileNotFoundError(f"标注文件不存在: {ann_path}")

                with open(ann_path, "r") as f:
                    lines = f.readlines()
                frame_anns = {}
                for line in lines:
                    parts = line.strip().split(",")
                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    x1, y1, w, h = map(float, parts[2:6])
                    x2, y2 = x1 + w, y1 + h  # 转换为[x1,y1,x2,y2]

                    # 检查标注框有效性（防止超出图像范围）
                    if w <= 0 or h <= 0 or x1 >= x2 or y1 >= y2:
                        print(f"跳过无效标注框: frame {frame_id}, track {track_id}")
                        continue

                    if frame_id not in frame_anns:
                        frame_anns[frame_id] = []
                    frame_anns[frame_id].append({
                        "track_id": track_id,
                        "class": self.config["classes"][0],
                        "bbox": [x1, y1, x2, y2]
                    })
                annotations[seq] = frame_anns
            return annotations

class MOTDataset:
    def __init__(self, base_dir):
        """
        多目标跟踪数据集加载器
        Args:
            base_dir: 数据集根目录，结构应为 {base_dir}/images/视频目录 和 {base_dir}/annotations/标注文件
        """
        self.base_dir = base_dir
        self.images_dir = os.path.join(base_dir, "images")
        self.annotations_dir = os.path.join(base_dir, "annotations")
        self.videos = os.listdir(self.images_dir) if os.path.exists(self.images_dir) else []

    def get_video_frames(self, video_id):
        """
        获取视频的所有帧
        Args:
            video_id: 视频标识符（对应images目录下的子目录名）
        Returns:
            list: 按帧顺序排列的图像列表，每个元素为cv2格式的图像数组 (H, W, 3)
        """
        frame_dir = os.path.join(self.images_dir, video_id)
        if not os.path.exists(frame_dir):
            raise ValueError(f"视频帧目录不存在: {frame_dir}")
        frames = sorted(os.listdir(frame_dir))
        return [cv2.imread(os.path.join(frame_dir, f)) for f in frames if f.endswith(('.jpg', '.png'))]

    def get_video_annotations(self, video_path):
        """
        获取视频的标注数据
        Args:
            video_id: 视频标识符
        Returns:
            dict: 标注字典，格式为 {
                "video_id": str,
                "frames": [
                    {
                        "frame_id": int,
                        "objects": [
                            {"track_id": int, "class": str, "bbox": [x1, y1, x2, y2]}  # bbox为像素坐标
                        ]
                    }
                ]
            }
        """
        annotations = {}  # key: frame_id, value: list of (track_id, bbox)
        label_path = os.path.join(video_path, "gt/gt.txt")  # MOT17标注路径
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x1, y1, w, h = map(float, parts[2:6])
                x2 = x1 + w
                y2 = y1 + h
                if frame_id not in annotations:
                    annotations[frame_id] = []
                annotations[frame_id].append((track_id, (x1, y1, x2, y2)))
        return annotations