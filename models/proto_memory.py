import numpy as np
import torch
from sklearn.cluster import DBSCAN
from collections import defaultdict, deque
from config import PROTO_MEMORY_CONFIG
from typing import List, Dict, Any

class ProtoMemory:
    """轨迹级原型记忆与在线聚类"""

    def __init__(self, device="cpu"):
        self.device = device
        self.visual_protos = defaultdict(deque)  # {track_id: [feat1, feat2, ...]}
        self.text_protos = {}  # {class_name: feat}
        self.combined_protos = {}  # {track_id: feat}
        self.ema_alpha = PROTO_MEMORY_CONFIG["ema_alpha"]  # 指数移动平均系数
        self.window_size = PROTO_MEMORY_CONFIG["window_size"]  # 原型更新窗口
        self.cluster_eps = PROTO_MEMORY_CONFIG["cluster_eps"]  # DBSCAN参数

    def update_visual_proto(self, track_id: int, feat: np.ndarray, frame_num: int):
        """更新轨迹视觉原型（EMA+滑动窗口）"""
        if track_id not in self.visual_protos:
            self.visual_protos[track_id] = deque(maxlen=self.window_size)
        self.visual_protos[track_id].append(feat)

        # 计算EMA原型
        if len(self.visual_protos[track_id]) == 1:
            self.combined_protos[track_id] = feat
        else:
            prev_proto = self.combined_protos[track_id]
            self.combined_protos[track_id] = self.ema_alpha * feat + (1 - self.ema_alpha) * prev_proto

    def update_text_proto(self, class_name: str, feat: np.ndarray):
        """更新类别文本原型"""
        if class_name not in self.text_protos:
            self.text_protos[class_name] = feat
        else:
            self.text_protos[class_name] = 0.7 * self.text_protos[class_name] + 0.3 * feat

    def discover_new_classes(self, unlabeled_feats: List[np.ndarray]) -> Dict[int, List[int]]:
        """
        在线聚类发现新类别
        返回：{cluster_id: [feat_indices]}
        """
        if len(unlabeled_feats) < 5:  # 样本不足时不聚类
            return {}

        # DBSCAN聚类（基于余弦距离）
        feats_np = np.stack(unlabeled_feats)
        clustering = DBSCAN(
            eps=self.cluster_eps,
            min_samples=3,
            metric="cosine"
        ).fit(feats_np)

        # 整理聚类结果（过滤噪声点）
        clusters = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            if label != -1:
                clusters[label].append(idx)
        return clusters

    def get_proto_similarity(self, feat: np.ndarray, track_id: int) -> float:
        """计算特征与轨迹原型的相似度"""
        if track_id not in self.combined_protos:
            return 0.0
        proto = self.combined_protos[track_id]
        return np.dot(feat, proto) / (np.linalg.norm(feat) * np.linalg.norm(proto) + 1e-8)