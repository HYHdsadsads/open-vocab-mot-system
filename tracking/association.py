import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from config import ASSOCIATION_CONFIG

class SemanticAssociation:
    """语义约束的目标关联器"""
    def __init__(self, dictionary, kg_graph, proto_memory):
        self.dictionary = dictionary
        self.kg_graph = kg_graph
        self.proto_memory = proto_memory
        self.alpha = ASSOCIATION_CONFIG["iou_weight"]  # IoU权重
        self.beta = ASSOCIATION_CONFIG["appearance_weight"]  # 外观权重
        self.gamma = ASSOCIATION_CONFIG["semantic_weight"]  # 语义权重
        self.delta = ASSOCIATION_CONFIG["uncertainty_weight"]  # 不确定性权重

    def compute_cost_matrix(self, tracks, detections, visual_reliability, lang_reliability):
        """
        计算跟踪目标与检测结果的代价矩阵
        代价 = α*IoU代价 + β*外观代价 + γ*语义代价 + δ*不确定性代价
        """
        num_tracks = len(tracks)
        num_dets = len(detections)
        cost_matrix = np.zeros((num_tracks, num_dets))

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                det_bbox, det_feat, det_class, _ = det
                track_bbox = track.bbox
                track_class = track.class_name

                # 1. IoU代价（1 - IoU）
                iou = self.dictionary.compute_iou(track_bbox, det_bbox)
                cost_iou = 1 - iou

                # 2. 外观代价（1 - 特征相似度）
                feat_sim = self.proto_memory.get_proto_similarity(det_feat, track.id)
                cost_app = 1 - feat_sim

                # 3. 语义代价（基于知识图谱层级）
                sem_sim = 0.5  # 默认相似度
                try:
                    if track_class in self.dictionary.classes and det_class in self.dictionary.classes:
                        class1_idx = self.dictionary.classes.index(track_class)
                        class2_idx = self.dictionary.classes.index(det_class)
                        sem_sim = self.kg_graph.get_hierarchy_similarity(class1_idx, class2_idx)
                except Exception as e:
                    # 如果知识图谱不可用或索引错误，使用默认值
                    pass
                cost_sem = 1 - sem_sim

                # 4. 不确定性代价（高不确定性样本惩罚）
                unc_cost = (1 - visual_reliability[j]) + (1 - lang_reliability[j])
                cost_unc = unc_cost / 2.0  # 归一化

                # 总代价
                cost_matrix[i, j] = (
                    self.alpha * cost_iou +
                    self.beta * cost_app +
                    self.gamma * cost_sem +
                    self.delta * cost_unc
                )

        return cost_matrix

    def associate(self, tracks, detections, visual_reliability, lang_reliability):
        """匈牙利算法进行目标关联"""
        cost_matrix = self.compute_cost_matrix(tracks, detections, visual_reliability, lang_reliability)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)  # 最优匹配

        # 过滤低置信度匹配
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < ASSOCIATION_CONFIG["match_threshold"]:
                matches.append((r, c))
        return matches