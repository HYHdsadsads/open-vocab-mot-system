"""多目标跟踪器（使用配置文件+消除循环依赖）"""
import numpy as np
import cv2
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from config import TRACKER_CONFIG
from config import SCENE_CONFIG  # 补充缺失的SCENE_CONFIG导入
from collections import deque  # 新增导入


# Track类定义（存储跟踪目标的ID、边界框、特征等）
class Track:
    def __init__(self, id, bbox, feature, class_name, start_frame):
        self.id = id
        self.bbox = bbox
        self.feature = feature
        self.class_name = class_name
        self.start_frame = start_frame
        self.hits = 1
        self.misses = 0
        self.feature_history = []  # 用于存储历史特征


class MultiObjectTracker:
    def __init__(self, dictionary, gpt_api_key=None):
        self.dictionary = dictionary
        self.gpt_api_key = gpt_api_key
        self.semantic_generator = None  # 可根据需要初始化
        self.semantic_matrix = TRACKER_CONFIG["semantic_matrix"]
        self.prev_frame = None
        self.flow = None
        self.tracks = []
        self.next_id = 1
        self.max_age = TRACKER_CONFIG["max_age"]
        self.min_hits = TRACKER_CONFIG["min_hits"]
        self.zero_shot_handler = None
        self.frame_counter = 0
        self.semantic_weights = {}
        self.adaptive_update_interval = TRACKER_CONFIG["adaptive_update_interval"]
        # 时空约束参数
        self.spatial_prior = SCENE_CONFIG["industrial"]["regions"]  # 场景先验
        self.temporal_smoothing = 0.8  # 时序平滑系数
        self.occlusion_threshold = 0.6  # 遮挡判断阈值
        self.occlusion_memory = {}  # 遮挡记忆 {track_id: (bbox, feature, frame)}
        self.scene_context = np.zeros(128)

    def set_zero_shot_handler(self, handler):
        self.zero_shot_handler = handler

    def update_semantic_matrix(self):
        if self.zero_shot_handler and self.semantic_generator:
            all_classes = self.dictionary.classes + list(self.zero_shot_handler.unknown_classes)
            self.semantic_matrix = self.semantic_generator.get_matrix(all_classes)
        elif self.gpt_api_key and len(self.dictionary.classes) > 0:
            try:
                import openai
                openai.api_key = self.gpt_api_key
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"生成类别间语义相似度矩阵（0-1）：{self.dictionary.classes}"},
                        {"role": "user", "content": "返回JSON格式"}
                    ]
                )
                import json
                self.semantic_matrix = json.loads(response.choices[0].message.content)
            except Exception as e:
                print(f"GPT语义矩阵生成失败: {e}")

    def compute_adaptive_weights(self, track, det):
        track_id = track.id
        if track_id not in self.semantic_weights:
            self.semantic_weights[track_id] = (0.6, 0.4)
        stability = min(1.0, track.hits / (track.hits + track.misses + 1e-5))
        visual_w = 0.4 + 0.4 * (1 - stability)
        semantic_w = 1 - visual_w
        self.semantic_weights[track_id] = (visual_w, semantic_w)
        return visual_w, semantic_w

    def compute_optical_flow(self, current_frame):
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return None
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        self.flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        self.prev_frame = current_gray
        return self.flow

    def compute_motion_similarity(self, track, det_bbox):
        if self.flow is None:
            return 0.0
        track_center = np.array([
            (track.bbox[0] + track.bbox[2]) / 2,
            (track.bbox[1] + track.bbox[3]) / 2
        ]).astype(int)
        flow_vec = self.flow[track_center[1], track_center[0]]
        predicted_center = track_center + flow_vec
        det_center = np.array([
            (det_bbox[0] + det_bbox[2]) / 2,
            (det_bbox[1] + det_bbox[3]) / 2
        ])
        dist = np.linalg.norm(predicted_center - det_center)
        return max(0, 1 - dist / 100)

    def compute_iou(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x1_, y1_, x2_, y2_ = bbox2
        inter_x1 = max(x1, x1_)
        inter_y1 = max(y1, y1_)
        inter_x2 = min(x2, x2_)
        inter_y2 = min(y2, y2_)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_ - x1_) * (y2_ - y1_)
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def compute_semantic_similarity(self, feature, class_name):
        atom = self.dictionary.get_atom(class_name)
        return cosine_similarity([feature], [atom])[0][0] if atom is not None else 0.0

    def check_scene_constraints(self, track, det_bbox):
        """检查目标是否符合场景约束"""
        # 1. 区域约束检查
        scene_regions = SCENE_CONFIG["industrial"]["regions"]
        track_center = [(track.bbox[0] + track.bbox[2]) / 2, (track.bbox[1] + track.bbox[3]) / 2]

        # 检查是否在工作区域内
        work_area = scene_regions["work_area"]
        in_work_area = (work_area[0] < track_center[0] < work_area[2] and
                        work_area[1] < track_center[1] < work_area[3])

        # 2. 速度约束检查
        prev_center = [(track.bbox[0] + track.bbox[2]) / 2, (track.bbox[1] + track.bbox[3]) / 2]
        curr_center = [(det_bbox[0] + det_bbox[2]) / 2, (det_bbox[1] + det_bbox[3]) / 2]
        speed = np.linalg.norm(np.array(curr_center) - np.array(prev_center))

        max_speed = SCENE_CONFIG["industrial"]["movement_limits"].get(
            track.class_name, SCENE_CONFIG["industrial"]["movement_limits"]["default"]
        )
        speed_valid = speed <= max_speed

        return in_work_area and speed_valid

    def compute_spatial_constraint(self, track, det_bbox):
        """计算空间约束得分（基于场景先验）"""
        track_center = [(track.bbox[0] + track.bbox[2]) / 2, (track.bbox[1] + track.bbox[3]) / 2]
        det_center = [(det_bbox[0] + det_bbox[2]) / 2, (det_bbox[1] + det_bbox[3]) / 2]

        # 1. 区域约束：目标是否在合理区域内
        in_work_area = (self.spatial_prior["work_area"][0] < det_center[0] < self.spatial_prior["work_area"][2] and
                        self.spatial_prior["work_area"][1] < det_center[1] < self.spatial_prior["work_area"][3])
        region_score = 1.0 if in_work_area else 0.3

        # 2. 空间平滑性：与历史位置的偏差
        pos_change = np.linalg.norm(np.array(det_center) - np.array(track_center))
        spatial_smooth = max(0, 1 - pos_change / (
                    max(self.spatial_prior["work_area"][2], self.spatial_prior["work_area"][3]) * 0.1))

        return 0.7 * region_score + 0.3 * spatial_smooth

    def compute_temporal_consistency(self, track, det_feat):
        """计算时序一致性得分（特征平滑性）"""
        if not hasattr(track, 'feature_history'):
            track.feature_history = deque(maxlen=5)
            track.feature_history.append(track.feature)

        # 特征时序平滑性
        feat_similarities = [cosine_similarity([det_feat], [f])[0][0] for f in track.feature_history]
        temporal_score = np.mean(feat_similarities)

        # 更新历史
        track.feature_history.append(det_feat)
        return temporal_score

    def handle_occlusion(self, track, detections, frame_num):
        """遮挡处理机制：结合运动预测和外观记忆"""
        if track.id in self.occlusion_memory:
            # 从遮挡中恢复：优先匹配记忆特征
            mem_bbox, mem_feat, mem_frame = self.occlusion_memory[track.id]
            for i, (det_bbox, det_feat, det_class) in enumerate(detections):
                # 运动预测匹配
                pred_bbox = self.predict_bbox(track, frame_num - mem_frame)
                iou = self.compute_iou(pred_bbox, det_bbox)
                # 外观特征匹配
                feat_sim = cosine_similarity([mem_feat], [det_feat])[0][0]
                if iou > 0.3 and feat_sim > 0.6:
                    del self.occlusion_memory[track.id]
                    return i  # 返回匹配的检测索引
            # 仍处于遮挡状态
            track.misses += 1
            return -1
        else:
            # 判断是否进入遮挡
            if track.misses > 2:
                self.occlusion_memory[track.id] = (track.bbox, track.feature, frame_num)
            return -1

    def _get_context_semantic_sim(self, class_a, class_b):
        """基于GPT的类别上下文相似度（如"工人"在"工厂"场景中与"机器"更相关）"""
        if not self.semantic_generator:
            return 0.5
        try:
            response = self.semantic_generator.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "评估两个物体在工业场景中的关联性（0-1），直接返回数值"},
                    {"role": "user", "content": f"{class_a}和{class_b}的关联性"}
                ]
            )
            return float(response.choices[0].message.content)
        except:
            return 0.5

    def _update_track(self, track, new_bbox, new_feature, new_class, frame_num):
        """更新现有跟踪目标的信息"""
        # 更新边界框（可使用平滑处理）
        track.bbox = [
            track.bbox[0] * self.temporal_smoothing + new_bbox[0] * (1 - self.temporal_smoothing),
            track.bbox[1] * self.temporal_smoothing + new_bbox[1] * (1 - self.temporal_smoothing),
            track.bbox[2] * self.temporal_smoothing + new_bbox[2] * (1 - self.temporal_smoothing),
            track.bbox[3] * self.temporal_smoothing + new_bbox[3] * (1 - self.temporal_smoothing)
        ]

        # 更新特征（保留历史特征用于时序一致性检查）
        track.feature = new_feature
        if hasattr(track, 'feature_history'):
            track.feature_history.append(new_feature)
        else:
            track.feature_history = deque([new_feature], maxlen=5)  # 保留最近5帧特征

        # 更新类别信息（如果有必要）
        if new_class != track.class_name:
            track.class_name = new_class

        # 更新跟踪状态
        track.hits += 1
        track.misses = 0  # 重置未命中计数

    def predict_bbox(self, track, frames_passed):
        """预测遮挡期间的边界框位置（简单线性预测）"""
        # 这里实现简单的运动预测，可根据需要扩展为更复杂的模型
        return track.bbox  # 简化实现，实际应基于历史运动轨迹预测

    def compute_cost_matrix(self, detections, frame_num, scene_context):
        """计算代价矩阵（包含场景上下文参数）"""
        num_tracks = len(self.tracks)
        num_dets = len(detections)
        cost_matrix = np.zeros((num_tracks, num_dets))

        for i, track in enumerate(self.tracks):
            for j, (det_bbox, det_feat, det_class) in enumerate(detections):
                # 计算各种相似度（可利用 scene_context 优化）
                iou = self.compute_iou(track.bbox, det_bbox)
                feat_sim = cosine_similarity([track.feature], [det_feat])[0][0]
                sem_sim = self.compute_semantic_similarity(det_feat, det_class)

                # 可根据场景上下文调整权重（例如工业场景更注重空间约束）
                if scene_context == "industrial":
                    spatial_score = self.compute_spatial_constraint(track, det_bbox)
                    cost_matrix[i, j] = 1 - (0.2 * iou + 0.5 * feat_sim + 0.2 * sem_sim + 0.1 * spatial_score)
                else:
                    cost_matrix[i, j] = 1 - (0.3 * iou + 0.5 * feat_sim + 0.2 * sem_sim)

        return cost_matrix

    def reset(self):
        """重置跟踪器状态，清除所有现有轨迹和相关数据"""
        self.tracks = []  # 清空现有轨迹
        self.next_id = 1  # 重置ID计数器
        self.prev_frame = None  # 清除上一帧图像
        self.flow = None  # 清除光流数据
        self.frame_counter = 0  # 重置帧计数器
        self.occlusion_memory = {}  # 清除遮挡记忆
        self.semantic_weights = {}  # 清除语义权重缓存
        # 重置轨迹特征历史（如果有需要）
        for track in self.tracks:
            if hasattr(track, 'feature_history'):
                track.feature_history.clear()

    def _add_new_track(self, bbox, feature, class_name, start_frame):
        """创建新的跟踪目标"""
        # 使用当前的 next_id 作为跟踪ID
        new_track = Track(
            id=self.next_id,
            bbox=bbox,
            feature=feature,
            class_name=class_name,
            start_frame=start_frame
        )
        self.tracks.append(new_track)
        self.next_id += 1  # 更新下一个可用ID

    def update(self, detections, frame_num):
        """更新跟踪器，加入遮挡处理"""
        matched_dets = set()
        # 先处理遮挡恢复
        for i, track in enumerate(self.tracks):
            det_idx = self.handle_occlusion(track, detections, frame_num)
            if det_idx != -1 and det_idx not in matched_dets:
                # 遮挡恢复匹配
                det_bbox, det_feat, det_class = detections[det_idx]
                self._update_track(track, det_bbox, det_feat, det_class, frame_num)
                matched_dets.add(det_idx)

        # 常规匹配（排除已匹配的检测）
        remaining_dets = [d for i, d in enumerate(detections) if i not in matched_dets]
        if len(self.tracks) > 0 and len(remaining_dets) > 0:
            # 传入scene_context参数
            scene_context = "industrial"  # 与场景配置保持一致
            cost_matrix = self.compute_cost_matrix(remaining_dets, frame_num, scene_context)
            track_indices, det_indices = linear_sum_assignment(cost_matrix)

            # 处理匹配结果
            for t_idx, d_idx in zip(track_indices, det_indices):
                if cost_matrix[t_idx, d_idx] < TRACKER_CONFIG.get("match_threshold", 0.5):  # 增加默认阈值
                    track = self.tracks[t_idx]
                    det_bbox, det_feat, det_class = remaining_dets[d_idx]
                    self._update_track(track, det_bbox, det_feat, det_class, frame_num)
                    # 记录已匹配的检测索引（原始索引）
                    original_idx = [i for i, d in enumerate(detections) if d == remaining_dets[d_idx]][0]
                    matched_dets.add(original_idx)

        # 处理未匹配的检测，创建新跟踪
        unmatched_dets = [i for i, d in enumerate(detections) if i not in matched_dets]
        for d_idx in unmatched_dets:
            det_bbox, det_feat, det_class = detections[d_idx]
            self._add_new_track(det_bbox, det_feat, det_class, frame_num)

        # 过滤过期跟踪（超过最大未命中次数）
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]

        # 只返回符合最小命中次数的有效跟踪
        return [t for t in self.tracks if t.hits >= self.min_hits]