"""
基于显式知识注入的跨模态关联机制
字典向量辅助 DeepSORT 关联 - 解决"跟不稳"的问题
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
from typing import List, Dict, Tuple, Optional
from config import TRACKER_CONFIG


class KnowledgeEnhancedFeatureExtractor(nn.Module):
    """知识增强特征提取器"""
    
    def __init__(self, visual_dim=512, semantic_dim=512, output_dim=256, device="cpu"):
        super().__init__()
        self.device = device
        
        # 视觉特征编码器
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        ).to(device)
        
        # 语义特征编码器
        self.semantic_encoder = nn.Sequential(
            nn.Linear(semantic_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        ).to(device)
        
        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            batch_first=True
        ).to(device)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        ).to(device)
        
    def forward(self, visual_feat: torch.Tensor, semantic_feat: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            visual_feat: 视觉特征 [batch, visual_dim]
            semantic_feat: 语义特征 [batch, semantic_dim]
            
        Returns:
            融合特征 [batch, output_dim]
        """
        # 编码
        v_encoded = self.visual_encoder(visual_feat)  # [batch, output_dim]
        s_encoded = self.semantic_encoder(semantic_feat)  # [batch, output_dim]
        
        # 跨模态注意力
        v_attended, _ = self.cross_attention(
            v_encoded.unsqueeze(1),  # [batch, 1, output_dim]
            s_encoded.unsqueeze(1),  # [batch, 1, output_dim]
            s_encoded.unsqueeze(1)
        )
        v_attended = v_attended.squeeze(1)  # [batch, output_dim]
        
        # 融合
        fused = self.fusion(torch.cat([v_attended, s_encoded], dim=-1))
        fused = torch.nn.functional.normalize(fused, dim=-1)
        
        return fused


class KnowledgeTrack:
    """知识增强的轨迹对象"""
    
    def __init__(self, track_id: int, bbox: List[float], visual_feat: np.ndarray,
                 semantic_feat: np.ndarray, class_name: str, frame_id: int):
        self.id = track_id
        self.bbox = bbox
        self.visual_feat = visual_feat
        self.semantic_feat = semantic_feat
        self.class_name = class_name
        self.start_frame = frame_id
        self.last_frame = frame_id
        
        # 特征历史
        self.visual_history = deque([visual_feat], maxlen=30)
        self.semantic_history = deque([semantic_feat], maxlen=30)
        
        # 状态
        self.hits = 1
        self.misses = 0
        self.age = 0
        
        # 知识增强特征
        self.knowledge_feat = None
        
    def update(self, bbox: List[float], visual_feat: np.ndarray, 
               semantic_feat: np.ndarray, frame_id: int):
        """更新轨迹"""
        self.bbox = bbox
        self.visual_feat = visual_feat
        self.semantic_feat = semantic_feat
        self.last_frame = frame_id
        
        self.visual_history.append(visual_feat)
        self.semantic_history.append(semantic_feat)
        
        self.hits += 1
        self.misses = 0
        self.age += 1
        
    def mark_missed(self):
        """标记未匹配"""
        self.misses += 1
        self.age += 1
        
    def get_averaged_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取平均特征"""
        visual_avg = np.mean(list(self.visual_history), axis=0)
        semantic_avg = np.mean(list(self.semantic_history), axis=0)
        return visual_avg, semantic_avg


class KnowledgeDeepSORT:
    """知识增强的 DeepSORT 跟踪器 - 优化版"""
    
    def __init__(self, dictionary, device="cpu"):
        self.dictionary = dictionary  # 领域字典
        self.device = device
        
        # 知识增强特征提取器
        self.feature_extractor = KnowledgeEnhancedFeatureExtractor(device=device)
        
        # 轨迹管理
        self.tracks: List[KnowledgeTrack] = []
        self.next_id = 1
        
        # 配置参数
        self.max_age = TRACKER_CONFIG.get("max_age", 30)
        self.min_hits = TRACKER_CONFIG.get("min_hits", 3)
        self.iou_threshold = 0.3
        
        # 代价矩阵权重
        self.w_iou = 0.3
        self.w_visual = 0.3
        self.w_semantic = 0.2
        self.w_knowledge = 0.2
        
        # 性能统计
        self.stats = {
            "total_frames": 0,
            "total_detections": 0,
            "total_tracks": 0,
            "avg_match_time": 0.0,
            "id_switches": 0
        }
        self._prev_track_ids = set()
        
    def compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """计算 IoU"""
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        inter_x1 = max(x1, x3)
        inter_y1 = max(y1, y3)
        inter_x2 = min(x2, x4)
        inter_y2 = min(y2, y4)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def compute_cost_matrix(self, detections: List[Dict]) -> np.ndarray:
        """
        计算代价矩阵（融合多种相似度）
        
        Args:
            detections: 检测结果列表
            
        Returns:
            代价矩阵 [num_tracks, num_detections]
        """
        num_tracks = len(self.tracks)
        num_dets = len(detections)
        
        if num_tracks == 0 or num_dets == 0:
            return np.zeros((num_tracks, num_dets))
        
        cost_matrix = np.zeros((num_tracks, num_dets))
        
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                # 1. IoU 代价
                iou = self.compute_iou(track.bbox, det["bbox"])
                cost_iou = 1 - iou
                
                # 2. 视觉特征代价
                visual_sim = cosine_similarity(
                    [track.visual_feat],
                    [det["visual_feature"]]
                )[0][0]
                cost_visual = 1 - visual_sim
                
                # 3. 语义特征代价
                semantic_feat = self.dictionary.get_embedding("general", det["class_name"])
                if semantic_feat is not None:
                    semantic_sim = cosine_similarity(
                        [track.semantic_feat],
                        [semantic_feat]
                    )[0][0]
                else:
                    semantic_sim = 0.5
                cost_semantic = 1 - semantic_sim
                
                # 4. 知识增强代价
                if track.knowledge_feat is not None:
                    # 使用知识增强特征计算相似度
                    visual_tensor = torch.tensor(det["visual_feature"], dtype=torch.float32).to(self.device).unsqueeze(0)
                    semantic_tensor = torch.tensor(semantic_feat if semantic_feat is not None else np.zeros(512), 
                                                  dtype=torch.float32).to(self.device).unsqueeze(0)
                    
                    with torch.no_grad():
                        det_knowledge_feat = self.feature_extractor(visual_tensor, semantic_tensor)
                        det_knowledge_feat = det_knowledge_feat.cpu().numpy().flatten()
                    
                    knowledge_sim = cosine_similarity(
                        [track.knowledge_feat],
                        [det_knowledge_feat]
                    )[0][0]
                else:
                    knowledge_sim = 0.5
                cost_knowledge = 1 - knowledge_sim
                
                # 加权融合
                total_cost = (
                    self.w_iou * cost_iou +
                    self.w_visual * cost_visual +
                    self.w_semantic * cost_semantic +
                    self.w_knowledge * cost_knowledge
                )
                
                cost_matrix[i, j] = total_cost
        
        return cost_matrix
    
    def update(self, detections: List[Dict], frame_id: int) -> List[KnowledgeTrack]:
        """
        更新跟踪器 - 优化版
        
        Args:
            detections: 检测结果
            frame_id: 当前帧 ID
            
        Returns:
            活跃轨迹列表
        """
        import time
        start_time = time.time()
        
        # 更新统计
        self.stats["total_frames"] += 1
        self.stats["total_detections"] += len(detections)
        
        # 计算代价矩阵
        cost_matrix = self.compute_cost_matrix(detections)
        
        # 匈牙利算法匹配
        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            matched_tracks = set()
            matched_dets = set()
            
            # 处理匹配
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < 0.7:  # 匹配阈值
                    track = self.tracks[row]
                    det = detections[col]
                    
                    # 获取语义特征
                    semantic_feat = self.dictionary.get_embedding("general", det["class_name"])
                    if semantic_feat is None:
                        semantic_feat = np.zeros(512)
                    
                    # 更新轨迹
                    track.update(det["bbox"], det["visual_feature"], semantic_feat, frame_id)
                    
                    # 更新知识增强特征
                    visual_tensor = torch.tensor(det["visual_feature"], dtype=torch.float32).to(self.device).unsqueeze(0)
                    semantic_tensor = torch.tensor(semantic_feat, dtype=torch.float32).to(self.device).unsqueeze(0)
                    
                    with torch.no_grad():
                        track.knowledge_feat = self.feature_extractor(visual_tensor, semantic_tensor)
                        track.knowledge_feat = track.knowledge_feat.cpu().numpy().flatten()
                    
                    matched_tracks.add(row)
                    matched_dets.add(col)
            
            # 处理未匹配的轨迹
            for i, track in enumerate(self.tracks):
                if i not in matched_tracks:
                    track.mark_missed()
            
            # 处理未匹配的检测（创建新轨迹）
            for j, det in enumerate(detections):
                if j not in matched_dets:
                    semantic_feat = self.dictionary.get_embedding("general", det["class_name"])
                    if semantic_feat is None:
                        semantic_feat = np.zeros(512)
                    
                    new_track = KnowledgeTrack(
                        track_id=self.next_id,
                        bbox=det["bbox"],
                        visual_feat=det["visual_feature"],
                        semantic_feat=semantic_feat,
                        class_name=det["class_name"],
                        frame_id=frame_id
                    )
                    
                    # 初始化知识增强特征
                    visual_tensor = torch.tensor(det["visual_feature"], dtype=torch.float32).to(self.device).unsqueeze(0)
                    semantic_tensor = torch.tensor(semantic_feat, dtype=torch.float32).to(self.device).unsqueeze(0)
                    
                    with torch.no_grad():
                        new_track.knowledge_feat = self.feature_extractor(visual_tensor, semantic_tensor)
                        new_track.knowledge_feat = new_track.knowledge_feat.cpu().numpy().flatten()
                    
                    self.tracks.append(new_track)
                    self.next_id += 1
        
        # 删除过期轨迹
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
        
        # 统计 ID 切换
        current_track_ids = {t.id for t in self.tracks}
        if self._prev_track_ids:
            disappeared = self._prev_track_ids - current_track_ids
            appeared = current_track_ids - self._prev_track_ids
            self.stats["id_switches"] += len(disappeared & appeared)
        self._prev_track_ids = current_track_ids
        
        # 更新性能统计
        match_time = time.time() - start_time
        alpha = 0.9
        self.stats["avg_match_time"] = alpha * self.stats["avg_match_time"] + (1 - alpha) * match_time
        self.stats["total_tracks"] = len(self.tracks)
        
        # 返回稳定轨迹
        return [t for t in self.tracks if t.hits >= self.min_hits]
    
    def get_statistics(self) -> Dict:
        """获取跟踪统计信息"""
        return self.stats.copy()
    
    def reset(self):
        """重置跟踪器"""
        self.tracks = []
        self.next_id = 1
