import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import entropy
from collections import Counter
from typing import List, Tuple
from config import RELIABILITY_CONFIG
import clip

class ReliabilityEstimator(nn.Module):
    """多维度模态可靠性评估器"""
    def __init__(self, device="cpu", clip_model=None):
        super().__init__()
        self.device = device
        self.clip_model = clip_model  # 添加CLIP模型引用
        # 温度校准网络（输出视觉/语言模态的温度参数）
        self.temp_calibrator = nn.Sequential(
            nn.Linear(512 + 2, 128),  # 视觉特征+2个统计特征
            nn.ReLU(),
            nn.Linear(128, 2)  # 输出视觉/语言温度
        ).to(device)

    def visual_reliability(self, image: np.ndarray, bbox: List[float]) -> float:
        """
        视觉可靠性：融合清晰度、遮挡程度、光照均匀度
        返回值范围：[0.2, 1.0]
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        roi = image[y1:y2, x1:x2]  # 目标区域

        # 1. 清晰度（拉普拉斯方差）
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
        sharpness = np.clip(laplacian / 300.0, 0.2, 1.0)  # 300为经验阈值

        # 2. 遮挡程度（ROI与图像边界的交叠）
        full_bbox = [0, 0, w, h]
        iou = self._compute_iou(bbox, full_bbox)
        occlusion = np.clip(iou, 0.2, 1.0)  # IOU越小遮挡越严重

        # 3. 光照均匀度（HSV亮度通道标准差）
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        brightness_std = np.std(hsv_roi[:, :, 2])
        light = np.clip(1 - brightness_std / 50.0, 0.2, 1.0)

        # 加权融合
        return 0.5 * sharpness + 0.3 * occlusion + 0.2 * light

    def language_reliability(self, class_name: str, freq_counter: Counter) -> float:
        """
        语言可靠性：融合类别频率与语义歧义度
        """
        # 1. 类别频率因子（频率越低越不可靠）
        freq = freq_counter.get(class_name, 1)
        freq_factor = np.clip(1 / (np.log(freq + 2)), 0.3, 1.0)

        # 2. 语义歧义度（基于CLIP文本特征熵）
        ambiguity_factor = 0.6  # 默认值
        if self.clip_model is not None:
            try:
                text = clip.tokenize([class_name]).to(self.device)
                with torch.no_grad():
                    text_feat = self.clip_model.encode_text(text)
                text_entropy = entropy(text_feat.softmax(dim=-1).cpu().numpy()[0])
                ambiguity = np.clip(1 - text_entropy / np.log(512), 0.3, 1.0)  # 512为特征维度
                ambiguity_factor = ambiguity
            except Exception as e:
                print(f"⚠️ 语义歧义度计算失败: {e}")

        return 0.6 * freq_factor + 0.4 * ambiguity_factor

    def compute_temperature(self, visual_feat: torch.Tensor, stats: List[float]) -> Tuple[float, float]:
        """计算温度参数（校准模态置信度）"""
        input_feat = torch.cat([visual_feat, torch.tensor(stats, device=self.device)], dim=0)
        temps = self.temp_calibrator(input_feat.unsqueeze(0))
        visual_temp = torch.clamp(temps[0, 0], 0.5, 2.0).item()  # 温度范围限制
        lang_temp = torch.clamp(temps[0, 1], 0.5, 2.0).item()
        return visual_temp, lang_temp

    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        inter_area = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        return inter_area / (area1 + area2 - inter_area + 1e-8)