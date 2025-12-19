"""跨模态词典模块（完整实现版）"""
import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from PIL import Image
import torch.nn.functional as F
from collections import deque, Counter
import clip
from config import ZERO_SHOT_CONFIG, DICTIONARY_CONFIG, MODEL_CONFIG
from typing import List, Tuple, Dict, Any, Optional


class CrossModalDictionary:
    def __init__(self, clip_model, clip_preprocess, yolo_model, device="cpu", gpt_api_key=None):
        self.device = device or MODEL_CONFIG["device"]

        # 视觉投影网络：将原始视觉特征投影到与语言特征对齐的空间
        self.visual_projection = torch.nn.Linear(512, 512).to(self.device)
        # 初始化投影网络权重（提升训练稳定性）
        torch.nn.init.xavier_uniform_(self.visual_projection.weight)
        torch.nn.init.zeros_(self.visual_projection.bias)

        # 动态权重预测网络：根据模态可靠性和特征相似度预测融合权重
        self.weight_predictor = torch.nn.Sequential(
            torch.nn.Linear(512 * 2 + 2, 256),  # 新增模态可靠性输入（2维）
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 2),
            torch.nn.Softmax(dim=-1)
        ).to(self.device)

        # 优化器：合并投影网络和权重预测网络参数
        self.optimizer = torch.optim.Adam(
            list(self.visual_projection.parameters()) +
            list(self.weight_predictor.parameters()),
            lr=DICTIONARY_CONFIG.get("lr", 1e-4),
            weight_decay=1e-5  # 增加权重衰减防止过拟合
        )

        # 模型与预处理工具
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.yolo_model = yolo_model

        # 数据存储结构
        self.language_atoms = {}          # 语言原子特征 {class_name: np.ndarray(512,)}
        self.visual_atoms = {}            # 视觉原子特征 {class_name: List[np.ndarray(512,)]}
        self.combined_atoms = {}          # 融合原子特征 {class_name: np.ndarray(512,)}
        self.historical_atoms = {}        # 历史融合原子 {class_name: deque[np.ndarray(512,)]}
        self.feature_weights = {}         # 特征融合权重 {class_name: np.ndarray(2,)} 0:视觉权重, 1:语言权重
        self.classes = []                 # 已知类别列表

        # GPT相关配置（语义增强）
        self.gpt_api_key = gpt_api_key
        self.gpt_semantic_generator = None

        # 难样本挖掘参数（可通过配置文件调整）
        self.hard_neg_threshold = DICTIONARY_CONFIG.get("hard_neg_threshold", 0.7)
        self.hard_pos_threshold = DICTIONARY_CONFIG.get("hard_pos_threshold", 0.3)

        # 时间演化参数（保持特征稳定性）
        self.atom_evolution_window = ZERO_SHOT_CONFIG.get("atom_evolution_window", 10)
        self.temporal_consistency_weight = ZERO_SHOT_CONFIG.get("temporal_consistency_weight", 0.2)

        # 聚类参数（视觉原子构建）
        self.num_clusters = DICTIONARY_CONFIG.get("num_clusters", 5)  # 每个类别聚类数量

        # 初始化GPT客户端（可选）
        if gpt_api_key:
            try:
                import openai
                openai.api_key = gpt_api_key
                self.gpt_semantic_generator = openai
            except ImportError:
                print("⚠️ 请安装openai库以使用GPT语义增强功能: pip install openai")
            except Exception as e:
                print(f"⚠️ GPT客户端初始化失败: {e}")

    def triplet_loss_with_hard_mining(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        改进的三元组损失（难样本挖掘）
        目标：拉近同类别（anchor-positive）距离，拉远不同类别（anchor-negative）距离
        """
        # 计算欧氏距离
        pos_dist = F.pairwise_distance(anchor, positive, p=2)  # 正样本距离 [batch_size,]
        neg_dist = F.pairwise_distance(anchor, negative, p=2)  # 负样本距离 [batch_size,]

        # 难正样本：距离大于阈值的正样本对（难区分的同类别）
        hard_pos_mask = pos_dist > self.hard_pos_threshold
        # 难负样本：距离小于阈值的负样本对（难区分的不同类别）
        hard_neg_mask = neg_dist < self.hard_neg_threshold

        # 过滤有效难样本
        hard_pos_dist = pos_dist[hard_pos_mask]
        hard_neg_dist = neg_dist[hard_neg_mask]

        # 确保难样本数量匹配
        min_hard_count = min(len(hard_pos_dist), len(hard_neg_dist))
        if min_hard_count > 0:
            # 只使用匹配数量的难样本计算损失
            hard_pos_dist = hard_pos_dist[:min_hard_count]
            hard_neg_dist = hard_neg_dist[:min_hard_count]
            # 三元组损失：pos_dist - neg_dist + margin > 0 时产生损失
            loss = F.relu(hard_pos_dist - hard_neg_dist + DICTIONARY_CONFIG["triplet_loss_margin"]).mean()
        else:
            # 无难样本时，使用普通三元组损失
            loss = F.relu(pos_dist - neg_dist + DICTIONARY_CONFIG["triplet_loss_margin"]).mean()

        return loss

    def _get_current_bbox(self, detection: Tuple[List[float], np.ndarray, str]) -> List[float]:
        """从检测结果中安全获取边界框"""
        if not detection or len(detection) < 1:
            return [0.0, 0.0, 0.0, 0.0]
        bbox = detection[0]
        # 确保边界框格式正确（x1,y1,x2,y2）
        if len(bbox) != 4:
            return [0.0, 0.0, 0.0, 0.0]
        return [float(x) for x in bbox]

    def compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """计算两个边界框的交并比（IOU）"""
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2

        # 计算交集区域
        inter_x1 = max(x1, x3)
        inter_y1 = max(y1, y3)
        inter_x2 = min(x2, x4)
        inter_y2 = min(y2, y4)

        # 交集面积（防止负面积）
        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        # 两个边界框面积
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        # 并集面积（防止除以0）
        union_area = max(1e-8, area1 + area2 - inter_area)

        return inter_area / union_area

    def get_class_frequency(self) -> Counter:
        """获取每个类别的视觉特征出现频率（用于语言可靠性评估）"""
        freq = Counter()
        for cls_name, feat_list in self.visual_atoms.items():
            freq[cls_name] = len(feat_list)  # 特征数量即出现频率
        return freq

    def compute_modal_reliability(self,
                                 visual_feat: torch.Tensor,
                                 lang_feat: torch.Tensor,
                                 image: Optional[np.ndarray] = None,
                                 detection: Optional[Tuple[List[float], np.ndarray, str]] = None,
                                 class_name: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        多维度模态可靠性评估
        返回：(视觉可靠性, 语言可靠性)，值域 [0.2, 1.0]
        """
        # -------------------------- 1. 视觉可靠性评估 --------------------------
        # 融合：清晰度（拉普拉斯方差）+ 遮挡程度 + 光照均匀度
        visual_reliability = torch.tensor(0.8, device=self.device, dtype=torch.float32)  # 默认值

        if image is not None and detection is not None:
            try:
                # 1.1 清晰度评估（拉普拉斯方差越大越清晰）
                gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray_img, cv2.CV_64F).var()
                sharpness = np.clip(laplacian_var / 300.0, 0.2, 1.0)  # 300为经验阈值

                # 1.2 遮挡程度评估（检测框与图像边界的交叠）
                h, w = image.shape[:2]
                bbox = self._get_current_bbox(detection)
                full_img_bbox = [0.0, 0.0, float(w), float(h)]
                iou_with_full = self.compute_iou(bbox, full_img_bbox)
                occlusion_factor = np.clip(iou_with_full, 0.2, 1.0)  # IOU越小遮挡越严重

                # 1.3 光照均匀度评估（HSV空间亮度标准差）
                hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                brightness_std = np.std(hsv_img[:, :, 2])  # 亮度通道标准差
                light_factor = np.clip(1 - brightness_std / 50.0, 0.2, 1.0)  # 标准差越小越均匀

                # 加权融合视觉可靠性
                visual_reliability = torch.tensor(
                    0.5 * sharpness + 0.3 * occlusion_factor + 0.2 * light_factor,
                    device=self.device, dtype=torch.float32
                )
            except Exception as e:
                print(f"⚠️ 视觉可靠性计算失败: {e}")

        # -------------------------- 2. 语言可靠性评估 --------------------------
        # 融合：类别频率 + 语义歧义度
        lang_reliability = torch.tensor(0.7, device=self.device, dtype=torch.float32)  # 默认值

        if class_name:
            try:
                # 2.1 类别频率因子（频率越低，语言特征越不可靠）
                class_freq = self.get_class_frequency()
                freq = class_freq.get(class_name, 1)
                freq_factor = np.clip(1 / (np.log(freq + 2)), 0.3, 1.0)  # 对数平滑

                # 2.2 语义歧义度因子（基于GPT生成描述的熵）
                if self.gpt_semantic_generator:
                    ambiguity = self._compute_semantic_ambiguity(class_name)
                    ambiguity_factor = np.clip(1 - ambiguity, 0.3, 1.0)  # 熵越小歧义度越低
                else:
                    ambiguity_factor = 0.6  # 无GPT时使用经验值

                # 加权融合语言可靠性
                lang_reliability = torch.tensor(
                    0.6 * freq_factor + 0.4 * ambiguity_factor,
                    device=self.device, dtype=torch.float32
                )
            except Exception as e:
                print(f"⚠️ 语言可靠性计算失败: {e}")

        return visual_reliability, lang_reliability

    @staticmethod
    def cluster_visual_atoms(features: List[np.ndarray], n_clusters: int = 5) -> Optional[KMeans]:
        """
        静态方法：对视觉特征进行KMeans聚类，提取代表性视觉原子
        输入：特征列表 [N, 512]
        输出：KMeans聚类模型（含聚类中心）
        """
        # 1. 输入验证
        if not features:
            print("❌ 聚类失败：无输入特征")
            return None

        # 2. 统一特征维度并过滤无效特征
        feature_dim = 512
        valid_features = []
        for feat in features:
            # 过滤维度不符的特征
            if len(feat) != feature_dim:
                print(f"⚠️ 过滤无效特征（维度 {len(feat)}，期望 {feature_dim}）")
                continue
            # 过滤全零特征
            if np.allclose(feat, 0):
                print("⚠️ 过滤全零特征")
                continue
            valid_features.append(feat)

        # 3. 确保有足够的有效特征
        if len(valid_features) < 2:
            print(f"❌ 聚类失败：有效特征仅 {len(valid_features)} 个（需至少2个）")
            return None

        # 4. 调整聚类数量（不超过有效特征数）
        n_clusters = min(n_clusters, len(valid_features))
        if n_clusters < 1:
            n_clusters = 1

        # 5. 执行KMeans聚类（增加异常捕获）
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  # n_init=10提升稳定性
            kmeans.fit(np.array(valid_features))
            print(f"✅ 聚类完成：{len(valid_features)} 个特征聚为 {n_clusters} 类")
            return kmeans
        except Exception as e:
            print(f"❌ 聚类执行失败：{str(e)}")
            return None

    def _compute_semantic_ambiguity(self, class_name: str) -> float:
        """
        基于GPT生成描述的熵计算语义歧义度
        熵越大，语义越模糊；值域 [0, 1]
        """
        try:
            # 生成3次不同的视觉特征描述（增加随机性）
            responses = []
            prompts = [
                f"用3个关键词描述 {class_name} 的核心视觉特征",
                f"简要说明 {class_name} 的形状、颜色、结构等视觉属性（3个关键词）",
                f"{class_name} 的独特视觉标识是什么？用3个词回答"
            ]

            for prompt in prompts:
                response = self.gpt_semantic_generator.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7  # 控制随机性
                )
                responses.append(response.choices[0].message.content.strip().lower())

            # 构建词袋并计算熵
            words = []
            for resp in responses:
                # 提取关键词（过滤标点和无意义词）
                valid_words = [w for w in resp.split() if len(w) > 1 and w not in ["和", "的", "是", "有"]]
                words.extend(valid_words)

            if not words:
                print(f"⚠️ {class_name} 无有效语义关键词")
                return 0.5

            # 计算词频分布的熵
            word_counts = Counter(words)
            total_words = len(words)
            probs = [count / total_words for count in word_counts.values()]
            entropy = -sum(p * np.log(p + 1e-8) for p in probs)  # 防止log(0)
            normalized_entropy = entropy / np.log(len(words) + 1e-8)  # 归一化到 [0, 1]

            print(f"📊 {class_name} 语义歧义度：{normalized_entropy:.3f}")
            return np.clip(normalized_entropy, 0.0, 1.0)
        except Exception as e:
            print(f"⚠️ 语义歧义度计算失败：{e}")
            return 0.5  # 出错时返回中间值

    def dynamic_feature_fusion(self,
                              visual_feat: torch.Tensor,
                              lang_feat: torch.Tensor,
                              image: Optional[np.ndarray] = None,
                              detection: Optional[Tuple[List[float], np.ndarray, str]] = None,
                              class_name: Optional[str] = None) -> torch.Tensor:
        """
        动态特征融合（核心功能）
        结合：1. 特征相似度 2. 模态可靠性 3. 动态权重预测
        输入：
            visual_feat: 视觉特征 [512,] 或 [batch_size, 512]
            lang_feat: 语言特征 [512,] 或 [batch_size, 512]
            image: 原始图像（用于可靠性评估）
            detection: 检测结果（bbox, feat, class_name）
            class_name: 目标类别名称
        输出：融合特征 [512,] 或 [batch_size, 512]
        """
        # 1. 统一输入维度（确保为2D张量 [batch_size, 512]）
        is_single = False
        if visual_feat.dim() == 1:
            visual_feat = visual_feat.unsqueeze(0)
            lang_feat = lang_feat.unsqueeze(0)
            is_single = True

        # 2. 计算模态可靠性（视觉+语言）
        vis_rel, lang_rel = self.compute_modal_reliability(
            visual_feat, lang_feat, image, detection, class_name
        )
        # 扩展为batch维度
        vis_rel = vis_rel.unsqueeze(0).repeat(visual_feat.shape[0], 1)  # [batch_size, 1]
        lang_rel = lang_rel.unsqueeze(0).repeat(visual_feat.shape[0], 1)  # [batch_size, 1]

        # 3. 计算特征相似度（视觉与语言特征的余弦相似度）
        feat_sim = F.cosine_similarity(visual_feat, lang_feat, dim=-1).unsqueeze(1)  # [batch_size, 1]
        feat_sim = torch.sigmoid(feat_sim)  # 归一化到 [0, 1]

        # 4. 动态权重预测（输入：特征拼接 + 可靠性 + 相似度）
        cat_feat = torch.cat([
            visual_feat, lang_feat,  # 特征 [batch_size, 1024]
            vis_rel, lang_rel,       # 可靠性 [batch_size, 2]
            feat_sim                 # 相似度 [batch_size, 1]
        ], dim=-1)  # [batch_size, 1024+2+1=1027]

        raw_weights = self.weight_predictor(cat_feat)  # [batch_size, 2] 原始权重

        # 5. 权重调整（结合模态可靠性）
        adjusted_vis_weight = raw_weights[:, 0:1] * vis_rel  # 视觉权重 = 预测权重 * 视觉可靠性
        adjusted_lang_weight = raw_weights[:, 1:2] * lang_rel  # 语言权重 = 预测权重 * 语言可靠性

        # 6. 权重归一化（确保和为1）
        total_weight = adjusted_vis_weight + adjusted_lang_weight + 1e-8
        normalized_vis_weight = adjusted_vis_weight / total_weight
        normalized_lang_weight = adjusted_lang_weight / total_weight

        # 7. 特征融合（加权求和）
        fused_feat = normalized_vis_weight * visual_feat + normalized_lang_weight * lang_feat
        # 特征归一化（提升后续计算稳定性）
        fused_feat = F.normalize(fused_feat, dim=-1, p=2)

        # 8. 恢复原始维度（单样本时返回1D张量）
        if is_single:
            fused_feat = fused_feat.squeeze(0)

        # 记录融合权重（单样本时）
        if is_single and class_name:
            self.feature_weights[class_name] = np.array([
                normalized_vis_weight.item(),
                normalized_lang_weight.item()
            ])

        return fused_feat

    def generate_language_embedding_with_gpt(self, class_name: str) -> Optional[np.ndarray]:
        """
        使用GPT生成增强的语言嵌入（比基础CLIP文本特征更精准）
        输出：512维语言特征
        """
        if not self.gpt_semantic_generator:
            print(f"⚠️ 无GPT API密钥，使用基础CLIP文本特征 for {class_name}")
            return None

        try:
            # 生成详细的视觉描述
            response = self.gpt_semantic_generator.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是计算机视觉专家，需要生成目标的详细视觉描述，用于特征提取"},
                    {"role": "user", "content": f"详细描述 {class_name} 的视觉特征，包括：形状、颜色、大小比例、表面纹理、典型姿态/状态、与其他相似目标的区别。用3-5句话描述，突出可区分的视觉属性。"}
                ],
                temperature=0.6
            )
            detailed_desc = response.choices[0].message.content.strip()
            print(f"📝 {class_name} GPT描述：{detailed_desc}")

            # 编码为CLIP特征
            text = clip.tokenize([detailed_desc]).to(self.device)
            with torch.no_grad():
                text_feat = self.clip_model.encode_text(text)
                text_feat = F.normalize(text_feat, dim=-1)  # 归一化

            return text_feat.cpu().numpy().flatten()
        except Exception as e:
            print(f"⚠️ GPT语言特征生成失败 for {class_name}: {e}")
            return None

    def align_features_with_triplet(self,
                                   visual_feats: List[np.ndarray],
                                   text_feats: List[np.ndarray],
                                   class_labels: List[str]) -> float:
        """
        使用三元组损失对齐视觉和语言特征（训练核心）
        输入：
            visual_feats: 视觉特征列表 [N, 512]
            text_feats: 语言特征列表 [N, 512]
            class_labels: 类别标签列表 [N,]
        输出：损失值
        """
        # 样本数量检查（太少则不训练）
        if len(visual_feats) < 32 or len(text_feats) < 32 or len(class_labels) < 32:
            print(f"⚠️ 样本不足，跳过特征对齐（需至少32个样本，当前 {len(visual_feats)} 个）")
            return 0.0

        # 转换为张量并移动到设备
        visual_tensor = torch.tensor(np.array(visual_feats), dtype=torch.float32).to(self.device)
        text_tensor = torch.tensor(np.array(text_feats), dtype=torch.float32).to(self.device)

        # 视觉特征投影（对齐到语言特征空间）
        projected_visual = self.visual_projection(visual_tensor)
        projected_visual = F.normalize(projected_visual, dim=-1)  # 归一化

        # 采样锚点、正样本、负样本（批量采样128个）
        batch_size = min(128, len(visual_feats))
        anchor_indices = torch.randint(0, len(visual_feats), (batch_size,), device=self.device)
        anchors = projected_visual[anchor_indices]  # [batch_size, 512]

        # 正样本：与锚点同类别且距离最远的样本（难正样本）
        positives = []
        for idx in anchor_indices:
            cls = class_labels[idx]
            # 同类别样本索引
            same_cls_indices = [i for i, label in enumerate(class_labels) if label == cls and i != idx]
            if not same_cls_indices:
                # 无同类别样本时，使用锚点自身
                positives.append(anchors[len(positives)])
                continue
            # 选择距离最远的同类别样本（难正样本）
            same_cls_feats = projected_visual[same_cls_indices]
            dists = F.pairwise_distance(anchors[len(positives):len(positives)+1], same_cls_feats)
            hard_pos_idx = same_cls_indices[torch.argmax(dists)]
            positives.append(projected_visual[hard_pos_idx])
        positives = torch.stack(positives)  # [batch_size, 512]

        # 负样本：与锚点不同类别且距离最近的样本（难负样本）
        negatives = []
        for idx in anchor_indices:
            cls = class_labels[idx]
            # 不同类别样本索引
            diff_cls_indices = [i for i, label in enumerate(class_labels) if label != cls]
            if not diff_cls_indices:
                # 无不同类别样本时，使用随机特征
                negatives.append(torch.randn_like(anchors[0], device=self.device))
                continue
            # 选择距离最近的不同类别样本（难负样本）
            diff_cls_feats = projected_visual[diff_cls_indices]
            dists = F.pairwise_distance(anchors[len(negatives):len(negatives)+1], diff_cls_feats)
            hard_neg_idx = diff_cls_indices[torch.argmin(dists)]
            negatives.append(projected_visual[hard_neg_idx])
        negatives = torch.stack(negatives)  # [batch_size, 512]

        # 计算三元组损失并反向传播
        self.optimizer.zero_grad()
        loss = self.triplet_loss_with_hard_mining(anchors, positives, negatives)
        loss.backward()
        self.optimizer.step()  # 更新模型参数

        return loss.item()

    def initialize_from_classes(self, class_names: List[str]):
        """
        从类别列表初始化跨模态词典
        1. 生成语言原子特征（优先GPT增强， fallback到基础CLIP）
        2. 初始化数据结构
        """
        if not class_names:
            raise ValueError("❌ 类别列表不能为空")

        self.classes = class_names
        # 初始化历史原子队列（限制窗口大小）
        self.historical_atoms = {cls: deque(maxlen=self.atom_evolution_window) for cls in class_names}
        # 初始化视觉原子存储
        self.visual_atoms = {cls: [] for cls in class_names}

        print(f"🚀 初始化跨模态词典，类别数：{len(class_names)}")
        for cls in class_names:
            # 优先使用GPT生成增强语言特征
            gpt_lang_feat = self.generate_language_embedding_with_gpt(cls)
            if gpt_lang_feat is not None and len(gpt_lang_feat) == 512:
                self.language_atoms[cls] = gpt_lang_feat
                continue

            # Fallback：使用基础CLIP文本特征
            try:
                text = clip.tokenize([f"a photo of a {cls}"]).to(self.device)
                with torch.no_grad():
                    clip_lang_feat = self.clip_model.encode_text(text)
                    clip_lang_feat = F.normalize(clip_lang_feat, dim=-1)
                self.language_atoms[cls] = clip_lang_feat.cpu().numpy().flatten()
                print(f"✅ {cls} 基础CLIP语言特征初始化完成")
            except Exception as e:
                print(f"❌ {cls} 语言特征初始化失败: {e}")
                self.language_atoms[cls] = np.random.randn(512)  # 异常时使用随机特征

    def extract_visual_features(self, image: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """
        提取单个目标的视觉特征
        输入：原始图像 + 目标边界框（x1,y1,x2,y2）
        输出：512维CLIP视觉特征
        """
        # 边界框有效性检查
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # 提取目标ROI（感兴趣区域）
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            print("⚠️ 目标ROI为空，跳过特征提取")
            return None

        # 转换为PIL图像并应用CLIP预处理
        try:
            pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            preprocessed = self.clip_preprocess(pil_roi).unsqueeze(0).to(self.device)

            # 提取CLIP视觉特征
            with torch.no_grad():
                visual_feat = self.clip_model.encode_image(preprocessed)
                visual_feat = F.normalize(visual_feat, dim=-1)

            return visual_feat.cpu().numpy().flatten()
        except Exception as e:
            print(f"⚠️ 单个视觉特征提取失败: {e}")
            return None

    def extract_visual_features_batch(self, image: np.ndarray, bboxes: List[List[float]]) -> List[np.ndarray]:
        """
        批量提取目标视觉特征（提升效率）
        输入：原始图像 + 边界框列表
        输出：视觉特征列表 [N, 512]
        """
        if not bboxes:
            return []

        preprocessed_imgs = []
        valid_indices = []  # 记录有效边界框索引

        for idx, bbox in enumerate(bboxes):
            # 边界框有效性检查
            x1, y1, x2, y2 = map(int, bbox)
            h, w = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # 提取ROI
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                print(f"⚠️ 边界框 {idx} ROI为空，跳过")
                continue

            # 预处理并保存
            try:
                pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                preprocessed = self.clip_preprocess(pil_roi)
                preprocessed_imgs.append(preprocessed)
                valid_indices.append(idx)
            except Exception as e:
                print(f"⚠️ 边界框 {idx} 预处理失败: {e}")
                continue

        # 批量提取特征
        if not preprocessed_imgs:
            return []

        try:
            batch_imgs = torch.stack(preprocessed_imgs).to(self.device)
            with torch.no_grad():
                batch_feats = self.clip_model.encode_image(batch_imgs)
                batch_feats = F.normalize(batch_feats, dim=-1)
            # 转换为numpy数组并返回
            return [feat.cpu().numpy().flatten() for feat in batch_feats]
        except Exception as e:
            print(f"⚠️ 批量视觉特征提取失败: {e}")
            return []

    def update_visual_atoms(self, class_name: str, visual_feat: np.ndarray):
        """
        更新指定类别的视觉原子特征
        1. 过滤无效特征
        2. 累计特征后聚类提取代表性原子
        """
        if class_name not in self.classes:
            print(f"⚠️ 类别 {class_name} 不在词典中，跳过视觉原子更新")
            return

        # 过滤无效特征
        if len(visual_feat) != 512 or np.allclose(visual_feat, 0):
            print(f"⚠️ 无效视觉特征，跳过 {class_name} 原子更新")
            return

        # 添加到视觉特征列表
        self.visual_atoms[class_name].append(visual_feat)
        print(f"📥 {class_name} 累计视觉特征数：{len(self.visual_atoms[class_name])}")

        # 当特征数达到阈值时，聚类更新视觉原子
        cluster_threshold = DICTIONARY_CONFIG.get("cluster_threshold", 20)  # 聚类阈值（20个特征）
        if len(self.visual_atoms[class_name]) >= cluster_threshold:
            # 执行聚类
            kmeans = self.cluster_visual_atoms(
                self.visual_atoms[class_name],
                n_clusters=self.num_clusters
            )
            if kmeans is not None:
                # 聚类中心作为新的视觉原子
                self.visual_atoms[class_name] = kmeans.cluster_centers_.tolist()
                print(f"✅ {class_name} 视觉原子更新完成（{self.num_clusters} 个聚类中心）")
            else:
                # 聚类失败时，保留最近20个特征
                self.visual_atoms[class_name] = self.visual_atoms[class_name][-20:]

    def align_and_combine(self):
        """
        核心流程：特征对齐 + 动态融合 + 时间演化
        1. 使用三元组损失对齐视觉和语言特征
        2. 动态融合生成跨模态原子
        3. 结合历史原子保持时间一致性
        """
        print("🔄 开始特征对齐与融合...")
        for class_name in self.classes:
            # 检查必要数据是否存在
            if class_name not in self.language_atoms or len(self.visual_atoms[class_name]) == 0:
                print(f"⚠️ {class_name} 数据不完整，跳过融合")
                continue

            # 准备对齐数据
            visual_feats = self.visual_atoms[class_name]  # [M, 512]
            lang_feat = self.language_atoms[class_name]    # [512,]
            # 扩展语言特征到批量维度
            text_feats = [lang_feat for _ in visual_feats]  # [M, 512]
            class_labels = [class_name for _ in visual_feats]  # [M,]

            # 1. 特征对齐（训练投影网络和权重预测网络）
            alignment_loss = self.align_features_with_triplet(visual_feats, text_feats, class_labels)
            print(f"📊 {class_name} 特征对齐损失：{alignment_loss:.4f}")

            # 2. 动态融合（使用对齐后的特征）
            with torch.no_grad():
                # 投影视觉特征
                visual_tensor = torch.tensor(np.array(visual_feats), dtype=torch.float32).to(self.device)
                projected_visual = self.visual_projection(visual_tensor)
                projected_visual = F.normalize(projected_visual, dim=-1)

                # 语言特征转换为张量
                lang_tensor = torch.tensor(lang_feat, dtype=torch.float32).to(self.device)

                # 批量动态融合
                fused_feats = []
                for vis_feat in projected_visual:
                    fused_feat = self.dynamic_feature_fusion(
                        vis_feat, lang_tensor,
                        class_name=class_name
                    )
                    fused_feats.append(fused_feat.cpu().numpy())

                # 融合特征均值（代表该类别的跨模态原子）
                current_fused_atom = np.mean(fused_feats, axis=0)
                current_fused_atom = current_fused_atom / np.linalg.norm(current_fused_atom)  # 归一化

            # 3. 时间演化（结合历史原子保持稳定性）
            if class_name in self.historical_atoms and len(self.historical_atoms[class_name]) > 0:
                # 历史原子均值
                historical_mean = np.mean(list(self.historical_atoms[class_name]), axis=0)
                historical_mean = historical_mean / np.linalg.norm(historical_mean)
                # 加权融合当前原子和历史原子
                current_fused_atom = (1 - self.temporal_consistency_weight) * current_fused_atom + \
                                     self.temporal_consistency_weight * historical_mean
                # 重新归一化
                current_fused_atom = current_fused_atom / np.linalg.norm(current_fused_atom)

            # 4. 更新词典
            self.combined_atoms[class_name] = current_fused_atom
            self.historical_atoms[class_name].append(current_fused_atom)
            print(f"✅ {class_name} 跨模态原子更新完成")

    def get_atom(self, class_name: str) -> Optional[np.ndarray]:
        """
        获取指定类别的跨模态原子特征
        优先级：融合原子 > 语言原子 > 视觉原子均值
        """
        # 1. 优先返回融合原子（最优）
        if class_name in self.combined_atoms:
            return self.combined_atoms[class_name].copy()
        # 2. 其次返回语言原子
        elif class_name in self.language_atoms:
            return self.language_atoms[class_name].copy()
        # 3. 最后返回视觉原子均值
        elif class_name in self.visual_atoms and len(self.visual_atoms[class_name]) > 0:
            visual_mean = np.mean(self.visual_atoms[class_name], axis=0)
            return visual_mean / np.linalg.norm(visual_mean)
        else:
            print(f"❌ 未找到 {class_name} 的原子特征")
            return None