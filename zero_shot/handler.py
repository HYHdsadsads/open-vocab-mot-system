"""零样本迁移处理模块，处理未见过的类别（增强知识融合与演化机制）"""
import os
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import ConvexHull
from collections import deque, defaultdict
from config import ZERO_SHOT_CONFIG, TRACKER_CONFIG, KG_CONFIG
from sklearn.cluster import KMeans
import clip
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any


class ZeroShotHandler:
    def __init__(self, dictionary, clip_model, clip_preprocess, device="cpu"):
        self.dictionary = dictionary
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device
        self.unknown_classes = set()
        self.synthetic_atoms = {}  # {类别: 特征向量}
        self.synthetic_atom_evolution = defaultdict(deque)  # 限制历史长度的双端队列
        self.convex_hull_samples = ZERO_SHOT_CONFIG.get("convex_hull_samples", 50)
        self.visual_atoms = defaultdict(lambda: deque(maxlen=100))  # 扩大视觉原子缓存
        self.kg_hierarchy = self._load_kg_hierarchy(KG_CONFIG.get("hierarchy_path"))  # 新增层级加载
        self.kg_relation_weights = KG_CONFIG.get("relation_weights", {"is_a": 0.8, "part_of": 0.5})  # 关系权重
        self.tracker = None
        self.recent_features = defaultdict(lambda: deque(maxlen=ZERO_SHOT_CONFIG.get("feature_window_size", 10)))
        self.confidence_history = defaultdict(lambda: deque(maxlen=5))  # 置信度历史用于稳定性判断

        # 知识图谱初始化
        self.kg_embeddings = self.load_kg_embeddings(KG_CONFIG["embeddings_path"])
        self.kg_classes = self._load_kg_classes(KG_CONFIG["classes_path"])
        self.kg_class_to_idx = {cls: i for i, cls in enumerate(self.kg_classes)}  # 加速查找

        # 语义特征融合层（增加残差连接）
        self.semantic_fusion = torch.nn.Sequential(
            torch.nn.Linear(512 + 512, 512),
            torch.nn.LayerNorm(512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 512)
        ).to(device)
        # 残差连接权重
        self.residual_weight = torch.nn.Parameter(torch.tensor(0.5, device=device))

        # 语义演化模型（增加门控机制）
        self.evolution_model = torch.nn.Sequential(
            torch.nn.Linear(512 + 512 + 128, 1024),
            torch.nn.LayerNorm(1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 512),
            torch.nn.LayerNorm(512)
        ).to(device)
        # 门控层控制演化强度
        self.evolution_gate = torch.nn.Sequential(
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
        ).to(device)
        self.evolution_optimizer = torch.optim.Adam(
            list(self.semantic_fusion.parameters()) +
            list(self.evolution_model.parameters()) +
            list(self.evolution_gate.parameters()) +
            [self.residual_weight],
            lr=ZERO_SHOT_CONFIG.get("evolution_lr", 5e-5)
        )

        # LSTM时序编码器（增加双向性）
        self.temporal_encoder = torch.nn.LSTM(
            input_size=512,
            hidden_size=256,  # 双向合并后保持512维
            num_layers=1,
            batch_first=True,
            bidirectional=True
        ).to(device)

    def _load_kg_hierarchy(self, path: str) -> Dict[str, List[Tuple[str, str]]]:
        """加载知识图谱层级关系（支持多关系类型）"""
        hierarchy = defaultdict(list)
        if not path or not os.path.exists(path):
            return hierarchy
        try:
            data = np.load(path, allow_pickle=True).item()
            for child, relations in data.items():
                for rel_type, parent in relations.items():
                    hierarchy[child].append((parent, rel_type))
            return hierarchy
        except Exception as e:
            print(f"加载知识图谱层级失败: {e}")
            return hierarchy

    def _load_kg_classes(self, path: str) -> List[str]:
        """安全加载知识图谱类别列表"""
        try:
            if os.path.exists(path):
                return np.load(path, allow_pickle=True).tolist()
            return []
        except Exception as e:
            print(f"加载知识图谱类别失败: {e}")
            return []

    def load_kg_embeddings(self, path: str) -> np.ndarray:
        """加载知识图谱嵌入（如ConceptNet）"""
        try:
            if os.path.exists(path):
                return np.load(path, allow_pickle=True)
            return np.zeros((0, 512))
        except Exception as e:
            print(f"加载知识图谱嵌入失败: {e}")
            return np.zeros((0, 512))

    def get_semantic_context(self, class_name: str) -> np.ndarray:
        """增强版语义上下文获取（多关系知识融合）"""
        # 1. 知识图谱特征（多关系融合）
        kg_feats = []
        weights = []

        # 自身特征
        if class_name in self.kg_class_to_idx:
            idx = self.kg_class_to_idx[class_name]
            if idx < len(self.kg_embeddings):
                kg_feats.append(self.kg_embeddings[idx])
                weights.append(1.0)

        # 关联特征（父类/部件等）
        for parent, rel_type in self.kg_hierarchy.get(class_name, []):
            if parent in self.kg_class_to_idx:
                idx = self.kg_class_to_idx[parent]
                if idx < len(self.kg_embeddings):
                    rel_weight = self.kg_relation_weights.get(rel_type, 0.3)
                    kg_feats.append(self.kg_embeddings[idx] * rel_weight)
                    weights.append(rel_weight)

        # 未知类别处理
        if not kg_feats:
            text_feat = self.get_text_feature(class_name)
            if len(self.kg_embeddings) > 0:
                similarities = cosine_similarity([text_feat], self.kg_embeddings)[0]
                top_k = min(3, len(similarities))
                top_indices = similarities.argsort()[-top_k:][::-1]
                for idx in top_indices:
                    kg_feats.append(self.kg_embeddings[idx])
                    weights.append(similarities[idx])
            else:
                kg_feats.append(np.zeros(512))
                weights.append(1.0)

        # 加权融合知识特征
        weights = np.array(weights) / sum(weights)
        kg_feat = np.sum(np.array(kg_feats) * weights[:, np.newaxis], axis=0)

        # 2. LLM语义特征（多提示词增强）
        llm_feats = [
            self.get_text_feature(f"a photo of a {class_name}"),
            self.get_text_feature(f"detailed visual description of {class_name}"),
            self.get_text_feature(f"{class_name} in industrial environment")
        ]
        llm_feat = np.mean(llm_feats, axis=0)

        # 3. 融合知识图谱和LLM特征（带残差连接）
        kg_tensor = torch.tensor(kg_feat, dtype=torch.float32).to(self.device).unsqueeze(0)
        llm_tensor = torch.tensor(llm_feat, dtype=torch.float32).to(self.device).unsqueeze(0)
        fused = self.semantic_fusion(torch.cat([kg_tensor, llm_tensor], dim=1))
        # 残差连接：融合结果 + 原始文本特征的加权组合
        residual = self.residual_weight * llm_tensor + (1 - self.residual_weight) * kg_tensor
        fused = fused + residual
        return F.normalize(fused, dim=1).cpu().detach().numpy().flatten()

    def generate_llm_semantic_feat(self, class_name: str) -> np.ndarray:
        """增强LLM语义特征生成（预留实际LLM接口）"""
        # 实际应用中可替换为GPT等模型的调用
        prompts = [
            f"physical characteristics of {class_name}",
            f"appearance features of {class_name} in industrial scenes",
            f"what does {class_name} look like"
        ]
        feats = [self.get_text_feature(prompt) for prompt in prompts]
        return np.mean(feats, axis=0)

    def _get_scene_context(self) -> np.ndarray:
        """增强场景上下文（动态权重）"""
        if self.tracker and hasattr(self.tracker, 'scene_context'):
            scene_feat = self.tracker.scene_context
            # 根据场景置信度调整权重
            if hasattr(self.tracker, 'scene_confidence'):
                conf = min(1.0, self.tracker.scene_confidence)
                scene_feat = scene_feat * conf + np.zeros_like(scene_feat) * (1 - conf)
            return np.pad(scene_feat, (0, 128 - len(scene_feat)), mode='constant')[:128]
        return np.zeros(128)

    def semantic_guided_evolution(self, unknown_class: str, current_feat: np.ndarray, recent_feats: List[np.ndarray]) -> np.ndarray:
        """增强语义引导演化（动态门控 + 多约束）"""
        # 1. 时序特征编码（双向LSTM + 注意力）
        temporal_feat = torch.tensor(current_feat, dtype=torch.float32).to(self.device)
        if len(recent_feats) >= 2:
            feat_array = np.array(recent_feats, dtype=np.float32)
            feat_seq = torch.from_numpy(feat_array).unsqueeze(0).to(self.device)
            lstm_out, _ = self.temporal_encoder(feat_seq)
            # 双向特征合并
            lstm_out = lstm_out[:, :, :256] + lstm_out[:, :, 256:]  # 256+256=512

            # 时序注意力（考虑特征稳定性）
            current_feat_tensor = torch.tensor(current_feat, dtype=torch.float32).to(self.device).unsqueeze(0)
            similarities = F.cosine_similarity(lstm_out[0], current_feat_tensor, dim=1)
            # 稳定性权重：与历史差异小的特征权重更高
            stability = 1 - torch.std(lstm_out[0], dim=0).mean()
            attn_weights = F.softmax(similarities * (1 + stability), dim=0)
            temporal_feat = torch.sum(attn_weights.unsqueeze(1) * lstm_out[0], dim=0)

        # 2. 多关系知识图谱约束
        hierarchy_constraint = torch.tensor(0.0, device=self.device)
        current_class = unknown_class
        depth = 0
        for parent, rel_type in self.kg_hierarchy.get(current_class, [])[:2]:  # 限制深度
            if parent in self.dictionary.classes:
                parent_feat = self.dictionary.get_atom(parent)
                if parent_feat is not None:
                    parent_feat_tensor = torch.tensor(parent_feat, dtype=torch.float32).to(self.device)
                    rel_weight = self.kg_relation_weights.get(rel_type, 0.3)
                    hierarchy_constraint += F.cosine_similarity(temporal_feat, parent_feat_tensor) * rel_weight * (0.9 ** depth)
            depth += 1

        # 3. 语义上下文融合
        scene_context = self._get_scene_context()
        sem_context = self.get_semantic_context(unknown_class)

        # 特征拼接
        sem_context_tensor = torch.tensor(sem_context, dtype=torch.float32).to(self.device)
        scene_context_tensor = torch.tensor(scene_context, dtype=torch.float32).to(self.device)
        input_feat = torch.cat([temporal_feat, sem_context_tensor, scene_context_tensor], dim=0)

        # 4. 特征演化（带门控机制）
        evolved_feat = self.evolution_model(input_feat)
        # 门控控制：根据特征稳定性动态调整演化强度
        gate = self.evolution_gate(temporal_feat.unsqueeze(0))[0, 0]
        evolved_feat = gate * evolved_feat + (1 - gate) * temporal_feat
        evolved_feat = F.normalize(evolved_feat, dim=0)

        # 5. 多目标损失优化
        self.evolution_optimizer.zero_grad()

        # 时序一致性损失
        temporal_loss = F.mse_loss(evolved_feat, temporal_feat)

        # 语义相似性损失
        contrast_loss = torch.tensor(0.0, device=self.device)
        similar_classes = self.find_similar_classes(unknown_class, top_k=2)
        for cls, sim in similar_classes:
            cls_feat = self.dictionary.get_atom(cls)
            if cls_feat is not None:
                cls_tensor = torch.tensor(cls_feat, dtype=torch.float32).to(self.device)
                contrast_loss += (1 - sim) * F.mse_loss(evolved_feat, cls_tensor)
        contrast_loss = contrast_loss / len(similar_classes) if similar_classes else contrast_loss

        # 知识约束损失
        loss = temporal_loss + 0.5 * contrast_loss - 0.2 * hierarchy_constraint
        loss.backward()
        self.evolution_optimizer.step()

        return evolved_feat.cpu().detach().numpy()

    def update_synthetic_atoms(self, frame_num: int, recent_features_map: Dict[str, List[np.ndarray]]):
        """更新合成原子（增加帧间隔控制）"""
        # 每N帧更新一次，减少计算量
        if frame_num % ZERO_SHOT_CONFIG.get("update_interval", 3) != 0:
            return

        for cls in list(recent_features_map.keys()):
            recent_features = recent_features_map[cls]
            if len(recent_features) < ZERO_SHOT_CONFIG.get("min_update_samples", 3):
                continue

            current_atom = self.synthetic_atoms.get(cls, recent_features[-1])
            updated_atom = self.semantic_guided_evolution(cls, current_atom, recent_features)

            self.synthetic_atom_evolution[cls].append(updated_atom)
            # 动态调整历史窗口（根据置信度）
            conf_mean = np.mean(self.confidence_history[cls]) if self.confidence_history[cls] else 0.5
            history_window = max(3, min(10, int(5 / (conf_mean + 1e-5))))
            while len(self.synthetic_atom_evolution[cls]) > history_window:
                self.synthetic_atom_evolution[cls].popleft()

            # 带权重的历史融合（新特征权重更高）
            weights = np.linspace(0.5, 1.0, len(self.synthetic_atom_evolution[cls]))
            weights /= weights.sum()
            weighted_mean = np.sum(np.array(self.synthetic_atom_evolution[cls]) * weights[:, np.newaxis], axis=0)
            self.synthetic_atoms[cls] = F.normalize(torch.tensor(weighted_mean), dim=0).numpy()

    def set_tracker(self, tracker):
        """设置跟踪器实例"""
        self.tracker = tracker

    def get_text_feature(self, text: str) -> np.ndarray:
        """获取CLIP文本特征（增强鲁棒性）"""
        try:
            text_tokens = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                text_feature = self.clip_model.encode_text(text_tokens)
                text_feature = F.normalize(text_feature, dim=-1)
            return text_feature.cpu().numpy().flatten()
        except Exception as e:
            print(f"获取文本特征失败: {e}")
            return np.zeros(512)

    def find_similar_classes(self, unknown_class: str, top_k: int = 2) -> List[Tuple[str, float]]:
        """寻找相似类别（结合知识图谱关系）"""
        if not self.dictionary.classes:
            return []

        unknown_feat = self.get_text_feature(unknown_class)
        similarities = []

        # 1. 基于特征的相似性
        for cls in self.dictionary.classes:
            cls_feat = self.dictionary.get_atom(cls)
            if cls_feat is None or not isinstance(cls_feat, np.ndarray) or cls_feat.size != 512:
                continue
            feat_sim = cosine_similarity([unknown_feat], [cls_feat])[0][0]
            similarities.append((cls, feat_sim))

        # 2. 基于知识图谱的相似性增强
        kg_boost = defaultdict(float)
        # 直接关联
        for parent, rel_type in self.kg_hierarchy.get(unknown_class, []):
            if parent in self.dictionary.classes:
                kg_boost[parent] += self.kg_relation_weights.get(rel_type, 0.3)
        # 间接关联
        for cls in self.dictionary.classes:
            for child, rels in self.kg_hierarchy.items():
                if child == unknown_class:
                    continue
                for parent, rel_type in rels:
                    if parent == unknown_class and cls == child:
                        kg_boost[cls] += self.kg_relation_weights.get(rel_type, 0.2)

        # 融合特征相似性和知识相似性
        final_sims = []
        for cls, feat_sim in similarities:
            kg_sim = kg_boost.get(cls, 0.0)
            # 加权融合
            final_sim = 0.7 * feat_sim + 0.3 * kg_sim
            final_sims.append((cls, final_sim))

        final_sims.sort(key=lambda x: x[1], reverse=True)
        return final_sims[:top_k]

    def classify_unknown(self, feature: np.ndarray, unknown_class: str) -> Tuple[str, float]:
        """未知类别分类（增加置信度过滤）"""
        self.unknown_classes.add(unknown_class)

        # 初始化合成原子
        if unknown_class not in self.synthetic_atoms:
            self.generate_synthetic_atom(unknown_class)

        # 缓存特征
        self.recent_features[unknown_class].append(feature)

        # 寻找相似类别
        similar_classes = self.find_similar_classes(unknown_class, top_k=3)
        if not similar_classes:
            self.confidence_history[unknown_class].append(0.3)
            return unknown_class, 0.3

        # 计算特征相似度
        similarities = []
        for cls, _ in similar_classes:
            cls_atom = self.dictionary.get_atom(cls)
            if cls_atom is not None:
                sim = cosine_similarity([feature], [cls_atom])[0][0]
                similarities.append((cls, sim))

        if not similarities:
            self.confidence_history[unknown_class].append(0.3)
            return unknown_class, 0.3

        # 确定最佳匹配
        similarities.sort(key=lambda x: x[1], reverse=True)
        best_class, best_sim = similarities[0]

        # 应用语义演化
        if len(self.recent_features[unknown_class]) >= 2:
            evolved_atom = self.synthetic_atoms[unknown_class]
            evolved_sim = cosine_similarity([feature], [evolved_atom])[0][0]
            best_sim = 0.6 * best_sim + 0.4 * evolved_sim  # 调整权重

        # 置信度平滑
        self.confidence_history[unknown_class].append(best_sim)
        smoothed_sim = np.mean(self.confidence_history[unknown_class])
        return best_class, min(smoothed_sim, 1.0)

    def generate_synthetic_atom(self, unknown_class: str) -> np.ndarray:
        """生成合成原子（增强特征多样性）"""
        if unknown_class in self.synthetic_atoms:
            return self.synthetic_atoms[unknown_class]

        similar_classes = self.find_similar_classes(unknown_class, top_k=3)
        if not similar_classes:
            return self._basic_synthetic_atom(unknown_class)

        # 收集多源特征点
        feature_points = []
        for cls, sim in similar_classes:
            # 基础原子
            current_atom = self.dictionary.get_atom(cls)
            if current_atom is not None:
                feature_points.append(current_atom * sim)  # 带相似度权重

            # 历史原子
            if hasattr(self.dictionary, 'historical_atoms') and cls in self.dictionary.historical_atoms:
                hist_atoms = self.dictionary.historical_atoms[cls]
                if isinstance(hist_atoms, list) and hist_atoms:
                    # 取最近的历史特征
                    feature_points.extend(hist_atoms[-3:] * sim)

            # 视觉原子
            if hasattr(self.dictionary, 'visual_atoms') and cls in self.dictionary.visual_atoms:
                visual_atoms = self.dictionary.visual_atoms[cls]
                if isinstance(visual_atoms, (list, deque)) and visual_atoms:
                    # 随机采样避免偏差
                    sample_size = min(5, len(visual_atoms))
                    samples = np.random.choice(visual_atoms, sample_size, replace=False)
                    feature_points.extend(samples * sim)

        # 过滤无效特征
        valid_features = [f for f in feature_points if isinstance(f, np.ndarray) and f.shape == (512,)]
        if len(valid_features) < 3:
            return self._basic_synthetic_atom(unknown_class)

        try:
            # 分层聚类增强多样性
            feature_array = np.array(valid_features)
            n_clusters = min(5, max(2, len(valid_features) // 5))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(feature_array)

            # 从每个聚类中采样
            virtual_atoms = []
            for cluster_id in range(n_clusters):
                cluster_points = feature_array[kmeans.labels_ == cluster_id]
                if len(cluster_points) >= 2:
                    # 聚类内凸包采样
                    try:
                        hull = ConvexHull(cluster_points)
                        for _ in range(self.convex_hull_samples // n_clusters):
                            a, b = np.random.choice(hull.vertices, 2, replace=False)
                            alpha = np.random.beta(2, 2)  # 偏向两端的采样分布
                            interpolated = alpha * cluster_points[a] + (1 - alpha) * cluster_points[b]
                            virtual_atoms.append(interpolated / np.linalg.norm(interpolated))
                    except:
                        # 凸包失败时随机采样
                        for _ in range(self.convex_hull_samples // n_clusters):
                            a, b = np.random.choice(len(cluster_points), 2, replace=False)
                            interpolated = 0.5 * cluster_points[a] + 0.5 * cluster_points[b]
                            virtual_atoms.append(interpolated / np.linalg.norm(interpolated))

            # 融合文本特征
            unknown_text_feat = self.get_text_feature(unknown_class)
            virtual_mean = np.mean(virtual_atoms, axis=0) if virtual_atoms else unknown_text_feat
            synthetic = 0.6 * virtual_mean + 0.4 * unknown_text_feat  # 增加文本特征权重
            synthetic = synthetic / np.linalg.norm(synthetic)

            self.synthetic_atoms[unknown_class] = synthetic
            self.synthetic_atom_evolution[unknown_class].append(synthetic)
            self.unknown_classes.add(unknown_class)
            return synthetic

        except Exception as e:
            print(f"合成原子生成失败: {e}")
            return self._basic_synthetic_atom(unknown_class)

    def _basic_synthetic_atom(self, unknown_class: str) -> np.ndarray:
        """基础合成原子（增强鲁棒性）"""
        text_feats = [
            self.get_text_feature(unknown_class),
            self.get_text_feature(f"object {unknown_class}"),
            self.get_text_feature(f"industrial {unknown_class}")
        ]
        text_feat = np.mean(text_feats, axis=0)
        self.synthetic_atoms[unknown_class] = text_feat
        self.synthetic_atom_evolution[unknown_class].append(text_feat)
        self.unknown_classes.add(unknown_class)
        return text_feat

    def detect_new_class(self, feature: np.ndarray, threshold: Optional[float] = None) -> Tuple[Optional[str], float]:
        """检测新类别（增加自适应阈值）"""
        threshold = threshold or ZERO_SHOT_CONFIG.get("detection_threshold", 0.5)
        # 动态调整阈值（根据已知类别分布）
        if self.dictionary.classes:
            known_feats = [self.dictionary.get_atom(cls) for cls in self.dictionary.classes if self.dictionary.get_atom(cls) is not None]
            if known_feats:
                intra_sim = np.mean([cosine_similarity([f1], [f2])[0][0] for i, f1 in enumerate(known_feats) for f2 in known_feats[i+1:]])
                threshold = max(threshold, min(0.8, intra_sim + 0.1))  # 基于类内相似度调整

        similarities = []
        # 已知类别匹配
        for cls in self.dictionary.classes:
            cls_feat = self.dictionary.get_atom(cls)
            if cls_feat is not None:
                sim = cosine_similarity([feature], [cls_feat])[0][0]
                similarities.append((cls, sim))

        # 合成类别匹配
        for cls in self.synthetic_atoms:
            cls_feat = self.synthetic_atoms[cls]
            sim = cosine_similarity([feature], [cls_feat])[0][0]
            similarities.append((cls, sim))

        if similarities:
            best_cls, best_sim = max(similarities, key=lambda x: x[1])
            if best_sim >= threshold:
                return best_cls, best_sim

        return None, 0.0

    def update_visual_atoms(self, class_name: str, features: List[np.ndarray], max_atoms: int = 50):
        """更新视觉原子（增强多样性保持）"""
        if not features:
            return

        # 过滤低质量特征
        valid_feats = []
        for feat in features:
            if isinstance(feat, np.ndarray) and feat.shape == (512,) and not np.allclose(feat, 0):
                valid_feats.append(feat)
        if not valid_feats:
            return

        # 添加新特征
        for feat in valid_feats:
            # 去重（相似特征过滤）
            if not any(cosine_similarity([feat], [f])[0][0] > 0.95 for f in self.visual_atoms[class_name]):
                self.visual_atoms[class_name].append(feat)

        # 聚类优化（保持多样性）
        if len(self.visual_atoms[class_name]) >= max_atoms:
            feats = np.array(list(self.visual_atoms[class_name]))
            try:
                # 密度聚类识别核心点
                from sklearn.cluster import DBSCAN
                dbscan = DBSCAN(eps=0.3, min_samples=3).fit(feats)
                core_mask = np.zeros(len(feats), dtype=bool)
                core_mask[dbscan.core_sample_indices_] = True
                core_feats = feats[core_mask]

                # 对核心点聚类
                n_clusters = min(10, max(2, len(core_feats) // 5))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(core_feats)

                # 保留聚类中心和边界点
                boundary_feats = []
                if len(core_feats) > 512:  # 维度足够时计算凸包
                    try:
                        hull = ConvexHull(core_feats)
                        boundary_feats = core_feats[hull.vertices]
                    except:
                        pass

                # 组合并去重
                combined = np.vstack([kmeans.cluster_centers_, boundary_feats])
                unique_combined = np.unique(combined, axis=0)[:max_atoms]

                self.visual_atoms[class_name] = deque(unique_combined.tolist(), maxlen=max_atoms)
            except Exception as e:
                print(f"视觉原子聚类失败: {e}")
                # 降级策略：保留最近的特征
                self.visual_atoms[class_name] = deque(
                    list(self.visual_atoms[class_name])[-max_atoms:],
                    maxlen=max_atoms
                )

    def process_unknown(self, bbox: List[float], feature: np.ndarray, class_name: str) -> Tuple[List[float], np.ndarray, str]:
        """处理未知类别主函数（增加自适应更新）"""
        self.unknown_classes.add(class_name)

        # 初始化或更新合成原子
        if class_name not in self.synthetic_atoms:
            self.generate_synthetic_atom(class_name)
        else:
            self.recent_features[class_name].append(feature)
            # 根据特征变化率动态调整更新时机
            if len(self.recent_features[class_name]) >= self.recent_features[class_name].maxlen:
                feats = np.array(list(self.recent_features[class_name]))
                feat_var = np.var(feats, axis=0).mean()  # 特征变化率
                update_threshold = ZERO_SHOT_CONFIG.get("update_threshold", 0.05)
                if feat_var > update_threshold or (self.tracker and self.tracker.frame_counter % 5 == 0):
                    self.update_synthetic_atoms(
                        frame_num=self.tracker.frame_counter if self.tracker else 0,
                        recent_features_map={class_name: list(self.recent_features[class_name])}
                    )

        # 更新视觉原子（批量处理提高效率）
        if len(self.recent_features[class_name]) % 3 == 0:  # 每3个特征更新一次
            self.update_visual_atoms(class_name, [feature])

        return (bbox, feature, class_name)