"""
YOLO-RD: 检索增强检测算法
解决"看不见"的问题 - 通过领域字典构建和检索增强机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import clip
from config import MODEL_CONFIG


class DomainDictionary:
    """领域字典构建模块 - 优化版"""
    
    def __init__(self, clip_model, device="cpu", enable_cache=True, cache_size=1000):
        self.clip_model = clip_model
        self.device = device
        self.dictionary = {}  # {domain: {class_name: embedding}}
        self.class_prototypes = {}  # {class_name: List[embeddings]}
        
        # 缓存机制
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self._embedding_cache = {}  # {text: embedding}
        self._cache_hits = 0
        self._cache_misses = 0
        
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """获取文本嵌入（带缓存）"""
        if self.enable_cache and text in self._embedding_cache:
            self._cache_hits += 1
            return self._embedding_cache[text].copy()
        
        self._cache_misses += 1
        
        text_tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_feat = self.clip_model.encode_text(text_tokens)
            text_feat = F.normalize(text_feat, dim=-1)
            embedding = text_feat.cpu().numpy().flatten()
        
        # 缓存管理（LRU）
        if self.enable_cache:
            if len(self._embedding_cache) >= self.cache_size:
                # 删除最旧的条目
                self._embedding_cache.pop(next(iter(self._embedding_cache)))
            self._embedding_cache[text] = embedding
        
        return embedding
    
    def build_from_classes(self, domain: str, class_names: List[str]):
        """从类别列表构建领域字典 - 优化版"""
        if domain not in self.dictionary:
            self.dictionary[domain] = {}
            
        print(f"🔨 构建领域字典: {domain}")
        
        for class_name in class_names:
            # 生成多个提示词以增强鲁棒性
            prompts = [
                f"a photo of a {class_name}",
                f"{class_name} in the scene",
                f"high quality image of {class_name}",
                f"{class_name} object"
            ]
            
            embeddings = []
            for prompt in prompts:
                embedding = self._get_text_embedding(prompt)
                embeddings.append(embedding)
            
            # 存储平均嵌入
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
            self.dictionary[domain][class_name] = avg_embedding
            
            # 存储所有嵌入作为原型
            if class_name not in self.class_prototypes:
                self.class_prototypes[class_name] = []
            self.class_prototypes[class_name].extend(embeddings)
        
        cache_hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses + 1e-8)
        print(f"✅ 领域字典构建完成: {domain}, 类别数: {len(class_names)}")
        print(f"   缓存命中率: {cache_hit_rate:.2%}")
        
    def retrieve_similar(self, query_embedding: np.ndarray, domain: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """检索相似类别"""
        if domain not in self.dictionary:
            return []
            
        similarities = []
        for class_name, class_emb in self.dictionary[domain].items():
            sim = cosine_similarity([query_embedding], [class_emb])[0][0]
            similarities.append((class_name, sim))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_embedding(self, domain: str, class_name: str) -> Optional[np.ndarray]:
        """获取类别嵌入"""
        if domain in self.dictionary and class_name in self.dictionary[domain]:
            return self.dictionary[domain][class_name]
        return None


class RetrievalAugmentedDetector(nn.Module):
    """检索增强检测器 (YOLO-RD)"""
    
    def __init__(self, yolo_model, clip_model, clip_preprocess, device="cpu"):
        super().__init__()
        self.yolo_model = yolo_model
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device
        
        # 领域字典
        self.domain_dict = DomainDictionary(clip_model, device)
        
        # 检索增强模块
        self.retrieval_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512)
        ).to(device)
        
        # 融合权重网络
        self.fusion_gate = nn.Sequential(
            nn.Linear(512 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(device)
        
    def forward(self, image: np.ndarray, domain: str = "general") -> List[Dict]:
        """
        检索增强检测前向传播
        
        Args:
            image: 输入图像
            domain: 领域名称
            
        Returns:
            增强后的检测结果
        """
        # 1. YOLO 基础检测
        yolo_results = self.yolo_model(image, verbose=False)
        detections = []
        
        for result in yolo_results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = [x1, y1, x2, y2]
                conf = float(box.conf)
                class_id = int(box.cls)
                class_name = self.yolo_model.names[class_id]
                
                # 提取 ROI
                h, w = image.shape[:2]
                x1_c = max(0, int(x1))
                y1_c = max(0, int(y1))
                x2_c = min(w, int(x2))
                y2_c = min(h, int(y2))
                roi = image[y1_c:y2_c, x1_c:x2_c]
                
                if roi.size == 0:
                    continue
                
                # 2. 提取 CLIP 视觉特征
                try:
                    from PIL import Image
                    import cv2
                    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    pil_roi = Image.fromarray(rgb_roi)
                    img_input = self.clip_preprocess(pil_roi).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        visual_feat = self.clip_model.encode_image(img_input)
                        visual_feat = F.normalize(visual_feat, dim=-1)
                        visual_feat_np = visual_feat.cpu().numpy().flatten()
                    
                    # 3. 检索增强
                    retrieved = self.domain_dict.retrieve_similar(visual_feat_np, domain, top_k=3)
                    
                    if retrieved:
                        # 获取检索到的类别嵌入
                        retrieved_class, retrieved_sim = retrieved[0]
                        retrieved_emb = self.domain_dict.get_embedding(domain, retrieved_class)
                        
                        if retrieved_emb is not None:
                            # 4. 特征融合
                            retrieved_tensor = torch.tensor(retrieved_emb, dtype=torch.float32).to(self.device)
                            
                            # 检索增强特征
                            enhanced_feat = self.retrieval_head(visual_feat)
                            
                            # 动态融合权重
                            concat_feat = torch.cat([visual_feat, enhanced_feat], dim=-1)
                            gate = self.fusion_gate(concat_feat)
                            
                            # 融合特征
                            fused_feat = gate * enhanced_feat + (1 - gate) * visual_feat
                            fused_feat = F.normalize(fused_feat, dim=-1)
                            
                            # 5. 重新分类（基于检索结果）
                            if retrieved_sim > 0.7:  # 高置信度检索
                                class_name = retrieved_class
                                conf = conf * retrieved_sim  # 调整置信度
                    
                    detections.append({
                        "bbox": bbox,
                        "class_name": class_name,
                        "confidence": conf,
                        "visual_feature": visual_feat_np,
                        "retrieved_classes": retrieved
                    })
                    
                except Exception as e:
                    print(f"⚠️ 检索增强失败: {e}")
                    continue
        
        return detections
    
    def build_domain_dictionary(self, domain: str, class_names: List[str]):
        """构建领域字典"""
        self.domain_dict.build_from_classes(domain, class_names)
    
    def update_dictionary_from_data(self, domain: str, images: List[np.ndarray], 
                                   labels: List[str]):
        """从数据更新字典"""
        for image, label in zip(images, labels):
            try:
                from PIL import Image
                import cv2
                rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
                img_input = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    visual_feat = self.clip_model.encode_image(img_input)
                    visual_feat = F.normalize(visual_feat, dim=-1)
                    visual_feat_np = visual_feat.cpu().numpy().flatten()
                
                # 更新原型
                if label not in self.domain_dict.class_prototypes:
                    self.domain_dict.class_prototypes[label] = []
                self.domain_dict.class_prototypes[label].append(visual_feat_np)
                
                # 更新字典嵌入（使用移动平均）
                if domain not in self.domain_dict.dictionary:
                    self.domain_dict.dictionary[domain] = {}
                
                if label in self.domain_dict.dictionary[domain]:
                    old_emb = self.domain_dict.dictionary[domain][label]
                    new_emb = 0.9 * old_emb + 0.1 * visual_feat_np
                    self.domain_dict.dictionary[domain][label] = new_emb
                else:
                    self.domain_dict.dictionary[domain][label] = visual_feat_np
                    
            except Exception as e:
                print(f"⚠️ 字典更新失败: {e}")
                continue
        
        print(f"✅ 字典更新完成: {domain}, 样本数: {len(images)}")
