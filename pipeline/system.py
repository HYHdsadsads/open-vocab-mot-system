import time
import cv2
import numpy as np
from tqdm import tqdm
from config import MODEL_CONFIG, DATASET_CONFIG, VISUALIZATION_CONFIG, ASSOCIATION_CONFIG
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from models.dictionary import CrossModalDictionary
from tracking.multi_tracker import MultiObjectTracker
from zero_shot.handler import ZeroShotHandler
from ultralytics import YOLO
import clip
from data.dataset import UniversalDatasetLoader
from PIL import Image
import torch
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Any, Optional
# 新增模块导入（根据实际文件位置调整）
from models.reliability_head import ReliabilityEstimator  # 假设可靠性评估器
from models.proto_memory import ProtoMemory  # 假设原型记忆模块
from models.kg_graph import HyperbolicKG  # 之前定义的知识图谱模块
from tracking.association import SemanticAssociation  # 之前定义的关联器


class OpenVocabMOTSystem:
    """开放词汇多目标跟踪系统完整流程"""

    def __init__(self, class_names, experiment_mode=False, batch_size=2):
        # 设备配置（优先使用GPU）
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_model = YOLO(MODEL_CONFIG["yolo_model_path"])
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        # 批处理配置
        self.batch_size = batch_size
        self.frame_buffer = deque(maxlen=batch_size)

        # 初始化已知类别
        self.known_classes = class_names
        self.unknown_classes = set()  # 记录遇到的未知类别

        # 初始化组件
        self.dictionary = CrossModalDictionary(
            self.clip_model, self.clip_preprocess, self.yolo_model, device=self.device
        )
        self.dictionary.initialize_from_classes(class_names)
        self.tracker = MultiObjectTracker(self.dictionary)
        self.zero_shot_handler = ZeroShotHandler(
            self.dictionary, self.clip_model, self.clip_preprocess, device=self.device
        )
        self.zero_shot_handler.set_tracker(self.tracker)  # 使用setter方法关联跟踪器

        # 初始化实验模式
        self.experiment_mode = experiment_mode
        self.ablation_logs = {
            "feature_weights": [],
            "track_matches": [],
            "zero_shot_accuracy": []
        }
        # 性能计时器
        self.perf_metrics = defaultdict(list)
        # 存储结果
        self.results = []
        # 可视化配置
        self.vis_config = VISUALIZATION_CONFIG

        # 新增模块初始化
        self.reliability_estimator = ReliabilityEstimator(device=self.device, clip_model=self.clip_model)
        self.proto_memory = ProtoMemory(device=self.device)
        self.kg_graph = HyperbolicKG(
            kg_path=MODEL_CONFIG["kg_path"],
            device=self.device
        )
        self.associator = SemanticAssociation(
            self.dictionary, self.kg_graph, self.proto_memory
        )

    def detect_objects(self, image, conf_threshold=0.5):
        """检测图像中的目标，强化特征有效性检查"""
        results = self.yolo_model(image, verbose=False, conf=conf_threshold)
        detections = []

        for result in results:
            for box in result.boxes:
                # 获取边界框并过滤过小目标
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = [x1, y1, x2, y2]
                bbox_area = (x2 - x1) * (y2 - y1)
                if bbox_area < self.vis_config["min_bbox_area"]:  # 使用面积过滤更合理
                    continue

                # 提取目标图像区域
                h, w = image.shape[:2]
                # 确保边界框在图像范围内（防止越界）
                x1_clamped = max(0, int(x1))
                y1_clamped = max(0, int(y1))
                x2_clamped = min(w, int(x2))
                y2_clamped = min(h, int(y2))
                obj_img = image[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

                # 再次检查目标区域有效性
                if obj_img.size == 0:
                    continue  # 跳过空区域

                # 提取CLIP特征
                try:
                    rgb_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_img)
                    img_input = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        feature = self.clip_model.encode_image(img_input)
                        feature = feature.cpu().numpy().flatten()  # 转为1D数组

                    # 验证特征维度（CLIP ViT-B/32输出应为512维）
                    if feature.shape[0] != 512:
                        if self.experiment_mode:
                            print(f"警告：特征维度异常 {feature.shape[0]}，跳过该目标")
                        continue

                    # 获取类别名称并添加到检测结果
                    class_name = self.yolo_model.names[int(box.cls)]
                    detections.append((bbox, feature, class_name, float(box.conf)))

                except Exception as e:
                    if self.experiment_mode:
                        print(f"特征提取失败: {e}")
                    continue  # 跳过提取失败的特征

        return detections

    def batch_detect_objects(self, frames):
        """批处理检测，提高GPU利用率"""
        if not frames:
            return []

        # 批量目标检测
        results = self.yolo_model(frames, verbose=False, conf=0.5)
        batch_detections = []

        for frame_idx, result in enumerate(results):
            frame = frames[frame_idx]
            h, w = frame.shape[:2]
            detections = []

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = [x1, y1, x2, y2]
                bbox_area = (x2 - x1) * (y2 - y1)
                if bbox_area < self.vis_config["min_bbox_area"]:
                    continue

                # 提取目标区域
                x1_clamped = max(0, int(x1))
                y1_clamped = max(0, int(y1))
                x2_clamped = min(w, int(x2))
                y2_clamped = min(h, int(y2))
                obj_img = frame[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

                if obj_img.size == 0:
                    continue

                # 收集用于批量特征提取的图像
                try:
                    rgb_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_img)
                    img_input = self.clip_preprocess(pil_img)
                    class_name = self.yolo_model.names[int(box.cls)]
                    detections.append((bbox, img_input, class_name, float(box.conf)))
                except Exception as e:
                    if self.experiment_mode:
                        print(f"批处理准备失败: {e}")
                    continue

            # 批量提取CLIP特征
            if detections:
                imgs_tensor = torch.stack([d[1] for d in detections]).to(self.device)
                with torch.no_grad():
                    features = self.clip_model.encode_image(imgs_tensor)
                    features = features.cpu().numpy()

                # 组装结果
                for i in range(len(detections)):
                    bbox, _, class_name, conf = detections[i]
                    feature = features[i].flatten()
                    if feature.shape[0] == 512:
                        batch_detections.append((frame_idx, bbox, feature, class_name, conf))

        # 按帧分组
        frame_results = [[] for _ in range(len(frames))]
        for det in batch_detections:
            frame_idx, bbox, feature, class_name, conf = det
            frame_results[frame_idx].append((bbox, feature, class_name, conf))

        return frame_results

    def process_frame(self, image, frame_num):
        """完善处理流程，加入可靠性评估与原型更新"""
        # 1. 目标检测
        start = time.time()
        detections = self.detect_objects(image)
        self.perf_metrics["detection_time"].append(time.time() - start)

        # 2. 可靠性评估
        visual_reliability = []
        lang_reliability = []
        class_freq = self.dictionary.get_class_frequency()
        for det in detections:
            bbox, feat, cls_name, _ = det
            # 视觉可靠性
            vis_rel = self.reliability_estimator.visual_reliability(image, bbox)
            # 语言可靠性
            lang_rel = self.reliability_estimator.language_reliability(cls_name, class_freq)
            visual_reliability.append(vis_rel)
            lang_reliability.append(lang_rel)

        # 3. 动态特征融合（基于可靠性）
        start = time.time()
        processed_detections = []
        for i, (bbox, feat, cls_name, conf) in enumerate(detections):
            # 计算动态权重
            vis_weight = visual_reliability[i] / (visual_reliability[i] + lang_reliability[i] + 1e-8)
            lang_weight = 1 - vis_weight
            # 融合视觉与语言特征
            lang_feat = self.dictionary.language_atoms.get(cls_name, np.zeros(512))
            fused_feat = vis_weight * feat + lang_weight * lang_feat
            processed_detections.append((bbox, fused_feat, cls_name, conf))
        self.perf_metrics["fusion_time"].append(time.time() - start)

        # 4. 原型记忆更新
        for det in processed_detections:
            _, feat, cls_name, _ = det
            self.proto_memory.update_text_proto(cls_name, self.dictionary.language_atoms[cls_name])

        # 5. 新类发现（对未知类别进行聚类）
        unlabeled_feats = [f for b, f, c, _ in processed_detections if c not in self.known_classes]
        if unlabeled_feats:
            clusters = self.proto_memory.discover_new_classes(unlabeled_feats)
            if clusters and self.experiment_mode:
                print(f"帧 {frame_num} 发现 {len(clusters)} 个新类簇")

        # 6. 目标关联与跟踪更新
        start = time.time()
        matches = self.associator.associate(self.tracker.tracks, processed_detections, visual_reliability,
                                            lang_reliability)
        
        # 使用匹配结果更新跟踪器
        matched_dets = set()
        for track_idx, det_idx in matches:
            track = self.tracker.tracks[track_idx]
            det_bbox, det_feat, det_class, det_conf = processed_detections[det_idx]
            self.tracker._update_track(track, det_bbox, det_feat, det_class, frame_num)
            matched_dets.add(det_idx)
            # 更新轨迹原型
            self.proto_memory.update_visual_proto(track.id, det_feat, frame_num)
        
        # 处理未匹配的检测，创建新跟踪
        for det_idx, det in enumerate(processed_detections):
            if det_idx not in matched_dets:
                det_bbox, det_feat, det_class, det_conf = det
                self.tracker._add_new_track(det_bbox, det_feat, det_class, frame_num)
        
        # 过滤过期跟踪
        self.tracker.tracks = [t for t in self.tracker.tracks if t.misses <= self.tracker.max_age]
        
        # 返回有效跟踪
        tracks = [t for t in self.tracker.tracks if t.hits >= self.tracker.min_hits]
        self.perf_metrics["tracking_time"].append(time.time() - start)

        return tracks

    def process_batch_frames(self, frames, start_frame_num):
        """批处理多帧，提高处理效率"""
        start = time.time()
        batch_detections = self.batch_detect_objects(frames)
        self.perf_metrics["batch_detection_time"].append(time.time() - start)

        batch_tracks = []
        start = time.time()

        for i, frame in enumerate(frames):
            frame_num = start_frame_num + i
            # 处理零样本
            processed, _ = self.process_unknown_classes(batch_detections[i])
            # 更新跟踪
            tracks = self.tracker.update(processed, frame_num)
            batch_tracks.append(tracks)

            # 实验日志
            if self.experiment_mode and frame_num % 10 == 0:
                self.ablation_logs["feature_weights"].append(
                    {c: self.dictionary.feature_weights.get(c, [0.5, 0.5]) for c in self.dictionary.classes}
                )

        self.perf_metrics["batch_processing_time"].append(time.time() - start)
        return batch_tracks

    def evaluate(self, ground_truth, metric_type=["mota", "idf1", "precision", "recall"]):
        """实现标准评估指标计算，强化无效ID过滤"""
        from motmetrics import metrics
        import motmetrics as mm

        # 初始化评估器
        acc = mm.MOTAccumulator(auto_id=True)

        # 确保ground_truth是按帧排序的列表
        sorted_frames = sorted(ground_truth.keys())
        ground_truth_list = [ground_truth[frame_id] for frame_id in sorted_frames]

        # 匹配结果与真值（确保两者帧数一致）
        min_length = min(len(ground_truth_list), len(self.results))
        for frame_idx in range(min_length):
            gt_frame = ground_truth_list[frame_idx]
            pred_frame = self.results[frame_idx]

            # 检查gt_frame是否为可迭代对象，若不是则视为空帧
            if not isinstance(gt_frame, (list, tuple)):
                gt_frame = []

            # 提取GT和预测的ID与边界框
            # 过滤GT中可能存在的无效ID
            gt_ids = []
            gt_bboxes = []
            for obj in gt_frame:
                track_id = obj.get("track_id")
                if isinstance(track_id, (int, np.integer)) and track_id != -1:
                    gt_ids.append(int(track_id))  # 统一转为int
                    gt_bboxes.append(obj["bbox"])

            # 过滤预测结果中可能存在的无效ID
            pred_ids = []
            pred_bboxes = []
            for obj in pred_frame:
                track_id = obj.get("track_id")
                # 确保ID是正整数且不为-1
                if isinstance(track_id, (int, np.integer)) and track_id != -1 and track_id > 0:
                    pred_ids.append(int(track_id))  # 强制转换为Python原生int
                    pred_bboxes.append(obj["bbox"])

            # 计算IOU矩阵（处理空检测/空标注情况）
            if len(gt_ids) == 0 or len(pred_ids) == 0:
                iou_matrix = np.empty((0, 0))
            else:
                iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))
                for i, gt_bbox in enumerate(gt_bboxes):
                    for j, pred_bbox in enumerate(pred_bboxes):
                        iou_matrix[i, j] = self.compute_iou(gt_bbox, pred_bbox)

            # 确保ID列表中没有-1
            assert -1 not in gt_ids, f"GT中存在无效ID -1，帧索引: {frame_idx}"
            assert -1 not in pred_ids, f"预测结果中存在无效ID -1，帧索引: {frame_idx}"

            acc.update(gt_ids, pred_ids, iou_matrix)

        # 计算指标
        mh = metrics.create()
        summary = mh.compute(acc, metrics=metric_type)
        return {metric: summary[metric].item() for metric in metric_type}

    def compute_iou(self, bbox1, bbox2):
        """计算两个边界框的交并比（IoU）"""
        x1, y1, x2, y2 = bbox1
        x1_, y1_, x2_, y2_ = bbox2

        # 计算交集区域
        inter_x1 = max(x1, x1_)
        inter_y1 = max(y1, y1_)
        inter_x2 = min(x2, x2_)
        inter_y2 = min(y2, y2_)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # 计算两个边界框的面积
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_ - x1_) * (y2_ - y1_)

        # 计算并集区域
        union_area = area1 + area2 - inter_area

        # 计算IoU
        return inter_area / union_area if union_area > 0 else 0.0

    def visualize_tracks(self, image, tracks):
        """可视化跟踪结果，支持多种显示模式"""
        vis_image = image.copy()

        # 添加帧率和目标计数信息
        track_count = len(set(t.id for t in tracks))
        class_count = defaultdict(int)
        for t in tracks:
            class_count[t.class_name] += 1

        info_text = [f"Tracks: {track_count}"]
        for cls, count in class_count.items():
            info_text.append(f"{cls}: {count}")

        # 绘制信息面板
        panel_height = 20 * len(info_text) + 10
        cv2.rectangle(vis_image, (0, 0), (200, panel_height), (0, 0, 0), -1)
        for i, text in enumerate(info_text):
            cv2.putText(vis_image, text, (10, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 绘制跟踪框和ID
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            # 确保坐标在图像范围内
            x1, y1 = max(0, x1), max(panel_height, y1)  # 避开信息面板
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            # 为不同类别分配不同颜色
            color = self.get_class_color(track.class_name)

            # 绘制边界框（根据跟踪稳定性调整透明度）
            if hasattr(track, 'hits') and hasattr(track, 'misses'):
                stability = min(1.0, track.hits / (track.hits + track.misses + 1e-5))
                thickness = max(2, int(3 * stability))
            else:
                thickness = 2

            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)

            # 绘制ID和类别
            label = f"ID: {track.id} ({track.class_name})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            # 绘制标签背景
            cv2.rectangle(vis_image, (x1, max(panel_height, y1 - label_size[1] - 5)),
                          (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_image, label, (x1, max(panel_height, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return vis_image

    def get_class_color(self, class_name):
        """获取类别对应的颜色，支持动态生成未知类别颜色"""
        # 预设颜色映射
        color_map = {
            "red_box": (0, 0, 255),
            "blue_box": (255, 0, 0),
            "person": (0, 255, 0),
            "car": (0, 255, 255),
            "bicycle": (128, 0, 128),
            "motorcycle": (0, 128, 128),
            "bus": (128, 128, 0)
        }

        if class_name in color_map:
            return color_map[class_name]

        # 为未知类别生成稳定的哈希颜色
        import hashlib
        hash_obj = hashlib.md5(class_name.encode())
        hex_color = hash_obj.hexdigest()[:6]
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (b, g, r)  # BGR格式

    def process_unknown_classes(self, detections):
        """增强零样本处理，支持置信度过滤和动态类别更新"""
        processed = []
        zero_shot_stats = {"correct": 0, "total": 0, "unknown_classes": set()}

        # 收集所有未知类别用于批量处理
        unknown_features = []
        unknown_info = []

        for det in detections:
            bbox, feature, class_name, conf = det  # 包含置信度

            if class_name in self.known_classes:
                processed.append((bbox, feature, class_name, conf))
                continue

            # 处理未知类别
            self.unknown_classes.add(class_name)
            zero_shot_stats["unknown_classes"].add(class_name)
            unknown_features.append(feature)
            unknown_info.append((bbox, class_name, conf))
            zero_shot_stats["total"] += 1

        # 批量处理未知类别（提高效率）
        if unknown_features:
            # 逐个处理未知类别（因为没有batch_classify_unknown方法）
            for i, (bbox, orig_class, conf) in enumerate(unknown_info):
                pred_class, pred_conf = self.zero_shot_handler.classify_unknown(
                    unknown_features[i], orig_class
                )
                processed.append((bbox, unknown_features[i], pred_class, conf))

                # 更新统计信息（如果有参考真值）
                if hasattr(self, 'current_gt') and self.current_gt:
                    # 简单的真值匹配逻辑
                    matched = False
                    for gt in self.current_gt:
                        if self.compute_iou(gt["bbox"], bbox) > 0.5:
                            if pred_class == gt["class"]:
                                zero_shot_stats["correct"] += 1
                            matched = True
                            break
                    if not matched and self.experiment_mode:
                        print(f"未匹配到真值: {pred_class} {bbox}")

        return processed, zero_shot_stats

    def evaluate_on_dataset(self, dataset_name, split="test", subset=None):
        """增强数据集评估，支持更多指标和详细报告"""
        # 1. 加载数据集
        loader = UniversalDatasetLoader(dataset_name)
        frames = loader.load_frames(split=split, subset=subset)
        annotations = loader.load_annotations(split=split, subset=subset)

        # 2. 针对不同数据集类型执行评估
        metrics = {"sequences": {}, "summary": {}}
        if dataset_name in ["MOT17", "MOT20"]:
            # 多目标跟踪评估（MOTA、IDF1等）
            for seq, seq_frames in frames.items():
                self.results = []  # 清空历史结果
                self.tracker.reset()  # 重置跟踪器
                seq_metrics = defaultdict(list)

                # 按批次处理帧
                for i in tqdm(range(0, len(seq_frames), self.batch_size),
                              desc=f"评估 {seq}"):
                    batch_frames = seq_frames[i:i + self.batch_size]
                    batch_tracks = self.process_batch_frames(batch_frames, i)

                    # 保存结果
                    for tracks in batch_tracks:
                        self.results.append([
                            {"track_id": t.id, "bbox": t.bbox, "class": t.class_name}
                            for t in tracks
                        ])

                # 计算当前序列的指标
                seq_results = self.evaluate(annotations[seq])
                metrics["sequences"][seq] = seq_results

                # 累积用于汇总统计
                for k, v in seq_results.items():
                    seq_metrics[k].append(v)

            # 计算数据集平均指标
            for k, v in seq_metrics.items():
                metrics["summary"][f"avg_{k}"] = np.mean(v)
                metrics["summary"][f"std_{k}"] = np.std(v)

        elif dataset_name in ["LVIS", "ODinW"]:
            # 开放词汇检测评估（mAP、零样本准确率等）
            total = 0
            correct = 0
            self.results = []  # 存储检测结果用于mAP计算

            for img_id, frame in tqdm(frames.items(), desc="评估检测"):
                self.current_gt = annotations.get(img_id, {}).get("objects", [])
                detections = self.detect_objects(frame)  # 仅检测不跟踪
                processed, stats = self.process_unknown_classes(detections)
                self.results.extend(processed)

                correct += stats["correct"]
                total += stats["total"]

            metrics["summary"]["accuracy"] = correct / total if total > 0 else 0
            # 计算mAP（使用COCO评估工具）
            try:
                metrics["summary"]["mAP"] = self.compute_coco_map(self.results, annotations)
                metrics["summary"]["unknown_classes_count"] = len(self.unknown_classes)
            except Exception as e:
                print(f"计算mAP失败: {e}")
                metrics["summary"]["mAP"] = 0.0

        return metrics

    def compute_coco_map(self, predictions, ground_truth):
        """改进COCO格式转换，提高评估准确性"""
        # 转换结果为COCO格式
        coco_gt = COCO()
        coco_gt.dataset = ground_truth  # 需按COCO格式组织
        coco_gt.createIndex()

        # 转换预测结果为COCO格式
        coco_results = []
        for img_id, det in enumerate(predictions):
            bbox, _, class_name, score = det
            x1, y1, x2, y2 = bbox
            # 查找类别ID
            if class_name in self.known_classes:
                cat_id = self.known_classes.index(class_name) + 1
            else:
                # 为未知类别分配临时ID
                cat_id = len(self.known_classes) + list(self.unknown_classes).index(class_name) + 1

            coco_results.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO格式是[x, y, width, height]
                "score": score  # 使用实际检测置信度
            })

        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # 返回多个mAP指标
        return {
            "mAP@0.5:0.95": coco_eval.stats[0],
            "mAP@0.5": coco_eval.stats[1],
            "mAP@0.75": coco_eval.stats[2],
            "mAP_small": coco_eval.stats[3],
            "mAP_medium": coco_eval.stats[4],
            "mAP_large": coco_eval.stats[5]
        }

    def set_tracker(self, tracker):
        """设置跟踪器引用"""
        self.tracker = tracker

    def process_video(self, video_path, output_path=None, visualize=True):
        """增强视频处理，支持批处理和进度保存"""
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return None

        # 获取视频信息
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 准备输出视频
        out = None
        if output_path:
            # 确保输出目录存在
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # 处理每一帧（支持批处理）
        frame_num = 0
        self.results = []  # 重置结果存储
        self.tracker.reset()  # 重置跟踪器

        with tqdm(total=total_frames, desc="处理视频") as pbar:
            while cap.isOpened():
                # 填充帧缓冲区
                self.frame_buffer.clear()
                for _ in range(self.batch_size):
                    ret, frame = cap.read()
                    if ret:
                        self.frame_buffer.append(frame)
                    else:
                        break

                if not self.frame_buffer:
                    break

                # 批处理帧
                batch_tracks = self.process_batch_frames(
                    list(self.frame_buffer), frame_num
                )

                # 处理结果
                for i, tracks in enumerate(batch_tracks):
                    current_frame_num = frame_num + i

                    # 保存跟踪结果
                    self.results.append([
                        {"track_id": t.id, "bbox": t.bbox, "class_name": t.class_name}
                        for t in tracks
                    ])

                    # 可视化
                    if visualize:
                        vis_frame = self.visualize_tracks(self.frame_buffer[i], tracks)
                        if output_path and out:
                            out.write(vis_frame)
                        # 可选：实时显示
                        # cv2.imshow('Open Vocab MOT', vis_frame)
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break

                frame_num += len(self.frame_buffer)
                pbar.update(len(self.frame_buffer))

        # 清理资源
        cap.release()
        if out:
            out.release()
        # cv2.destroyAllWindows()

        print(f"视频处理完成，共处理{frame_num}帧")
        return self.results


# 测试完整系统
if __name__ == "__main__":
    # 定义已知类别
    known_classes = ["person", "car", "bicycle", "motorcycle", "bus"]

    # 初始化系统
    mot_system = OpenVocabMOTSystem(known_classes, experiment_mode=True, batch_size=4)


    # 可以使用示例视频或摄像头
    # 1. 使用示例视频
    # 如果没有示例视频，可以使用OpenCV创建一个简单的测试视频
    def create_test_video(output_path="test_video.mp4", duration=5):
        width, height = 640, 480
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for i in range(duration * fps):
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255

            # 人（绿色矩形）
            x = int(100 + i * 0.5) % (width - 50)  # 确保不超出边界
            cv2.rectangle(frame, (x, 200), (x + 50, 280), (0, 255, 0), -1)  # 绿色对应person

            # 车（黄色矩形）
            x = int(400 - i * 0.7) % (width - 100)
            cv2.rectangle(frame, (x, 300), (x + 100, 350), (0, 255, 255), -1)  # 黄色对应car

            # 自行车（蓝色矩形）
            x = int(200 + i * 0.3) % (width - 60)
            cv2.rectangle(frame, (x, 150), (x + 60, 190), (255, 0, 0), -1)  # 蓝色对应bicycle

            # 摩托车（青色矩形）
            x = int(300 + i * 0.4) % (width - 70)
            cv2.rectangle(frame, (x, 180), (x + 70, 220), (0, 128, 128), -1)  # 青色对应motorcycle

            # 添加未知类别（红色矩形 - 消防车）
            x = int(250 + i * 0.6) % (width - 80)
            cv2.rectangle(frame, (x, 250), (x + 80, 300), (0, 0, 255), -1)

            out.write(frame)
        out.release()
        return output_path


    # 创建测试视频
    test_video = create_test_video()

    # 处理视频
    results = mot_system.process_video(test_video, output_path="./output_video.mp4")

    # 打印部分结果
    print("\n部分跟踪结果:")
    if results:
        for i in range(min(5, len(results))):
            print(f"帧 {i}:")
            for obj in results[i]:
                print(f"  ID: {obj['track_id']}, 类别: {obj['class_name']}, BBox: {[int(x) for x in obj['bbox']]}")

    # 打印性能指标
    print("\n性能指标:")
    for metric, values in mot_system.perf_metrics.items():
        if values:
            print(f"  平均{metric}: {np.mean(values):.4f}秒")

    # 打印零样本处理统计
    if mot_system.experiment_mode and mot_system.ablation_logs["zero_shot_accuracy"]:
        print("\n零样本处理统计:")
        total_correct = sum(stats["correct"] for stats in mot_system.ablation_logs["zero_shot_accuracy"])
        total = sum(stats["total"] for stats in mot_system.ablation_logs["zero_shot_accuracy"])
        if total > 0:
            print(f"  总体准确率: {total_correct / total:.2f}")
        unknown_classes = set()
        for stats in mot_system.ablation_logs["zero_shot_accuracy"]:
            unknown_classes.update(stats["unknown_classes"])
        print(f"  检测到的未知类别: {len(unknown_classes)}个")
        print(f"  类别列表: {', '.join(unknown_classes)}")