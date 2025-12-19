"""集中管理所有超参数和路径配置"""
import os

# 数据集配置
DATASET_CONFIG = {
    "base_dir": os.path.join(os.path.expanduser("~"), "datasets"),  # 使用用户目录，跨平台兼容
    "datasets": {
        # 适配实际路径的MOT17配置
        "MOT17": {
            "path": "${base_dir}/MOT17",
            "split": {
                # 训练集：与实际路径 E:\pythonProject2\datasets\MOT17\train\MOT17-02-DPM 匹配
                "train": ["train/MOT17-02-DPM", "train/MOT17-04-FRCNN"],
                # 测试集：修改为实际存在的序列 MOT17-01-DPM
                "test": ["test/MOT17-01-DPM"]  # 仅保留实际存在的测试序列
            },
            "classes": ["person"],
            "format": "MOTChallenge",
            "annotation_suffix": ".txt",
            "image_dir": "img1",
            "annotation_dir": {
                "train": "gt",  # 对应train集的gt文件夹
                "test": "det"   # 对应test集的det文件夹
            }
        },
        "MOT20": {
            "path": "${base_dir}/MOT20",  # 假设 MOT20 数据集路径
            "split": {
                "train": ["train/MOT20-01", "train/MOT20-02","train/MOT20-03","train/MOT20-05"],  # 根据实际序列修改
                "test": ["test/MOT20-04","test/MOT20-06","test/MOT20-07","test/MOT20-08"]  # 根据实际序列修改
            },
            "classes": ["person"],  # MOT20 主要类别也是人
            "format": "MOTChallenge",
            "annotation_suffix": ".txt",
            "image_dir": "img1",
            "annotation_dir": {
                "train": "gt",
                "test": "det"
            }
        },
        # LVIS 开放词汇检测数据集（保持不变）
        "LVIS": {
            "path": "${base_dir}/LVIS",
            "split": {"train": "lvis_v1_train.json", "val": "lvis_v1_val.json"},
            "unknown_classes": ["bicycle", "motorcycle"],
            "format": "COCO",
            "image_dir": {  # 区分训练集和验证集的图像目录
                "train": "train/train2017",
                "val": "val/val2017"
            }
        },
        # ODinW 开放域目标检测数据集（保持不变）
        # "ODinW": {
        #     "path": "${base_dir}/ODinW",
        #     "subdatasets": ["aircraft", "caltech", "dtd"],
        #     "split": {"train": "train", "test": "test"},
        #     "format": "PascalVOC",
        #     "image_dir": "JPEGImages",
        #     "annotation_dir": "Annotations"
        # }
    }
}

# 其余配置保持不变...
KG_CONFIG = {
    "classes_path": os.path.join(os.path.dirname(__file__), "knowledge_graph/classes.npy"),
    "embeddings_path": os.path.join(os.path.dirname(__file__), "knowledge_graph/conceptnet_embeddings.npy"),
    "embedding_dim": 512,  # 知识图谱嵌入维度，需与CLIP特征维度保持一致
    "hierarchy_path": None,  # 知识图谱层级关系路径（可选）
    "relation_weights": {"is_a": 0.8, "part_of": 0.5}  # 关系权重
}

MODEL_CONFIG = {
    "yolo_model_path": os.path.join(os.path.dirname(__file__), "yolov8x-worldv2.pt"),
    "clip_model_name": "ViT-B/32",
    "kg_path": os.path.join(os.path.dirname(__file__), "knowledge_graph/knowledge_graph.json"),
    "device": "cuda" if os.environ.get("USE_CUDA", "False").lower() == "true" else "cpu"
}

# 在 config.py 末尾添加
LOSS_CONFIG = {
    "triplet_margin": 0.5,          # 三元组损失的margin值
    "classification_weight": 1.0,   # 分类损失权重
    "regression_weight": 0.5,       # 回归损失权重
    "semantic_weight": 0.3,         # 语义损失权重
    "smooth_l1_beta": 1.0           # 平滑L1损失的beta参数
}

EXPERIMENT_CONFIG = {
    "ablation": {
        "baseline": {"dynamic_weight": False, "semantic_evolution": False, "spatio_temporal": False},
        "full_model": {"dynamic_weight": True, "semantic_evolution": True, "spatio_temporal": True},
        "no_dynamic": {"dynamic_weight": False, "semantic_evolution": True, "spatio_temporal": True}
    },
    "robustness_tests": {
        "occlusion_levels": [0.3, 0.5, 0.7],
        "illumination_changes": [0.2, 0.5, 0.8],
        "small_target_scales": [0.05, 0.1, 0.2]
    }
}

OUTPUT_CONFIG = {
    "base_dir": "./experiments",
    "checkpoints": "./experiments/checkpoints",
    "results": "./experiments/results",
    "visualizations": "./experiments/videos",
    "logs": "./experiments/logs",
    "metrics": "./experiments/metrics.json"
}

SCENE_CONFIG = {
    "industrial": {
        "regions": {"work_area": [100, 100, 800, 600], "safe_area": [0, 0, 100, 600]},
        "movement_limits": {"person": 50, "car": 100, "default": 70}
    }
}

DICTIONARY_CONFIG = {
    "feature_fusion_weights": (0.6, 0.4),
    "triplet_loss_margin": 0.5,
    "max_visual_atoms": 50,
    "kmeans_clusters": 20,
    "historical_weight": 0.2,
    "hard_neg_threshold": 0.7,
    "hard_pos_threshold": 0.3,
    "num_clusters": 5,
    "cluster_threshold": 20,
    "lr": 1e-4
}

ENV_CONFIG = {
    "dependencies": {
        "python": "3.10", "torch": "2.0.1", "ultralytics": "8.0.196",
        "clip": "1.0", "scikit-learn": "1.2.2"
    },
    "random_seed": 42
}

RELIABILITY_CONFIG = {
    # 视觉可靠性参数
    "visual": {
        "sharpness_weight": 0.5,
        "occlusion_weight": 0.3,
        "light_weight": 0.2,
        "min_value": 0.2,
        "max_value": 1.0
    },
    # 语言可靠性参数
    "language": {
        "frequency_weight": 0.6,
        "ambiguity_weight": 0.4,
        "min_value": 0.3,
        "max_value": 1.0
    }
}

# 目标关联器配置（用于 SemanticAssociation 类）
ASSOCIATION_CONFIG = {
    "iou_weight": 0.3,           # IoU在代价矩阵中的权重（0-1）
    "appearance_weight": 0.4,    # 外观特征相似度权重（0-1）
    "semantic_weight": 0.2,      # 语义相似度权重（0-1）
    "uncertainty_weight": 0.1,   # 不确定性惩罚权重（0-1）
    "match_threshold": 0.7       # 匹配阈值（代价低于此值视为有效匹配）
}

VISUALIZATION_CONFIG = {
    "min_bbox_area": 500,  # 对应代码中使用的参数
    "draw_tracks": True,
    "color_map": {"person": (0, 255, 0), "car": (0, 0, 255)}
}

# 原型记忆配置（用于 ProtoMemory 类）
PROTO_MEMORY_CONFIG = {
    "ema_alpha": 0.3,  # 指数移动平均系数（控制新特征对原型的影响程度）
    "window_size": 20,  # 原型更新滑动窗口大小（保留最近多少帧的特征）
    "cluster_eps": 0.3,  # DBSCAN 聚类的邻域半径（余弦距离阈值）
    "min_cluster_size": 3,  # 最小聚类样本数（小于此数不形成聚类）
    "proto_dim": 512  # 原型特征维度（与CLIP特征维度保持一致）
}

TRACKER_CONFIG = {
    "max_age": 30, "min_hits": 1, "adaptive_update_interval": 5,
    "semantic_matrix": {
        "red_box": {"red_box": 1.0, "blue_box": 0.2, "person": 0.1, "car": 0.1},
        "blue_box": {"red_box": 0.2, "blue_box": 1.0, "person": 0.1, "car": 0.1},
        "person": {"red_box": 0.1, "blue_box": 0.1, "person": 1.0, "car": 0.3},
        "car": {"red_box": 0.1, "blue_box": 0.1, "person": 0.3, "car": 1.0}
    },
    "kalman": {"process_noise": 1e-4, "measurement_noise": 1e-3},
    "cost_matrix_weights": {"visual": 0.5, "iou": 0.3, "motion": 0.2},
    "match_threshold": 0.7
}

ZERO_SHOT_CONFIG = {
    "convex_hull_samples": 10,
    "atom_evolution_window": 10,
    "temporal_consistency_weight": 0.3,
    "feature_window_size": 10,
    "evolution_lr": 5e-5,
    "update_interval": 3,
    "min_update_samples": 3,
    "update_threshold": 0.05,
    "detection_threshold": 0.5
}

# YOLO-RD 配置
YOLO_RD_CONFIG = {
    "retrieval_top_k": 5,
    "retrieval_threshold": 0.7,
    "fusion_gate_threshold": 0.5,
    "enable_cache": True,
    "cache_size": 1000
}

# 知识增强跟踪配置
KNOWLEDGE_TRACKER_CONFIG = {
    "feature_dim": 256,
    "attention_heads": 4,
    "dropout": 0.1,
    "max_track_age": 30,
    "min_track_hits": 3
}

# 优化配置
OPTIMIZATION_CONFIG = {
    "enable_fusion": True,
    "enable_quantization": True,
    "quantization_type": "dynamic",  # dynamic, static
    "profiling_runs": 100,
    "save_results": True,
    "output_dir": "./optimization_results"
}

DETECTION_CONFIG = {
    "conf_threshold": 0.5, "default_classes": ["person", "helmet"]
}
