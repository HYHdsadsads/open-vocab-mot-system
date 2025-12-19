"""统一模型加载管理器"""
from config import MODEL_CONFIG
import clip
from ultralytics import YOLOWorld
import torch

class ModelLoader:
    """模型加载管理器，统一处理CLIP和YOLO模型的加载与管理"""

    _instance = None  # 单例模式

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # 设备配置
        self.device = MODEL_CONFIG["device"] or ("cuda" if torch.cuda.is_available() else "cpu")

        # 模型缓存
        self._clip_model = None
        self._clip_preprocess = None
        self._yolo_model = None

        self._initialized = True

    def load_clip(self, force_reload=False):
        """加载CLIP模型及预处理函数"""
        if self._clip_model is None or force_reload:
            self._clip_model, self._clip_preprocess = clip.load(
                MODEL_CONFIG["clip_model_name"],
                device=self.device
            )
        return self._clip_model, self._clip_preprocess, self.device

    def load_yolo(self, force_reload=False):
        """加载YOLO模型"""
        if self._yolo_model is None or force_reload:
            self._yolo_model = YOLOWorld(MODEL_CONFIG["yolo_model_path"])
        return self._yolo_model

    def get_clip_model(self):
        """获取已加载的CLIP模型"""
        if self._clip_model is None:
            return self.load_clip()
        return self._clip_model, self._clip_preprocess, self.device

    def get_yolo_model(self):
        """获取已加载的YOLO模型"""
        if self._yolo_model is None:
            return self.load_yolo()
        return self._yolo_model

    def release_memory(self):
        """释放模型占用的内存"""
        if self._clip_model is not None:
            del self._clip_model
            self._clip_model = None

        if self._yolo_model is not None:
            del self._yolo_model
            self._yolo_model = None

        torch.cuda.empty_cache()

# 提供便捷的模块级函数接口
model_loader = ModelLoader()

def load_clip(device=None):
    """兼容旧接口的CLIP加载函数"""
    if device is not None:
        model_loader.device = device
    return model_loader.load_clip()

def load_yolo():
    """兼容旧接口的YOLO加载函数"""
    return model_loader.load_yolo()