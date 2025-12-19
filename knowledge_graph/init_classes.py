# 在knowledge_graph目录下创建init_classes.py并运行
import numpy as np

# 初始化常见类别列表
default_classes = ["person", "car", "bicycle", "motorcycle", "bus", "red_box", "blue_box"]
np.save("classes.npy", default_classes)
