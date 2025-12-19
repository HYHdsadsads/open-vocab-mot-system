# 在项目根目录创建 generate_kg_embeddings.py 并运行
import numpy as np
from config import KG_CONFIG

# 生成示例嵌入（实际应根据ConceptNet数据生成）
# 这里创建一个随机嵌入作为临时替代（维度需与代码匹配：类别数 × 512）
num_classes = len(np.load(KG_CONFIG["classes_path"]))  # 与classes.npy保持一致
embeddings = np.random.randn(num_classes, 512).astype(np.float32)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # 归一化

# 保存为合规的npy文件（不包含pickle数据）
np.save(KG_CONFIG["embeddings_path"], embeddings, allow_pickle=False)
print(f"已生成知识图谱嵌入文件: {KG_CONFIG['embeddings_path']}")