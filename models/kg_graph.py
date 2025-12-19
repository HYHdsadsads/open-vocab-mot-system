import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from config import KG_CONFIG


class HyperbolicKG(nn.Module):
    """知识图谱双曲嵌入与图注意力传播"""

    def __init__(self, kg_path: str, device="cpu"):
        super().__init__()
        self.device = device
        self.emb_dim = KG_CONFIG["embedding_dim"]
        self.graph = self._load_kg(kg_path)  # 加载知识图谱（父子类关系）
        self.node_emb = self._initialize_hyperbolic_emb()  # 双曲空间嵌入
        self.gat = GATConv(self.emb_dim, self.emb_dim, heads=2)  # 图注意力层

    def _load_kg(self, kg_path: str) -> Data:
        """加载知识图谱（格式：每个节点为类别，边为父子关系）"""
        # 示例：从JSON文件加载节点与边
        import json
        with open(kg_path, "r") as f:
            kg_data = json.load(f)
        nodes = kg_data["nodes"]  # 类别列表
        edges = kg_data["edges"]  # [(parent_idx, child_idx), ...]

        # 构建PyTorch Geometric数据结构
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return Data(edge_index=edge_index, num_nodes=len(nodes))

    def _initialize_hyperbolic_emb(self) -> nn.Embedding:
        """双曲空间嵌入初始化（使用Poincaré球模型）"""
        emb = nn.Embedding(
            self.graph.num_nodes,
            self.emb_dim,
            max_norm=1.0 - 1e-5  # 确保在Poincaré球内
        )
        nn.init.uniform_(emb.weight, -0.001, 0.001)  # 初始值靠近原点
        return emb

    def hyperbolic_distance(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """计算双曲空间距离（Poincaré球模型）"""
        norm_u = torch.norm(u, dim=-1, keepdim=True)
        norm_v = torch.norm(v, dim=-1, keepdim=True)
        numerator = torch.norm(u - v, dim=-1) ** 2
        denominator = (1 - norm_u ** 2) * (1 - norm_v ** 2)
        return 2 * torch.arctanh(torch.sqrt(numerator / denominator))

    def propagate_semantics(self) -> torch.Tensor:
        """通过图注意力传播语义信息"""
        x = self.node_emb.weight  # [num_nodes, emb_dim]
        x = self.gat(x, self.graph.edge_index.to(self.device))  # 图注意力更新
        # 投影回双曲空间（保持范数约束）
        x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8) * (1 - 1e-5)
        return x

    def get_hierarchy_similarity(self, class_idx1: int, class_idx2: int) -> float:
        """基于双曲距离的类别层级相似度"""
        emb1 = self.node_emb(torch.tensor(class_idx1, device=self.device))
        emb2 = self.node_emb(torch.tensor(class_idx2, device=self.device))
        dist = self.hyperbolic_distance(emb1, emb2).item()
        return np.clip(1 - dist / 2.0, 0.0, 1.0)  # 距离越小相似度越高