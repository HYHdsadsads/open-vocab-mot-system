import torch
import torch.nn.functional as F
from config import LOSS_CONFIG

def combined_loss(
    det_preds, track_preds,
    anchors, positives, negatives,
    hyper_emb, lang_emb,
    temp_params
):
    """
    总损失函数：检测损失 + 跟踪损失 + 一致性损失 + 双曲对齐损失 + 校准损失
    """
    # 1. 检测损失（Focal Loss）
    det_loss = FocalLoss()(det_preds["cls"], det_preds["gt_cls"]) + \
               1.5 * F.l1_loss(det_preds["bbox"], det_preds["gt_bbox"])

    # 2. 跟踪损失（三元组损失）
    track_loss = triplet_loss_with_hard_mining(anchors, positives, negatives)

    # 3. 跨模态/跨帧一致性损失
    frame_consist_loss = F.l1_loss(track_preds["curr_feat"], track_preds["prev_feat"])
    modal_consist_loss = F.l1_loss(hyper_emb, lang_emb)  # 双曲嵌入与语言嵌入一致性
    consist_loss = LOSS_CONFIG["lambda1"] * (frame_consist_loss + modal_consist_loss)

    # 4. 双曲层级对齐损失
    hyper_loss = hyperbolic_alignment_loss(hyper_emb, track_preds["hierarchy_labels"])

    # 5. 温度校准损失（使温度与可靠性正相关）
    temp_loss = F.mse_loss(temp_params["pred_temp"], temp_params["gt_reliability"])

    total_loss = det_loss + track_loss + consist_loss + \
                 LOSS_CONFIG["lambda2"] * hyper_loss + LOSS_CONFIG["lambda3"] * temp_loss
    return total_loss

class FocalLoss(torch.nn.Module):
    """用于类别不平衡的Focal Loss"""
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()

def hyperbolic_alignment_loss(hyper_emb, labels):
    """双曲嵌入与层级标签的对齐损失"""
    # 计算同类样本距离应小于异类
    loss = 0.0
    for i in range(len(hyper_emb)):
        for j in range(i + 1, len(hyper_emb)):
            dist = HyperbolicKG.hyperbolic_distance(hyper_emb[i], hyper_emb[j])
            if labels[i] == labels[j]:
                loss += torch.relu(dist - 0.5)  # 同类距离应小于0.5
            else:
                loss += torch.relu(1.0 - dist)  # 异类距离应大于1.0
    return loss / (len(hyper_emb) ** 2)