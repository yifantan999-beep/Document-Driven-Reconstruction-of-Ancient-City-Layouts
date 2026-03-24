# optimizer.py
# -*- coding: utf-8 -*-


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional
import numpy as np


class VAELoss(nn.Module):

    
    def __init__(
        self,
        pos_weight: float = 1.0,
        norm: float = 1.0,
        free_bits: float = 0.05,
        kl_target: float = 1.0,
        warmup_steps: int = 1000
    ):
        super(VAELoss, self).__init__()
        self.pos_weight = pos_weight
        self.norm = norm
        self.free_bits = free_bits
        self.kl_target = kl_target
        self.warmup_steps = warmup_steps
        self.global_step = 0
    
    def update_pos_weight(self, pos_weight: float):
        """更新正样本权重"""
        self.pos_weight = pos_weight
    
    def update_norm(self, norm: float):
        """更新归一化系数"""
        self.norm = norm
    
    def recon_loss(self, preds: Tensor, labels: Tensor) -> Tensor:

        loss = F.binary_cross_entropy_with_logits(
            preds, labels,
            pos_weight=torch.tensor(self.pos_weight, device=preds.device)
        )
        return self.norm * loss
    
    def kl_loss(self, z_mean: Tensor, z_log_std: Tensor) -> Tensor:

        # 裁剪z_log_std防止exp爆炸（关键修复！）
        z_log_std_clipped = torch.clamp(z_log_std, min=-10.0, max=2.0)
        
        # 每个节点的KL散度
        kl_per_node = 0.5 * torch.sum(
            z_mean ** 2 + torch.exp(2.0 * z_log_std_clipped) - 1.0 - 2.0 * z_log_std_clipped,
            dim=1
        )
        
        # 归一化：除以维度数
        kl_per_node = kl_per_node / z_mean.size(1)
        
        # 平均KL散度
        kl_raw = torch.mean(kl_per_node)
        
        # 裁剪KL防止过大（关键修复！）
        kl_raw = torch.clamp(kl_raw, max=100.0)
        
        # 自由比特（不使用预热权重，在外部控制）
        kl_free = torch.clamp(kl_raw - self.free_bits, min=0.0)
        
        return kl_free, kl_raw
    
    def forward(
        self,
        adj_recon: Tensor,
        adj_label: Tensor,
        z_mean: Tensor,
        z_log_std: Tensor,
        extra_loss: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:

        # 重构损失
        recon = self.recon_loss(adj_recon, adj_label)
        
        # KL散度损失
        kl_used, kl_raw = self.kl_loss(z_mean, z_log_std)
        
        # 总损失
        total_loss = recon + kl_used
        
        # 额外损失
        if extra_loss is not None:
            total_loss = total_loss + extra_loss
        
        # log_std惩罚（防止崩溃）
        log_std_penalty = torch.mean(torch.clamp(z_log_std + 4.0, min=0.0)) * 1e-3
        total_loss = total_loss + log_std_penalty
        
        # 更新全局步数
        self.global_step += 1
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon,
            'kl_loss': kl_used,
            'kl_raw': kl_raw,
            'log_std_penalty': log_std_penalty
        }


class AttributeLoss(nn.Module):

    
    def __init__(
        self,
        w_pos: float = 3.0,
        w_size: float = 2.0,
        w_type: float = 1.0,
        type_class_weights: Optional[np.ndarray] = None
    ):
        super(AttributeLoss, self).__init__()
        self.w_pos = w_pos
        self.w_size = w_size
        self.w_type = w_type
        
        if type_class_weights is not None:
            self.register_buffer('type_class_weights', 
                               torch.FloatTensor(type_class_weights))
        else:
            self.type_class_weights = None
    
    def forward(
        self,
        pos_pred: Tensor,
        size_pred: Tensor,
        type_logits: Tensor,
        gt_pos: Tensor,
        gt_size: Tensor,
        gt_type_id: Tensor,
        valid_mask: Tensor
    ) -> Dict[str, Tensor]:

        valid_count = valid_mask.sum() + 1e-6
        
        # 位置损失（MSE）
        pos_mse_per = torch.sum((pos_pred - gt_pos) ** 2, dim=1)
        pos_loss = torch.sum(pos_mse_per * valid_mask) / valid_count
        
        # 尺寸损失（MSE）
        size_mse_per = torch.sum((size_pred - gt_size) ** 2, dim=1)
        size_loss = torch.sum(size_mse_per * valid_mask) / valid_count
        
        # 类型损失（交叉熵）
        type_ce_per = F.cross_entropy(type_logits, gt_type_id, reduction='none')
        
        # 应用类别权重
        if self.type_class_weights is not None:
            type_weights = self.type_class_weights[gt_type_id]
            type_ce_per = type_ce_per * type_weights
        
        type_loss = torch.sum(type_ce_per * valid_mask) / valid_count
        
        # 总属性损失
        total_attr_loss = self.w_pos * pos_loss + self.w_size * size_loss + self.w_type * type_loss
        
        return {
            'attr_loss': total_attr_loss,
            'pos_loss': pos_loss,
            'size_loss': size_loss,
            'type_loss': type_loss
        }


class ConditionalPriorKL(nn.Module):

    
    def __init__(self, weight: float = 1.0, ramp_steps: int = 3000):
        super(ConditionalPriorKL, self).__init__()
        self.weight = weight
        self.ramp_steps = ramp_steps
        self.global_step = 0
    
    def forward(
        self,
        z_mean_q: Tensor,
        z_log_std_q: Tensor,
        prior_mu: Tensor,
        prior_log_std: Tensor,
        valid_mask: Tensor
    ) -> Tensor:
        """计算条件先验KL散度
        
        Args:
            z_mean_q: 后验均值 [N, hidden_dim]
            z_log_std_q: 后验对数标准差 [N, hidden_dim]
            prior_mu: 先验均值 [1, hidden_dim]
            prior_log_std: 先验对数标准差 [1, hidden_dim]
            valid_mask: 有效节点mask [N]
            
        Returns:
            KL散度损失
        """
        valid_count = valid_mask.sum() + 1e-6
        
        # 扩展先验参数
        prior_mu_tiled = prior_mu.repeat(z_mean_q.size(0), 1)
        prior_log_std_tiled = prior_log_std.repeat(z_mean_q.size(0), 1)
        
        # 计算方差
        var_q = torch.exp(2.0 * z_log_std_q)
        var_p = torch.exp(2.0 * prior_log_std_tiled)
        
        # KL散度（每个节点）
        kl_node = 0.5 * (
            2.0 * (prior_log_std_tiled - z_log_std_q) +
            (var_q + (z_mean_q - prior_mu_tiled) ** 2) / var_p - 1.0
        )
        kl_node = torch.sum(kl_node, dim=1)
        
        # 只计算有效节点的KL
        kl_cond = torch.sum(kl_node * valid_mask) / valid_count
        
        # 预热权重
        ramp = min(1.0, self.global_step / self.ramp_steps)
        kl_cond_loss = self.weight * ramp * kl_cond
        
        self.global_step += 1
        
        return kl_cond_loss


class OverlapLoss(nn.Module):
    """重叠惩罚损失
    
    惩罚预测的矩形之间的重叠（IoU）
    
    Args:
        weight: 损失权重
        iou_margin: IoU阈值（只惩罚超过此值的重叠）
        ramp_steps: 权重预热步数
        repel_margin: 中心距离排斥边界（0表示禁用）
    """
    
    def __init__(
        self,
        weight: float = 0.5,
        iou_margin: float = 0.0,
        ramp_steps: int = 2000,
        repel_margin: float = 0.06
    ):
        super(OverlapLoss, self).__init__()
        self.weight = weight
        self.iou_margin = iou_margin
        self.ramp_steps = ramp_steps
        self.repel_margin = repel_margin
        self.global_step = 0
    
    def forward(
        self,
        pos_pred: Tensor,
        size_pred: Tensor,
        valid_mask: Tensor
    ) -> Tensor:
        """计算重叠损失
        
        Args:
            pos_pred: 预测的位置 [N, 2]
            size_pred: 预测的尺寸 [N, 2]
            valid_mask: 有效节点mask [N]
            
        Returns:
            重叠损失
        """
        x = pos_pred[:, 0]
        y = pos_pred[:, 1]
        w = size_pred[:, 0]
        h = size_pred[:, 1]
        
        # 计算矩形边界
        l = x - 0.5 * w
        r = x + 0.5 * w
        b = y - 0.5 * h
        t = y + 0.5 * h
        
        # 扩展维度用于成对计算
        l1, l2 = l.unsqueeze(1), l.unsqueeze(0)
        r1, r2 = r.unsqueeze(1), r.unsqueeze(0)
        b1, b2 = b.unsqueeze(1), b.unsqueeze(0)
        t1, t2 = t.unsqueeze(1), t.unsqueeze(0)
        
        # 计算交集
        inter_w = torch.clamp(torch.min(r1, r2) - torch.max(l1, l2), min=0.0)
        inter_h = torch.clamp(torch.min(t1, t2) - torch.max(b1, b2), min=0.0)
        inter = inter_w * inter_h
        
        # 计算并集
        area = w * h
        area1, area2 = area.unsqueeze(1), area.unsqueeze(0)
        union = area1 + area2 - inter
        
        # IoU
        iou = inter / (union + 1e-6)
        
        # 只计算有效节点对（上三角，不包括对角线）
        valid_pair = valid_mask.unsqueeze(1) * valid_mask.unsqueeze(0)
        upper = torch.triu(torch.ones_like(iou), diagonal=1)
        pair_mask = valid_pair * upper
        
        # IoU惩罚
        iou_pen = torch.clamp(iou - self.iou_margin, min=0.0)
        pair_denom = pair_mask.sum() + 1e-6
        overlap_loss_raw = torch.sum(iou_pen * pair_mask) / pair_denom
        
        # 可选：中心距离排斥
        repel_loss_raw = 0.0
        if self.repel_margin > 0.0:
            dx = x.unsqueeze(1) - x.unsqueeze(0)
            dy = y.unsqueeze(1) - y.unsqueeze(0)
            dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
            repel_pen = torch.clamp(self.repel_margin - dist, min=0.0)
            repel_loss_raw = torch.sum(repel_pen ** 2 * pair_mask) / pair_denom
        
        # 预热权重
        ramp = min(1.0, self.global_step / self.ramp_steps)
        overlap_loss = self.weight * ramp * (overlap_loss_raw + repel_loss_raw)
        
        self.global_step += 1
        
        return overlap_loss