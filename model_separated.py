# model_separated.py
# -*- coding: utf-8 -*-


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple

from layers import (
    GraphConvolution,
    GraphConvolutionSparse,
    InnerProductDecoder
)


class DirectAttributeDecoder(nn.Module):

    
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        
        # 特征提取器（深度MLP）
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # 位置解码器
        self.pos_decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Tanh()  # [-1, 1]
        )
        
        # 尺寸解码器
        self.size_decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()  # [0, 1]
        )
        
        # 类型解码器
        self.type_decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 14)  # NUM_TYPES
        )
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        # 提取特征（不经过GCN）
        feat = self.feature_extractor(x)
        
        # 解码位置
        pos = self.pos_decoder(feat)  # [-1, 1]
        
        # 解码尺寸
        size_sigmoid = self.size_decoder(feat)
        size = size_sigmoid * 0.78 + 0.02  # [0.02, 0.8]
        
        # 解码类型
        type_logits = self.type_decoder(feat)
        
        return pos, size, type_logits


class SeparatedGCNModelVAE(nn.Module):

    
    def __init__(
        self,
        num_features: int,
        hidden1: int = 256,
        hidden2: int = 128,
        hidden3: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_features = num_features
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.dropout = dropout
        
        # ===== 图结构分支（用于邻接矩阵重构） =====
        self.gc1 = GraphConvolutionSparse(num_features, hidden1, activation=nn.ReLU(), dropout=dropout)
        self.gc2 = GraphConvolution(hidden1, hidden2, activation=nn.ReLU(), dropout=dropout)
        
        # 潜在空间
        self.gc_mean = GraphConvolution(hidden2, hidden3, activation=None, dropout=dropout)
        self.gc_logstd = GraphConvolution(hidden2, hidden3, activation=None, dropout=dropout, is_log_std=True)
        
        # 邻接矩阵解码器
        self.decoder = InnerProductDecoder(dropout=dropout, activation=None)
        
        # ===== 属性分支（直接从输入学习） =====
        self.attr_decoder = DirectAttributeDecoder(
            input_dim=num_features,
            hidden_dim=256,
            dropout=dropout
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def encode(self, x: torch.sparse.FloatTensor, adj: torch.sparse.FloatTensor) -> Tuple[Tensor, Tensor]:
        """编码器（仅用于图结构）"""
        h1 = self.gc1(x, adj)
        h2 = self.gc2(h1, adj)
        
        z_mean = self.gc_mean(h2, adj)
        z_log_std = self.gc_logstd(h2, adj)
        
        z_log_std = torch.clamp(z_log_std, min=-5.0, max=2.0)
        
        return z_mean, z_log_std
    
    def reparameterize(self, mean: Tensor, log_std: Tensor) -> Tensor:
        """重参数化"""
        if self.training:
            std = torch.exp(log_std)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean
    
    def forward(self, x: torch.sparse.FloatTensor, adj: torch.sparse.FloatTensor) -> Dict[str, Tensor]:
        """前向传播
        
        关键：属性预测直接从输入特征，不经过GCN
        """
        # ===== 图结构分支 =====
        z_mean, z_log_std = self.encode(x, adj)
        z = self.reparameterize(z_mean, z_log_std)
        adj_recon = self.decoder(z)
        
        # ===== 属性分支（直接从输入） =====
        x_dense = x.to_dense() if x.is_sparse else x
        pos_pred, size_pred, type_logits = self.attr_decoder(x_dense)
        
        return {
            'z': z,
            'z_mean': z_mean,
            'z_log_std': z_log_std,
            'adj_recon': adj_recon,
            'pos_pred': pos_pred,
            'size_pred': size_pred,
            'type_logits': type_logits,
            'type_pred': F.softmax(type_logits, dim=-1)
        }


class HybridGCNModelVAE(nn.Module):

    
    def __init__(
        self,
        num_features: int,
        hidden1: int = 256,
        hidden2: int = 128,
        hidden3: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_features = num_features
        
        # GCN分支
        self.gc1 = GraphConvolutionSparse(num_features, hidden1, activation=nn.ReLU(), dropout=dropout)
        self.gc2 = GraphConvolution(hidden1, hidden2, activation=nn.ReLU(), dropout=dropout)
        self.gc_mean = GraphConvolution(hidden2, hidden3, activation=None, dropout=dropout)
        self.gc_logstd = GraphConvolution(hidden2, hidden3, activation=None, dropout=dropout, is_log_std=True)
        self.decoder = InnerProductDecoder(dropout=dropout, activation=None)
        
        # 直接特征分支
        self.direct_encoder = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden3 + 128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 属性解码器
        self.pos_decoder = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )
        
        self.size_decoder = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )
        
        self.type_decoder = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 14)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def encode(self, x, adj):
        h1 = self.gc1(x, adj)
        h2 = self.gc2(h1, adj)
        z_mean = self.gc_mean(h2, adj)
        z_log_std = self.gc_logstd(h2, adj)
        z_log_std = torch.clamp(z_log_std, min=-5.0, max=2.0)
        return z_mean, z_log_std
    
    def reparameterize(self, mean, log_std):
        if self.training:
            std = torch.exp(log_std)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean
    
    def forward(self, x, adj):
        # GCN分支
        z_mean, z_log_std = self.encode(x, adj)
        z = self.reparameterize(z_mean, z_log_std)
        adj_recon = self.decoder(z)
        
        # 直接特征分支
        x_dense = x.to_dense() if x.is_sparse else x
        direct_feat = self.direct_encoder(x_dense)
        
        # 融合
        fused = self.fusion(torch.cat([z, direct_feat], dim=1))
        
        # 属性解码
        pos = self.pos_decoder(fused)
        size_sigmoid = self.size_decoder(fused)
        size = size_sigmoid * 0.78 + 0.02
        type_logits = self.type_decoder(fused)
        
        return {
            'z': z,
            'z_mean': z_mean,
            'z_log_std': z_log_std,
            'adj_recon': adj_recon,
            'pos_pred': pos,
            'size_pred': size,
            'type_logits': type_logits,
            'type_pred': F.softmax(type_logits, dim=-1)
        }