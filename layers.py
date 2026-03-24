# layers.py
# -*- coding: utf-8 -*-


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import math


class GraphConvolution(nn.Module):

    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
        is_log_std: bool = False
    ):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.is_log_std = is_log_std
        
        # 权重初始化
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 激活函数
        if is_log_std:
            self.activation = None  # log_std层不使用激活
            # 保守初始化
            nn.init.normal_(self.weight, mean=-2.0, std=0.01)
        else:
            self.activation = activation if activation is not None else nn.ReLU()
            # Glorot初始化
            self._glorot_init()
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def _glorot_init(self):
        """Glorot uniform初始化（更保守）"""
        stdv = math.sqrt(6.0 / (self.in_features + self.out_features))
        # 缩小初始化范围，防止初始梯度过大
        stdv = stdv * 0.1  # 降低10倍
        nn.init.uniform_(self.weight, -stdv, stdv)
    
    def forward(self, x: Tensor, adj: torch.sparse.FloatTensor) -> Tensor:

        # 裁剪输入（防止过大）
        x = torch.clamp(x, min=-10.0, max=10.0)
        
        # Dropout
        if self.training and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=True)
        
        # 特征变换: X * W
        support = torch.mm(x, self.weight)
        
        # 裁剪中间结果
        support = torch.clamp(support, min=-10.0, max=10.0)
        
        # 图卷积: A * (X * W)
        output = torch.sparse.mm(adj, support)
        
        # 裁剪输出
        output = torch.clamp(output, min=-10.0, max=10.0)
        
        # 加偏置
        if self.bias is not None:
            output = output + self.bias
        
        # 激活
        if self.activation is not None:
            output = self.activation(output)
        
        # 最终裁剪
        output = torch.clamp(output, min=-10.0, max=10.0)
        
        return output


class GraphConvolutionSparse(nn.Module):

    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
        is_log_std: bool = False
    ):
        super(GraphConvolutionSparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.is_log_std = is_log_std
        
        # 权重初始化
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 激活函数
        if is_log_std:
            self.activation = None
            nn.init.normal_(self.weight, mean=-2.0, std=0.01)
        else:
            self.activation = activation if activation is not None else nn.ReLU()
            self._glorot_init()
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def _glorot_init(self):
        """Glorot uniform初始化（更保守）"""
        stdv = math.sqrt(6.0 / (self.in_features + self.out_features))
        # 缩小初始化范围，防止初始梯度过大
        stdv = stdv * 0.1  # 降低10倍
        nn.init.uniform_(self.weight, -stdv, stdv)
    
    def forward(self, x: torch.sparse.FloatTensor, adj: torch.sparse.FloatTensor) -> Tensor:

        # 稀疏dropout
        if self.training and self.dropout > 0:
            x = self._sparse_dropout(x, self.dropout)
        
        # 稀疏矩阵乘法: X * W
        support = torch.sparse.mm(x, self.weight)
        
        # 裁剪中间结果
        support = torch.clamp(support, min=-10.0, max=10.0)
        
        # 图卷积: A * (X * W)
        output = torch.sparse.mm(adj, support)
        
        # 裁剪输出
        output = torch.clamp(output, min=-10.0, max=10.0)
        
        # 加偏置
        if self.bias is not None:
            output = output + self.bias
        
        # 激活
        if self.activation is not None:
            output = self.activation(output)
        
        # 最终裁剪
        output = torch.clamp(output, min=-10.0, max=10.0)
        
        return output
    
    def _sparse_dropout(self, x: torch.sparse.FloatTensor, dropout: float) -> torch.sparse.FloatTensor:
        """稀疏张量的dropout"""
        if dropout == 0.0:
            return x
        
        keep_prob = 1.0 - dropout
        # 对非零值应用dropout
        noise = torch.rand(x._values().size(0), device=x.device)
        mask = noise < keep_prob
        
        # 保留的值需要缩放
        values = x._values()[mask] / keep_prob
        indices = x._indices()[:, mask]
        
        return torch.sparse.FloatTensor(indices, values, x.size())


class GraphAttention(nn.Module):

    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        alpha: float = 0.2,
        dropout: float = 0.0,
        concat: bool = False,
        is_log_std: bool = False
    ):
        super(GraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.is_log_std = is_log_std
        
        # 权重矩阵
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a_left = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.a_right = nn.Parameter(torch.FloatTensor(out_features, 1))
        
        if is_log_std:
            nn.init.normal_(self.W, mean=-2.0, std=0.01)
            self.activation = None
        else:
            nn.init.xavier_uniform_(self.W, gain=1.414)
            self.activation = nn.ReLU()
        
        nn.init.xavier_uniform_(self.a_left, gain=1.414)
        nn.init.xavier_uniform_(self.a_right, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, x: Tensor, adj: torch.sparse.FloatTensor) -> Tensor:

        N = x.size(0)
        
        # Dropout输入
        if self.training and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=True)
        
        # 线性变换
        Wh = torch.mm(x, self.W)  # [N, out_features]
        
        # 获取边的索引
        edge_index = adj._indices()  # [2, E]
        
        # 计算注意力系数
        e_left = torch.mm(Wh, self.a_left).squeeze(1)  # [N]
        e_right = torch.mm(Wh, self.a_right).squeeze(1)  # [N]
        
        # 边的注意力分数
        e = e_left[edge_index[0]] + e_right[edge_index[1]]  # [E]
        e = self.leakyrelu(e)
        
        # Softmax归一化（按源节点分组）
        e_max = torch.zeros(N, device=x.device)
        e_max.scatter_reduce_(0, edge_index[0], e, reduce='amax', include_self=False)
        e_exp = torch.exp(e - e_max[edge_index[0]])
        
        e_sum = torch.zeros(N, device=x.device)
        e_sum.scatter_add_(0, edge_index[0], e_exp)
        
        alpha = e_exp / (e_sum[edge_index[0]] + 1e-9)  # [E]
        
        # 聚合邻居特征
        messages = alpha.unsqueeze(1) * Wh[edge_index[1]]  # [E, out_features]
        output = torch.zeros(N, self.out_features, device=x.device)
        output.scatter_add_(0, edge_index[0].unsqueeze(1).expand(-1, self.out_features), messages)
        
        # 激活
        if self.activation is not None:
            output = self.activation(output)
        
        return output


class GraphAttentionSparse(nn.Module):

    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        alpha: float = 0.2,
        dropout: float = 0.0,
        concat: bool = False,
        is_log_std: bool = False
    ):
        super(GraphAttentionSparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.is_log_std = is_log_std
        
        # 权重矩阵
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a_left = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.a_right = nn.Parameter(torch.FloatTensor(out_features, 1))
        
        if is_log_std:
            nn.init.normal_(self.W, mean=-2.0, std=0.01)
            self.activation = None
        else:
            nn.init.xavier_uniform_(self.W, gain=1.414)
            self.activation = nn.ReLU()
        
        nn.init.xavier_uniform_(self.a_left, gain=1.414)
        nn.init.xavier_uniform_(self.a_right, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, x: torch.sparse.FloatTensor, adj: torch.sparse.FloatTensor) -> Tensor:

        N = x.size(0)
        
        # 稀疏dropout
        if self.training and self.dropout > 0:
            x = self._sparse_dropout(x, self.dropout)
        
        # 稀疏矩阵乘法
        Wh = torch.sparse.mm(x, self.W)  # [N, out_features]
        
        # Dropout
        if self.training and self.dropout > 0:
            Wh = F.dropout(Wh, p=self.dropout, training=True)
        
        # 获取边的索引
        edge_index = adj._indices()  # [2, E]
        
        # 计算注意力系数
        e_left = torch.mm(Wh, self.a_left).squeeze(1)  # [N]
        e_right = torch.mm(Wh, self.a_right).squeeze(1)  # [N]
        
        # 边的注意力分数
        e = e_left[edge_index[0]] + e_right[edge_index[1]]  # [E]
        e = self.leakyrelu(e)
        
        # Softmax归一化
        e_max = torch.zeros(N, device=Wh.device)
        e_max.scatter_reduce_(0, edge_index[0], e, reduce='amax', include_self=False)
        e_exp = torch.exp(e - e_max[edge_index[0]])
        
        e_sum = torch.zeros(N, device=Wh.device)
        e_sum.scatter_add_(0, edge_index[0], e_exp)
        
        alpha = e_exp / (e_sum[edge_index[0]] + 1e-9)  # [E]
        
        # 聚合邻居特征
        messages = alpha.unsqueeze(1) * Wh[edge_index[1]]  # [E, out_features]
        output = torch.zeros(N, self.out_features, device=Wh.device)
        output.scatter_add_(0, edge_index[0].unsqueeze(1).expand(-1, self.out_features), messages)
        
        # 激活
        if self.activation is not None:
            output = self.activation(output)
        
        return output
    
    def _sparse_dropout(self, x: torch.sparse.FloatTensor, dropout: float) -> torch.sparse.FloatTensor:
        """稀疏张量的dropout"""
        if dropout == 0.0:
            return x
        
        keep_prob = 1.0 - dropout
        noise = torch.rand(x._values().size(0), device=x.device)
        mask = noise < keep_prob
        
        values = x._values()[mask] / keep_prob
        indices = x._indices()[:, mask]
        
        return torch.sparse.FloatTensor(indices, values, x.size())


class InnerProductDecoder(nn.Module):

    
    def __init__(self, dropout: float = 0.0, activation: Optional[nn.Module] = None):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.activation = activation if activation is not None else nn.Sigmoid()
    
    def forward(self, z: Tensor) -> Tensor:
        """前向传播
        
        Args:
            z: 潜在表示 [N, hidden_dim]
            
        Returns:
            重构的邻接矩阵（展平）[N*N]
        """
        if self.training and self.dropout > 0:
            z = F.dropout(z, p=self.dropout, training=True)
        
        # 裁剪z防止过大（关键！）
        z = torch.clamp(z, min=-5.0, max=5.0)
        
        # 归一化z（关键修复！防止内积过大）
        z_norm = F.normalize(z, p=2, dim=1)  # L2归一化
        
        # 内积: Z * Z^T（现在范围在[-1, 1]）
        adj_pred = torch.mm(z_norm, z_norm.t())
        
        # 缩放到合理范围（大幅降低缩放因子！）
        adj_pred = adj_pred * 2.0  # 从5.0降低到2.0
        
        # 裁剪输出防止过大
        adj_pred = torch.clamp(adj_pred, min=-10.0, max=10.0)
        
        # 展平
        adj_pred = adj_pred.view(-1)
        
        # 不需要额外激活，因为在损失函数中使用logits
        # sigmoid会在binary_cross_entropy_with_logits中自动应用
        
        return adj_pred