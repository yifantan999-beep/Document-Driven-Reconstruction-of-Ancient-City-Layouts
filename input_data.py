# input_data.py
# -*- coding: utf-8 -*-


from __future__ import annotations

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os
from typing import List, Tuple, Optional
import torch


# 类型映射
type_mapping = {
    '宫殿': 0, '官署': 1, '寺院': 2, '道观': 3, 
    '学校': 4, '园林': 5, 
    '市民生活': 6, '经济活动设施': 7, '军事设施': 8,
    '社会福利设施': 9, 
    '府邸宅邸': 10, '民居': 11, '边界': 12, None: 13
}

NUM_TYPES = max(type_mapping.values()) + 1  # 14


def get_node_attribute(g: nx.Graph, attr_name: str, dtype, default=0.0) -> np.ndarray:

    n = g.number_of_nodes()
    vals = np.full(n, default, dtype=dtype)
    for i, node in enumerate(sorted(g.nodes())):
        vals[i] = g.nodes[node].get(attr_name, default)
    return vals


def build_feature_matrix(
    posx: np.ndarray,
    posy: np.ndarray,
    size_x: np.ndarray,
    size_y: np.ndarray,
    type_id: Optional[np.ndarray] = None,
    add_bias: bool = True
) -> sp.csr_matrix:

    n = posx.shape[0]
    geo = sp.csr_matrix(np.stack([posx, posy, size_x, size_y], axis=1), dtype=np.float32)
    
    parts = [geo]
    
    if type_id is not None:
        # 将未知类型限制到最后一个索引
        type_id = np.asarray(type_id, dtype=np.int32)
        type_id = np.clip(type_id, 0, NUM_TYPES - 1)
        rows = np.arange(n, dtype=np.int32)
        cols = type_id
        data = np.ones(n, dtype=np.float32)
        type_onehot = sp.coo_matrix((data, (rows, cols)), shape=(n, NUM_TYPES), dtype=np.float32).tocsr()
        parts.append(type_onehot)
    
    if add_bias:
        parts.append(sp.csr_matrix(np.ones((n, 1), dtype=np.float32)))
    
    return sp.hstack(parts, format='csr', dtype=np.float32)


def extract_features(G: nx.Graph, is_arch: bool = True) -> sp.csr_matrix:

    nodes = list(G.nodes())
    
    posx = np.array([G.nodes[i].get('posx', 0.0) for i in nodes], dtype=np.float32)
    posy = np.array([G.nodes[i].get('posy', 0.0) for i in nodes], dtype=np.float32)
    size_x = np.array([G.nodes[i].get('size_x', 0.1) for i in nodes], dtype=np.float32)
    size_y = np.array([G.nodes[i].get('size_y', 0.1) for i in nodes], dtype=np.float32)
    
    if is_arch:
        type_str = [G.nodes[i].get('type', None) for i in nodes]
        # 未知类型映射到最后一个索引
        type_id = np.array([type_mapping.get(t, NUM_TYPES - 1) for t in type_str], dtype=np.int32)
        return build_feature_matrix(posx, posy, size_x, size_y, type_id=type_id, add_bias=True)
    else:
        return build_feature_matrix(posx, posy, size_x, size_y, type_id=None, add_bias=True)


def load_gpickle_dataset(data_dir: str = 'processed/') -> Tuple[List[sp.csr_matrix], List[sp.csr_matrix], List[nx.Graph]]:

    adj_list = []
    features_list = []
    graphs = []
    
    for file in sorted(os.listdir(data_dir)):
        if not file.endswith('.gpickle'):
            continue
        path = os.path.join(data_dir, file)
        with open(path, 'rb') as f:
            G = pkl.load(f)
        
        # 邻接矩阵
        try:
            adj = nx.to_scipy_sparse_array(G, format='csr')  # NetworkX >= 3.0
        except AttributeError:
            adj = nx.to_scipy_sparse_matrix(G, format='csr')  # 兼容旧版本
        
        adj_list.append(adj)
        
        # 节点特征
        n = G.number_of_nodes()
        posx = get_node_attribute(G, 'posx', np.float32, default=0.0)
        posy = get_node_attribute(G, 'posy', np.float32, default=0.0)
        size_x = get_node_attribute(G, 'size_x', np.float32, default=0.0)
        size_y = get_node_attribute(G, 'size_y', np.float32, default=0.0)
        type_str = get_node_attribute(G, 'type', str, default=None)
        
        # 安全映射
        type_id = np.array([
            type_mapping[t] if t in type_mapping else 999
            for t in type_str
        ], dtype=np.int32)
        
        # 合并特征：几何(4) + one-hot(type) + bias
        feats_sp = build_feature_matrix(posx, posy, size_x, size_y, type_id=type_id, add_bias=True)
        features_list.append(feats_sp)
        
        graphs.append(G)
    
    return adj_list, features_list, graphs


def load_dual_dataset(
    arch_dir: str = 'processed/',
    bound_dir: str = 'boundary_nodes/'
) -> Tuple[Tuple[List, List, List], Tuple[List, List, List]]:

    arch_adj, arch_feat, arch_graphs = load_gpickle_dataset(arch_dir)
    bound_adj, bound_feat, bound_graphs = load_gpickle_dataset(bound_dir)
    
    # 边界图：无边 → 单位矩阵
    bound_adj = [sp.eye(g.number_of_nodes(), format='csr') for g in bound_graphs]
    
    return (arch_adj, arch_feat, arch_graphs), (bound_adj, bound_feat, bound_graphs)


def load_triple_dataset(
    arch1_dir: str = 'linan/building/',
    arch2_dir: Optional[str] = 'linan/orientation/',
    bound_dir: Optional[str] = 'linan/boundary/'
) -> Tuple[Tuple[List, List, List], Tuple[List, List, List], Tuple[List, List, List]]:

    arch1_files = sorted([f for f in os.listdir(arch1_dir) if f.endswith('.gpickle')])
    
    arch1_adj, arch1_feat = [], []
    arch2_adj, arch2_feat = [], []
    bound_adj, bound_feat = [], []
    
    # 如果只加载建筑图1
    if arch2_dir is None or bound_dir is None:
        for arch1_file in arch1_files:
            arch1_path = os.path.join(arch1_dir, arch1_file)
            
            # 加载建筑图1
            G_arch1 = pkl.load(open(arch1_path, 'rb'))
            adj_arch1 = nx.to_scipy_sparse_array(G_arch1, format='csr')
            feat_arch1 = extract_features(G_arch1, is_arch=True)
            
            arch1_adj.append(adj_arch1)
            arch1_feat.append(feat_arch1)
        
        return (arch1_adj, arch1_feat, arch1_files), \
               ([], [], []), \
               ([], [], [])
    
    # 加载完整的三数据集
    valid_files = []
    for arch1_file in arch1_files:
        arch2_file = arch1_file  # 同名
        bound_file = arch1_file  # 同名
        
        arch1_path = os.path.join(arch1_dir, arch1_file)
        arch2_path = os.path.join(arch2_dir, arch2_file)
        bound_path = os.path.join(bound_dir, bound_file)
        
        if not os.path.exists(arch2_path) or not os.path.exists(bound_path):
            print(f"跳过 {arch1_file} (缺少配对文件)")
            continue
        
        # 加载建筑图1
        G_arch1 = pkl.load(open(arch1_path, 'rb'))
        adj_arch1 = nx.to_scipy_sparse_array(G_arch1, format='csr')
        feat_arch1 = extract_features(G_arch1, is_arch=True)
        
        # 加载建筑图2
        G_arch2 = pkl.load(open(arch2_path, 'rb'))
        adj_arch2 = nx.to_scipy_sparse_array(G_arch2, format='csr')
        feat_arch2 = extract_features(G_arch2, is_arch=True)
        
        # 加载边界图（无边）
        G_bound = pkl.load(open(bound_path, 'rb'))
        adj_bound = sp.eye(G_bound.number_of_nodes(), format='csr')
        feat_bound = extract_features(G_bound, is_arch=True)
        
        arch1_adj.append(adj_arch1)
        arch1_feat.append(feat_arch1)
        
        arch2_adj.append(adj_arch2)
        arch2_feat.append(feat_arch2)
        
        bound_adj.append(adj_bound)
        bound_feat.append(feat_bound)
        
        valid_files.append(arch1_file)
    
    return (arch1_adj, arch1_feat, valid_files), \
           (arch2_adj, arch2_feat, valid_files), \
           (bound_adj, bound_feat, valid_files)