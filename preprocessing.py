# preprocessing.py
# -*- coding: utf-8 -*-


from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor
from typing import Tuple


def sparse_to_tuple(mx: sp.spmatrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    mx = mx.tocoo().astype(np.float32)
    coords = np.vstack((mx.row, mx.col)).transpose().astype(np.int64)
    values = mx.data.astype(np.float32)
    shape = np.array(mx.shape, dtype=np.int64)
    return coords, values, shape


def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.sparse.FloatTensor:

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_graph(adj: sp.spmatrix) -> sp.spmatrix:

    adj = adj.tocsr().astype(np.float32)
    adj_ = adj + sp.eye(adj.shape[0], dtype=np.float32, format='csr')
    rowsum = np.array(adj_.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum > 0)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    adj_norm = (d_mat @ adj_ @ d_mat).tocoo()
    return adj_norm


def preprocess_graph_torch(adj: sp.spmatrix) -> torch.sparse.FloatTensor:

    adj_norm = preprocess_graph(adj)
    return sparse_mx_to_torch_sparse_tensor(adj_norm)


def normalize_features(features: sp.spmatrix) -> sp.spmatrix:

    features = features.tocsr().astype(np.float32)
    rowsum = np.array(features.sum(1)).flatten()
    r_inv = np.power(rowsum, -1.0, where=rowsum > 0)
    r_inv[~np.isfinite(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv @ features
    return features


def to_dense_tensor(sparse_mx: sp.spmatrix) -> Tensor:

    return torch.FloatTensor(sparse_mx.toarray())