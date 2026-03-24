# train_separated.py
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import networkx as nx
import argparse
import torch
import torch.optim as optim
from typing import List, Tuple
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model_separated import SeparatedGCNModelVAE, HybridGCNModelVAE
from optimizer import VAELoss, OverlapLoss
from preprocessing import preprocess_graph_torch, sparse_mx_to_torch_sparse_tensor
from input_data import extract_features, type_mapping, NUM_TYPES

TYPE_ID_TO_NAME = {v: k for k, v in type_mapping.items()}


def parse_args():
    parser = argparse.ArgumentParser(description='Separated Architecture Training')
    

    parser.add_argument('--model_type', type=str, default='hybrid', 
                       choices=['separated', 'hybrid'])
    

    parser.add_argument('--hidden1', type=int, default=256)
    parser.add_argument('--hidden2', type=int, default=128)
    parser.add_argument('--hidden3', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--warmup_epochs', type=int, default=1000)
    

    parser.add_argument('--w_pos', type=float, default=0.2)
    parser.add_argument('--w_size', type=float, default=0.2)
    parser.add_argument('--w_type', type=float, default=0.1)
    parser.add_argument('--w_overlap', type=float, default=0.5)
    parser.add_argument('--w_vae', type=float, default=0.1)
    

    parser.add_argument('--kl_weight', type=float, default=0.001)
    parser.add_argument('--kl_warmup_steps', type=int, default=5000)
    

    parser.add_argument('--arch1_dir', type=str, default='out2/train/a')
    parser.add_argument('--arch2_dir', type=str, default='out2/train/b')
    parser.add_argument('--bound_dir', type=str, default='out2/train/c')
    parser.add_argument('--test_arch1_dir', type=str, default='out2/test/a')
    parser.add_argument('--test_arch2_dir', type=str, default='out2/test/b')
    parser.add_argument('--test_bound_dir', type=str, default='out2/test/c')
    parser.add_argument('--save_dir', type=str, default='models_pytorch/')
    parser.add_argument('--output_dir', type=str, default='out/')
    parser.add_argument('--vis_dir', type=str, default='visualization_separated/')
    

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--grad_clip', type=float, default=5.0)  
    parser.add_argument('--visualize', action='store_true')
    

    parser.add_argument('--mask_input_type', action='store_true', default=True
                       )
    parser.add_argument('--no_mask_input_type', dest='mask_input_type', action='store_false'
                       )
    parser.add_argument('--mask_ratio', type=float, default=0.8
                       )
    

    parser.add_argument('--ablation_config', type=int, default=4, choices=[1, 2, 3, 4]
                       )
    

    
    args = parser.parse_args()
    

    if args.ablation_config is not None:
        if args.ablation_config == 1:

            args.use_arch2 = False
            args.use_bound = False
        elif args.ablation_config == 2:

            args.use_arch2 = False
            args.use_bound = True
        elif args.ablation_config == 3:

            args.use_arch2 = True
            args.use_bound = False
        elif args.ablation_config == 4:

            args.use_arch2 = True
            args.use_bound = True
    
    return args


def load_triple_dataset(arch1_dir: str, arch2_dir: str, bound_dir: str,
                        use_arch2: bool = True, use_bound: bool = True) -> Tuple[List, List, List]:

    files = sorted([f for f in os.listdir(arch1_dir) if f.endswith('.gpickle')])
    adj_list, feat_list = [], []
    
    # 确定输入配置
    input_config = "arch1"
    if use_arch2:
        input_config += "+arch2"
    if use_bound:
        input_config += "+bound"
    
    print(f"\n加载数据集 (消融实验配置: {input_config})...")
    print(f"  arch1目录: {arch1_dir}")
    if use_arch2:
        print(f"  arch2目录: {arch2_dir}")
    if use_bound:
        print(f"  bound目录: {bound_dir}")
    
    mismatch_count = 0
    
    for file in files:
        # 加载arch1（主图）- 作为基准
        path1 = os.path.join(arch1_dir, file)
        G1 = pkl.load(open(path1, 'rb'))
        adj1 = nx.to_scipy_sparse_array(G1, format='csr')
        feat1 = extract_features(G1, is_arch=True)  # [N1, 18]
        n1 = feat1.shape[0]  # arch1的节点数作为基准
        
        # 根据配置加载其他图
        feat_list_to_concat = [feat1]  # 始终包含arch1
        
        if use_arch2:
            # 加载arch2（辅助图）
            path2 = os.path.join(arch2_dir, file)
            G2 = pkl.load(open(path2, 'rb'))
            feat2 = extract_features(G2, is_arch=True)  # [N2, 18]
            n2 = feat2.shape[0]
            
            # 对齐arch2到arch1的节点数
            if n2 != n1:
                mismatch_count += 1
                if n2 < n1:
                    pad_rows = n1 - n2
                    feat2 = sp.vstack([feat2, sp.csr_matrix((pad_rows, feat2.shape[1]))])
                elif n2 > n1:
                    feat2 = feat2[:n1]
            
            feat_list_to_concat.append(feat2)
        
        if use_bound:
            # 加载bound（边界图）
            path_bound = os.path.join(bound_dir, file)
            G_bound = pkl.load(open(path_bound, 'rb'))
            feat_bound = extract_features(G_bound, is_arch=True)  # [N_bound, 18]
            n_bound = feat_bound.shape[0]
            
            # 对齐bound到arch1的节点数
            if n_bound != n1:
                mismatch_count += 1
                if n_bound < n1:
                    pad_rows = n1 - n_bound
                    feat_bound = sp.vstack([feat_bound, sp.csr_matrix((pad_rows, feat_bound.shape[1]))])
                elif n_bound > n1:
                    feat_bound = feat_bound[:n1]
            
            feat_list_to_concat.append(feat_bound)
        
        # 特征拼接（水平拼接）
        if len(feat_list_to_concat) == 1:
            feat_combined = feat1  # 只有arch1
        else:
            feat_combined = sp.hstack(feat_list_to_concat, format='csr')
        
        adj_list.append(adj1)  # 只使用arch1的邻接矩阵
        feat_list.append(feat_combined)
    
    print(f"  加载完成: {len(files)}个图")
    if mismatch_count > 0:
        print(f"  节点数不一致的图: {mismatch_count}/{len(files)}")
    
    # 计算特征维度
    feat_dim = feat_list[0].shape[1]
    feat_dim_desc = "18(arch1)"
    if use_arch2:
        feat_dim_desc += "+18(arch2)"
    if use_bound:
        feat_dim_desc += "+18(bound)"
    print(f"  特征维度: {feat_dim} ({feat_dim_desc})")
    
    return adj_list, feat_list, files


def pad_matrix(mat: sp.spmatrix, size: int) -> sp.spmatrix:
    if mat.shape[0] >= size:
        return mat[:size, :size]
    pad_row = size - mat.shape[0]
    pad_col = size - mat.shape[1]
    return sp.vstack([
        sp.hstack([mat, sp.csr_matrix((mat.shape[0], pad_col))]),
        sp.csr_matrix((pad_row, size))
    ])[:size, :size]


def pad_features(feat: sp.spmatrix, size: int) -> sp.spmatrix:
    if feat.shape[0] >= size:
        return feat[:size]
    return sp.vstack([feat, sp.csr_matrix((size - feat.shape[0], feat.shape[1]))])[:size]


def make_partial_eye(max_nodes: int, n_eff: int) -> sp.spmatrix:
    diag = np.zeros((max_nodes,), dtype=np.float32)
    diag[:n_eff] = 1.0
    return sp.dia_matrix((diag[np.newaxis, :], [0]), shape=(max_nodes, max_nodes))


def compute_attribute_loss(pos_pred, size_pred, type_logits, gt_pos, gt_size, gt_type_id, 
                           valid_mask, w_pos, w_size, w_type):
    """计算属性损失（简化版，直接MSE）"""
    valid_count = valid_mask.sum() + 1e-6
    
    # 位置损失（MSE）
    pos_loss = torch.sum(((pos_pred - gt_pos) ** 2).sum(dim=1) * valid_mask) / valid_count
    
    # 尺寸损失（MSE）
    size_loss = torch.sum(((size_pred - gt_size) ** 2).sum(dim=1) * valid_mask) / valid_count
    
    # 类型损失
    type_loss = torch.sum(F.cross_entropy(type_logits, gt_type_id, reduction='none') * valid_mask) / valid_count
    
    total = w_pos * pos_loss + w_size * size_loss + w_type * type_loss
    
    return {
        'attr_loss': total,
        'pos_loss': pos_loss,
        'size_loss': size_loss,
        'type_loss': type_loss
    }


def visualize_sample(gt_pos, gt_size, pred_pos, pred_size, save_path, title=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    for i in range(len(gt_pos)):
        x, y = gt_pos[i]
        w, h = gt_size[i]
        rect = plt.Rectangle((x-w/2, y-h/2), w, h, fill=False, edgecolor='blue', linewidth=1)
        ax1.add_patch(rect)
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.set_title('Ground Truth')
    ax1.grid(True, alpha=0.3)
    
    for i in range(len(pred_pos)):
        x, y = pred_pos[i]
        w, h = pred_size[i]
        rect = plt.Rectangle((x-w/2, y-h/2), w, h, fill=False, edgecolor='red', linewidth=1)
        ax2.add_patch(rect)
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.set_title('Prediction')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_model(args, model, device, data_dict):
    print("\n" + "="*60)
    print(f"分离式架构训练 - 消融实验 ({args.model_type})")
    print("="*60)
    
    # 确定输入配置
    input_config = "arch1"
    if args.use_arch2:
        input_config += "+arch2"
    if args.use_bound:
        input_config += "+bound"
    
    # 显示消融实验配置编号
    config_num = None
    if not args.use_arch2 and not args.use_bound:
        config_num = 1
    elif not args.use_arch2 and args.use_bound:
        config_num = 2
    elif args.use_arch2 and not args.use_bound:
        config_num = 3
    elif args.use_arch2 and args.use_bound:
        config_num = 4
    
    feat_dim = data_dict['feat_list'][0].shape[1]
    if config_num:
        print(f"消融实验配置: {config_num}/4")
    print(f"输入配置: {input_config}")
    print(f"特征维度: {feat_dim}")
    print(f"关键特性: 属性直接从输入学习，不经过GCN")
    if args.mask_input_type:
        if args.mask_ratio >= 0.99:
            print(f"⚠️  Type学习模式: 完全MASK输入type特征")
        elif args.mask_ratio <= 0.01:
            print(f"Type学习模式: 不mask type特征")
        else:
            print(f"⚠️  Type学习模式: 部分MASK (mask比例={args.mask_ratio*100:.0f}%)")
    else:
        print(f"Type学习模式: 使用完整输入特征")
    print(f"批量大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"位置权重: {args.w_pos}")
    print(f"尺寸权重: {args.w_size}")
    print(f"类型权重: {args.w_type}")
    print(f"VAE权重: {args.w_vae} (大幅降低)")
    
    adj_list = data_dict['adj_list']
    feat_list = data_dict['feat_list']
    max_nodes = data_dict['max_nodes']
    num_graphs = data_dict['num_graphs']
    
    # 损失函数
    vae_loss = VAELoss(free_bits=0.0, kl_target=0.05)
    overlap_loss = OverlapLoss(weight=args.w_overlap, iou_margin=0.0, ramp_steps=2000).to(device)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=2000, T_mult=2, eta_min=1e-6
    )
    
    perm = np.random.permutation(num_graphs)
    sample_idx = 0
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        
        kl_weight_current = args.kl_weight * min(1.0, global_step / args.kl_warmup_steps)
        
        if global_step < args.warmup_epochs:
            lr_scale = global_step / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate * lr_scale
        
        batch_indices = []
        for _ in range(args.batch_size):
            if sample_idx >= num_graphs:
                perm = np.random.permutation(num_graphs)
                sample_idx = 0
            batch_indices.append(perm[sample_idx])
            sample_idx += 1
        
        optimizer.zero_grad()
        batch_total_loss = 0.0
        batch_recon_loss = 0.0
        batch_kl_loss = 0.0
        batch_pos_loss = 0.0
        batch_size_loss = 0.0
        batch_type_loss = 0.0
        batch_overlap_loss = 0.0
        
        for idx in batch_indices:
            n_real = int(adj_list[idx].shape[0])
            
            adj = pad_matrix(adj_list[idx], max_nodes)
            feat = pad_features(feat_list[idx], max_nodes)
            
            n_edges = adj_list[idx].nnz
            n_possible = n_real * n_real
            n_neg = n_possible - 2 * n_edges
            
            if n_edges > 0 and n_neg > 0:
                pos_weight = float(n_neg) / float(2 * n_edges)
                norm = float(n_possible) / float(n_neg * 2)
            else:
                pos_weight = 1.0
                norm = 1.0
            
            vae_loss.update_pos_weight(pos_weight)
            vae_loss.update_norm(norm)
            
            adj_norm = preprocess_graph_torch(adj).to(device)
            adj_orig = adj.copy()
            adj_orig -= sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
            adj_orig.eliminate_zeros()
            eye_partial = make_partial_eye(max_nodes, n_real)
            adj_label = (adj_orig + eye_partial).toarray().reshape(-1)
            adj_label = torch.FloatTensor(adj_label).to(device)
            
            x = sparse_mx_to_torch_sparse_tensor(feat).to(device)
            
            # 保存原始特征用于提取ground truth
            x_orig = x.to_dense()
            
            # 如果启用mask_input_type，部分mask type特征
            # 强制模型从位置、尺寸等其他特征学习type
            if args.mask_input_type and args.mask_ratio > 0:
                x_dense = x_orig.clone()
                
                # 生成随机mask（每个节点独立随机）
                mask = torch.rand(max_nodes, 1, device=device) < args.mask_ratio
                
                # 根据输入配置mask对应的type特征
                # arch1的type特征（始终存在）
                x_dense[:, 4:4+NUM_TYPES] *= (1 - mask.float())
                
                # arch2的type特征（如果使用arch2）
                if args.use_arch2:
                    x_dense[:, 18+4:18+4+NUM_TYPES] *= (1 - mask.float())
                
                # bound的type特征（如果使用bound）
                if args.use_bound:
                    offset = 18 if not args.use_arch2 else 36
                    x_dense[:, offset+4:offset+4+NUM_TYPES] *= (1 - mask.float())
                
                # 转换回稀疏张量（GCN需要稀疏输入）
                x = x_dense.to_sparse()
            
            outputs = model(x=x, adj=adj_norm)
            
            # VAE损失（降低权重）
            vae_losses = vae_loss(
                adj_recon=outputs['adj_recon'],
                adj_label=adj_label,
                z_mean=outputs['z_mean'],
                z_log_std=outputs['z_log_std']
            )
            
            # 属性损失（ground truth从原始特征提取，不是mask后的）
            valid_mask = torch.zeros(max_nodes, device=device)
            valid_mask[:n_real] = 1.0
            # 注意：x_orig是[N, 54]，前18维是arch1的特征
            gt_pos = x_orig[:, 0:2]  # arch1的位置
            gt_size = x_orig[:, 2:4]  # arch1的尺寸
            gt_type_id = torch.argmax(x_orig[:, 4:4+NUM_TYPES], dim=1)  # arch1的类型（从原始特征）
            
            attr_losses = compute_attribute_loss(
                outputs['pos_pred'], outputs['size_pred'], outputs['type_logits'],
                gt_pos, gt_size, gt_type_id, valid_mask,
                args.w_pos, args.w_size, args.w_type
            )
            
            # 重叠损失
            overlap_l = overlap_loss(
                pos_pred=outputs['pos_pred'],
                size_pred=outputs['size_pred'],
                valid_mask=valid_mask
            )
            
            # 总损失（VAE权重大幅降低）
            kl_loss = vae_losses['kl_raw'] * kl_weight_current
            recon_loss = vae_losses['recon_loss']
            sample_loss = (
                args.w_vae * (recon_loss + kl_loss) +  # VAE损失权重降低
                attr_losses['attr_loss'] +              # 属性损失权重为1
                overlap_l
            )
            
            normalized_loss = sample_loss / args.batch_size
            normalized_loss.backward()
            
            batch_total_loss += sample_loss.item()
            batch_recon_loss += recon_loss.item()
            batch_kl_loss += vae_losses['kl_raw'].item()
            batch_pos_loss += attr_losses['pos_loss'].item()
            batch_size_loss += attr_losses['size_loss'].item()
            batch_type_loss += attr_losses['type_loss'].item()
            batch_overlap_loss += overlap_l.item()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        
        if global_step >= args.warmup_epochs:
            scheduler.step()
        
        global_step += 1
        
        avg_total_loss = batch_total_loss / args.batch_size
        avg_recon_loss = batch_recon_loss / args.batch_size
        avg_kl_loss = batch_kl_loss / args.batch_size
        avg_pos_loss = batch_pos_loss / args.batch_size
        avg_size_loss = batch_size_loss / args.batch_size
        avg_type_loss = batch_type_loss / args.batch_size
        avg_overlap_loss = batch_overlap_loss / args.batch_size
        
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
        
        if epoch % args.print_every == 0 or epoch == args.epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:6d} | "
                  f"总损失: {avg_total_loss:.4f} (最佳: {best_loss:.4f})")
            print(f"  位置损失: {avg_pos_loss:.4f} | "
                  f"尺寸损失: {avg_size_loss:.4f} | "
                  f"类型损失: {avg_type_loss:.4f}")
            print(f"  重叠损失: {avg_overlap_loss:.4f} | "
                  f"VAE损失: {avg_recon_loss:.4f} | "
                  f"KL损失: {avg_kl_loss:.4f}")
            print(f"  学习率: {current_lr:.6f}")
            
            if args.visualize and epoch % 1000 == 0 and epoch > 0:
                model.eval()
                with torch.no_grad():
                    idx = batch_indices[0]
                    n_real = int(adj_list[idx].shape[0])
                    adj = pad_matrix(adj_list[idx], max_nodes)
                    feat = pad_features(feat_list[idx], max_nodes)
                    adj_norm = preprocess_graph_torch(adj).to(device)
                    x = sparse_mx_to_torch_sparse_tensor(feat).to(device)
                    outputs = model(x=x, adj=adj_norm)
                    
                    x_dense = x.to_dense().cpu().numpy()
                    # ground truth从arch1提取（前18维）
                    gt_pos = x_dense[:n_real, 0:2]  # arch1的位置
                    gt_size = x_dense[:n_real, 2:4]  # arch1的尺寸
                    pred_pos = outputs['pos_pred'][:n_real].cpu().numpy()
                    pred_size = outputs['size_pred'][:n_real].cpu().numpy()
                    
                    vis_path = os.path.join(args.vis_dir, f'epoch_{epoch}.png')
                    visualize_sample(gt_pos, gt_size, pred_pos, pred_size, vis_path,
                                   title=f'Epoch {epoch} - Pos: {avg_pos_loss:.4f}, Size: {avg_size_loss:.4f}')
                model.train()
    
    print(f"\n训练完成！最佳损失: {best_loss:.4f}")
    return model, optimizer


def reconstruct_dataset(args, model, device, data_dict, file_names, output_dir, dataset_name="训练集"):
    print(f"\n开始重建{dataset_name}...")
    model.eval()
    
    adj_list = data_dict['adj_list']
    feat_list = data_dict['feat_list']
    max_nodes = data_dict['max_nodes']
    num_graphs = data_dict['num_graphs']
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_pos_error = 0.0
    total_size_error = 0.0
    total_type_acc = 0.0
    total_nodes = 0
    
    # 检查type是否完全一致（用于诊断）
    perfect_match_count = 0
    total_graphs = 0
    
    with torch.no_grad():
        for idx in range(num_graphs):
            n_real = int(adj_list[idx].shape[0])
            adj = pad_matrix(adj_list[idx], max_nodes)
            feat = pad_features(feat_list[idx], max_nodes)
            
            adj_norm = preprocess_graph_torch(adj).to(device)
            x = sparse_mx_to_torch_sparse_tensor(feat).to(device)
            
            # 保存原始特征用于提取ground truth
            x_orig = x.to_dense()
            
            # 测试时使用相同的mask策略
            if args.mask_input_type and args.mask_ratio > 0:
                x_dense = x_orig.clone()
                
                # 测试时使用固定的mask比例（不随机）
                # 将type特征乘以(1-mask_ratio)，相当于减弱type信息
                # arch1的type特征（始终存在）
                x_dense[:, 4:4+NUM_TYPES] *= (1 - args.mask_ratio)
                
                # arch2的type特征（如果使用arch2）
                if args.use_arch2:
                    x_dense[:, 18+4:18+4+NUM_TYPES] *= (1 - args.mask_ratio)
                
                # bound的type特征（如果使用bound）
                if args.use_bound:
                    offset = 18 if not args.use_arch2 else 36
                    x_dense[:, offset+4:offset+4+NUM_TYPES] *= (1 - args.mask_ratio)
                
                # 转换回稀疏张量
                x = x_dense.to_sparse()
            
            outputs = model(x=x, adj=adj_norm)
            
            adj_recon = torch.sigmoid(outputs['adj_recon']).view(max_nodes, max_nodes)
            pos_pred = outputs['pos_pred'][:n_real].cpu().numpy()
            size_pred = outputs['size_pred'][:n_real].cpu().numpy()
            type_pred = torch.argmax(outputs['type_logits'][:n_real], dim=1).cpu().numpy()
            type_prob = torch.softmax(outputs['type_logits'][:n_real], dim=1).cpu().numpy()
            
            adj_recon_np = (adj_recon[:n_real, :n_real].cpu().numpy() > 0.5).astype(np.int32)
            np.fill_diagonal(adj_recon_np, 0)
            
            G_recon = nx.from_numpy_array(adj_recon_np)
            
            for i in range(n_real):
                G_recon.nodes[i]['posx'] = float(pos_pred[i, 0])
                G_recon.nodes[i]['posy'] = float(pos_pred[i, 1])
                G_recon.nodes[i]['size_x'] = float(size_pred[i, 0])
                G_recon.nodes[i]['size_y'] = float(size_pred[i, 1])
                G_recon.nodes[i]['type_id'] = int(type_pred[i])
                G_recon.nodes[i]['type'] = TYPE_ID_TO_NAME.get(int(type_pred[i]), None)
                G_recon.nodes[i]['type_conf'] = float(type_prob[i, type_pred[i]])
            
            output_path = os.path.join(output_dir, file_names[idx])
            with open(output_path, 'wb') as f:
                pkl.dump(G_recon, f)
            
            # ground truth从原始特征提取（不是mask后的）
            x_orig_np = x_orig.cpu().numpy()
            gt_pos = x_orig_np[:n_real, 0:2]  # arch1的位置
            gt_size = x_orig_np[:n_real, 2:4]  # arch1的尺寸
            gt_type = np.argmax(x_orig_np[:n_real, 4:4+NUM_TYPES], axis=1)  # arch1的类型（从原始特征）
            
            pos_error = np.mean(np.abs(pos_pred - gt_pos))
            size_error = np.mean(np.abs(size_pred - gt_size))
            type_acc = np.mean(type_pred == gt_type)
            
            # 检查是否所有type都完全一致
            if np.all(type_pred == gt_type):
                perfect_match_count += 1
            
            total_pos_error += pos_error * n_real
            total_size_error += size_error * n_real
            total_type_acc += type_acc * n_real
            total_nodes += n_real
            total_graphs += 1
    
    print(f"\n{dataset_name}重建完成！")
    print(f"平均位置误差: {total_pos_error / total_nodes:.4f}")
    print(f"平均尺寸误差: {total_size_error / total_nodes:.4f}")
    print(f"平均类型准确率: {total_type_acc / total_nodes:.4f}")
    
    # 诊断信息：检查type是否异常一致
    perfect_match_ratio = perfect_match_count / total_graphs if total_graphs > 0 else 0
    print(f"\n⚠️  Type一致性诊断:")
    print(f"  完全匹配的图: {perfect_match_count}/{total_graphs} ({perfect_match_ratio*100:.1f}%)")
    
    if perfect_match_ratio > 0.95:  # 超过95%完全匹配
        print(f"  ❌ 警告: Type预测异常一致！")
        print(f"     可能原因:")
        print(f"     1. 模型直接复制输入的type特征，未真正学习")
        print(f"     2. Type损失权重过低 (当前w_type={args.w_type})")
        print(f"     3. 输入特征中包含了过多type信息")
        print(f"  建议:")
        print(f"     - 增加type损失权重 (如 --w_type 1.0)")
        print(f"     - 检查模型是否过拟合")
        print(f"     - 考虑在输入时mask掉type特征")
    elif perfect_match_ratio > 0.7:
        print(f"  ⚠️  注意: Type预测一致性较高")
        print(f"     建议增加type损失权重以提高学习效果")
    else:
        print(f"  ✓ Type预测正常，模型在真正学习type特征")
    
    return {
        'pos_error': total_pos_error / total_nodes,
        'size_error': total_size_error / total_nodes,
        'type_acc': total_type_acc / total_nodes,
        'perfect_match_ratio': perfect_match_ratio
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    print(f"模型类型: {args.model_type}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    
    # 加载训练集（根据消融实验配置）
    adj_list, feat_list, file_names = load_triple_dataset(
        args.arch1_dir, args.arch2_dir, args.bound_dir,
        use_arch2=args.use_arch2, use_bound=args.use_bound
    )
    feat_dim = int(feat_list[0].shape[1]) if len(feat_list) > 0 else 18
    num_graphs = len(adj_list)
    max_nodes = max(a.shape[0] for a in adj_list)
    
    # 确定输入配置描述
    input_config_desc = "arch1[18]"
    if args.use_arch2:
        input_config_desc += " + arch2[18]"
    if args.use_bound:
        input_config_desc += " + bound[18]"
    
    print(f"数据: {num_graphs}个图, 最大节点数: {max_nodes}")
    print(f"特征维度: {feat_dim} ({input_config_desc})")
    
    data_dict = {
        'adj_list': adj_list,
        'feat_list': feat_list,
        'max_nodes': max_nodes,
        'num_graphs': num_graphs
    }
    
    # 选择模型
    if args.model_type == 'separated':
        model = SeparatedGCNModelVAE(
            num_features=feat_dim,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            hidden3=args.hidden3,
            dropout=args.dropout
        ).to(device)
    else:  # hybrid
        model = HybridGCNModelVAE(
            num_features=feat_dim,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            hidden3=args.hidden3,
            dropout=args.dropout
        ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}")
    
    model, optimizer = train_model(args, model, device, data_dict)
    
    save_path = os.path.join(args.save_dir, f'{args.model_type}_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args
    }, save_path)
    print(f"\n模型已保存: {save_path}")
    
    # 重建测试集（三图输入）
    print("\n" + "="*60)
    print("重建测试集")
    print("="*60)
    test_adj_list, test_feat_list, test_file_names = load_triple_dataset(
        args.test_arch1_dir, args.test_arch2_dir, args.test_bound_dir,
        use_arch2=args.use_arch2, use_bound=args.use_bound
    )
    test_data_dict = {
        'adj_list': test_adj_list,
        'feat_list': test_feat_list,
        'max_nodes': max_nodes,  # 使用训练集的max_nodes
        'num_graphs': len(test_adj_list)
    }
    test_metrics = reconstruct_dataset(args, model, device, test_data_dict, test_file_names,
                                       args.output_dir, "测试集")
    
    print("\n" + "="*60)
    print("训练与测试集重建完成！")
    print("="*60)
    print("✓ 使用分离式架构")
    
    # 根据配置显示输入信息
    input_desc = "arch1"
    if args.use_arch2:
        input_desc += " + arch2"
    if args.use_bound:
        input_desc += " + bound"
    print(f"✓ 输入配置: {input_desc}")
    
    if args.use_arch2 or args.use_bound:
        print("✓ 辅助图作为arch1的补充信息")
    
    feat_dim_desc = "18"
    if args.use_arch2:
        feat_dim_desc += "+18"
    if args.use_bound:
        feat_dim_desc += "+18"
    print(f"✓ 特征拼接: [N, {feat_dim}] = [N, {feat_dim_desc}]")
    
    print("✓ 属性直接从输入学习")
    print("✓ 避免GCN邻居聚合")
    print("✓ 最小化VAE影响")
    print("\n测试集重建指标:")
    print(f"  位置误差: {test_metrics['pos_error']:.4f}")
    print(f"  尺寸误差: {test_metrics['size_error']:.4f}")
    print(f"  类型准确率: {test_metrics['type_acc']:.4f}")


if __name__ == '__main__':
    main()