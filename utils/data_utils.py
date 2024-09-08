import os
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from scipy import sparse as sp
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, precision_score, \
    recall_score, balanced_accuracy_score, adjusted_rand_score, normalized_mutual_info_score

from torch_sparse import SparseTensor

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


def to_sparse_tensor(edge_index, edge_feat, num_nodes):
    """ converts the edge_index into SparseTensor
    """
    num_edges = edge_index.size(1)

    (row, col), N, E = edge_index, num_nodes, num_edges
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]

    value = edge_feat[perm]
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(N, N), is_sorted=True)

    # Pre-process some important attributes.
    adj_t.storage.rowptr()
    adj_t.storage.csr2csc()

    return adj_t


def eval_f1(y_true, y_pred):
    f1_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        true_column = y_true[is_labeled, i]
        pred_column = y_pred[is_labeled, i]
        
        if len(true_column) == 0:
            f1_list.append(float(0))
        else:
            f1 = f1_score(true_column, pred_column, average='weighted')
            f1_list.append(float(f1))

    return sum(f1_list) / len(f1_list)

def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        if len(correct) == 0:
            acc_list.append(float(0))
        else:
            acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list)/len(acc_list)

def eval_cohen_kappa(y_true, y_pred):
    """
    Compute Cohen's Kappa score for multi-class or multi-label classification.
    
    Args:
    y_true (torch.Tensor): True labels.
    y_pred (torch.Tensor): Predicted probabilities or logits.
    
    Returns:
    float: Average Cohen's Kappa score across all classes.
    """
    kappa_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        true_column = y_true[is_labeled, i]
        pred_column = y_pred[is_labeled, i]
        
        if len(true_column) == 0:
            kappa_list.append(float(0))
        else:
            kappa = cohen_kappa_score(true_column, pred_column)
            kappa_list.append(float(kappa))

    return sum(kappa_list) / len(kappa_list)

def eval_rocauc(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    if n_classes > 2:
        y_true_bin = label_binarize(y_true, classes=classes)
        if y_pred.shape[1] == n_classes:
            y_pred_bin = y_pred
        else:
            y_pred_bin = label_binarize(y_pred.argmax(axis=1), classes=classes)
        
        return roc_auc_score(y_true_bin, y_pred_bin, multi_class='ovr', average='macro')
    else:
        if y_pred.shape[1] == 2:
            return roc_auc_score(y_true, y_pred[:, 1])
        else:
            return roc_auc_score(y_true, y_pred)

def eval_balanced_accuracy(y_true, y_pred):
    """
    Compute Balanced Accuracy score for multi-class classification.
    
    Args:
    y_true (torch.Tensor): True labels.
    y_pred (torch.Tensor): Predicted probabilities or logits.
    
    Returns:
    float: Balanced Accuracy score.
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1).detach().cpu().numpy()

    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    return balanced_acc

def eval_adjusted_rand_index(y_true, y_pred):
    """
    Compute Adjusted Rand Index for clustering evaluation.
    
    Args:
    y_true (torch.Tensor): True labels.
    y_pred (torch.Tensor): Predicted probabilities or logits.
    
    Returns:
    float: Adjusted Rand Index score.
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    ari_list = []

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        
        if np.any(is_labeled):
            ari = adjusted_rand_score(y_true[is_labeled, i], y_pred[is_labeled, i])
            ari_list.append(ari)
        else:
            ari_list.append(0.0)
    return sum(ari_list) / len(ari_list) if len(ari_list) > 0 else 0.0
    
def eval_recall(y_true, y_pred):
    recall_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    recall = recall_score(y_true, y_pred, average='macro')
    return recall

def eval_precision(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    precision = precision_score(y_true, y_pred, average='macro')
    return precision


def eval_normalized_mutual_info(y_true, y_pred):
    """
    Compute Normalized Mutual Information score.
    
    Args:
    y_true (torch.Tensor): True labels.
    y_pred (torch.Tensor): Predicted probabilities or logits.
    
    Returns:
    float: Normalized Mutual Information score.
    """
    y_true = y_true.detach().cpu().numpy().flatten()
    y_pred = y_pred.argmax(dim=-1).detach().cpu().numpy().flatten()

    nmi_score = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    return nmi_score

