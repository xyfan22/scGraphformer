import argparse
import os, random, gc
import numpy as np
from numpy import *
import torch
import pickle as pkl
import time

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops, subgraph

from utils.logger import Logger, save_result
from utils.dataset import load_dataset, get_split_indices
from utils.data_utils import eval_acc
from utils.eval import evaluate, evaluate_cpu, evaluate_CP, evaluate_CM
from utils.parse import parser_add_main_args

from utils.scGraphformer import *

import warnings
warnings.filterwarnings('ignore')
torch.cuda.empty_cache()

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
if args.cross_platform:
    if args.query_dataset:
        # loading query and reference dataset
        if args.query_dataset == args.dataset:
            raise ValueError("Reference and Query are the same")
        dataset = load_dataset(args.data_dir, args.dataset, args.use_HVG, args.use_knn, args.query_dataset)
        query_name = args.query_dataset
        split_idx_lst = [get_split_indices(dataset, train_prop=args.train_prop) 
                         for _ in range(args.runs)]
    else:
        raise ValueError("Query dataset needed if cross platform")
else:
    dataset = load_dataset(args.data_dir, args.dataset, args.use_HVG, args.use_knn)
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                    for _ in range(args.runs)]

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

dataset_name = args.dataset

n = dataset.graph['num_nodes']
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

if args.use_knn:
    e = dataset.graph['edge_index'].shape[1]
else:
    print("predefined KNN structure is not needed")
    e = 0
    
print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")
X = dataset.graph['node_feat']
true_label = dataset.label

if args.use_knn:
    adjs = []
    adj, _ = remove_self_loops(dataset.graph['edge_index'])
    adj, _ = add_self_loops(adj, num_nodes=n)
    adjs.append(adj)
    dataset.graph['adjs'] = adjs

### Load method ###
model = scGraphformer(d, c, num_layers=args.num_layers, alpha=args.alpha, 
                     dropout=args.dropout, num_heads=args.num_heads, use_bn=args.use_bn, 
                     use_residual=args.use_residual, use_weight=args.use_weight, use_graph=args.use_graph).to(device)

### Loss function and metric ###
criterion = nn.CrossEntropyLoss()
eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

### Training loop ###
best_emb,best_model=None,None
attention = []
TIME = []

for run in range(args.runs):
    start = time.time()
    split_idx = split_idx_lst[run]
    train_idx = split_idx['train']
   
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')
    num_batch = train_idx.size(0) // args.batch_size + 1

    for epoch in range(args.epochs):
        model.to(device)
        model.train()
        y_true = true_label.to(device)
        idx = torch.randperm(train_idx.size(0))

        for i in range(num_batch):
            idx_i = train_idx[idx[i*args.batch_size : (i+1)*args.batch_size]]
            x_i = X[idx_i].to(device)
            if args.use_knn:
                adjs_i = []
                
                edge_index_i, _ = subgraph(idx_i, adjs[0], num_nodes=n, relabel_nodes=True)
                adjs_i.append(edge_index_i.to(device))
                optimizer.zero_grad()
                out_i = model(x_i, adjs_i[0])
            else:

                optimizer.zero_grad()
                out_i = model(x_i, edge_index=None)

            out_i = F.log_softmax(out_i, dim=1)

            loss = criterion(
                out_i, y_true.squeeze(1)[idx_i])

            loss.backward()
            optimizer.step()
            
            print(f'Epoch: {epoch:02d}, Batch: {i:04d}, Loss: {loss:.4f}')

        if epoch % 1 == 0:
            if args.large_scale:
                result = evaluate_cpu(model, dataset, split_idx, eval_func, criterion, args)
            else:
                result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
        
            # result[:-1] obtain all other objects except the last one
            logger.add_result(run, result[:-1])

            if result[1] > best_val:
                best_val = result[1]

            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%')

    if args.get_attn:
        attn = model.get_attentions(dataset.graph['node_feat'].to(device), dataset.graph['edge_index'].to(device))
        attention.append(attn)

    logger.print_statistics(run)
    end = time.time()
    run_time = end-start
    TIME.append(run_time)
    print(f'Running time: {run_time: .2f}s')
results = logger.print_statistics()
if args.cross_platform:
    print(f"Ref->Query: {args.dataset}->{args.query_dataset}: {results.mean():.2f} ± {results.std():.2f}")
print(f'Running time: {np.mean(TIME): .2f}s ± {np.std(TIME): .2f}')

### get attention Gene-Gene interaction ###
if args.get_attn:
    attn_folder = "cache/attention"
    if args.use_HVG:
        attn_path = '{}/{}_N8_HVG_attn.pt'.format(attn_folder, dataset_name)
    else:
        attn_path = '{}/{}_N8_attn.pt'.format(attn_folder, dataset_name)
    if os.path.exists(attn_path):
        os.remove(attn_path)
    print('Saving attention pt!')
    torch.save(attention, attn_path)
       
### Save results ###
if args.save_result:
    save_result(args, results, TIME, args.cross_platform)

gc.collect()
torch.cuda.empty_cache()