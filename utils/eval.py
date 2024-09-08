import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, result=None):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if result is not None:
        out = result
    else:
        model.eval()
        if args.use_knn:
            out = model(dataset.graph['node_feat'].to(device), dataset.graph['edge_index'].to(device))
        else:
            out = model(dataset.graph['node_feat'].to(device), dataset.graph['edge_index'])

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])

    out = F.log_softmax(out, dim=1)
    valid_loss = criterion(
        out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out

@torch.no_grad()
def evaluate_cpu(model, dataset, split_idx, eval_func, criterion, args, result=None):
    model.eval()
    model.to(torch.device("cpu"))
    dataset.label = dataset.label.to(torch.device("cpu"))
    adjs = dataset.graph['adjs'][0]
    out = model(dataset.graph['node_feat'], adjs)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])

    out = F.log_softmax(out, dim=1)
    valid_loss = criterion(
        out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out

@torch.no_grad()
def evaluate_batch(model, dataset, x_i, adjs_i, idx_i, train_mask_i, split_idx, eval_func, criterion, args):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    n = dataset.graph['num_nodes']
    y_i = dataset.label[idx_i]

    test_idx = split_idx['test']
    test_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[test_idx] = True
    test_mask_i = test_mask[idx_i]

    valid_idx = split_idx['valid']
    valid_mask = torch.zeros(n, dtype=torch.bool)
    valid_mask[valid_idx] = True
    valid_mask_i = valid_mask[idx_i]

    model.eval()
    out_i = model(x_i.to(device), adjs_i[0].to(device))

    train_acc = eval_func(
        y_i[train_mask_i], out_i[train_mask_i])
    valid_acc = eval_func(
        y_i[valid_mask_i], out_i[valid_mask_i])
    test_acc = eval_func(
        y_i[test_mask_i], out_i[test_mask_i])

    out_i = F.log_softmax(out_i, dim=1)
    valid_loss = criterion(
        out_i[valid_mask_i], y_i.squeeze(1)[valid_mask_i])

    return train_acc, valid_acc, test_acc, valid_loss, out_i

@torch.no_grad()
def evaluate_CP(model, query_dataset, eval_func, args):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if args.use_graph:
        query_out = model(query_dataset.graph['node_feat'].to(device), query_dataset.graph['edge_index'].to(device))
    else:
        query_out = model(query_dataset.graph['node_feat'].to(device), query_dataset.graph['edge_index']) # which is None
    query_out = F.log_softmax(query_out, dim=1)
    query_y = query_dataset.label.unsqueeze(1)

    acc = eval_func(query_y, query_out)

    return acc

@torch.no_grad()
def evaluate_CM(model, query_dataset, args):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if args.use_graph:
        query_out = model(query_dataset.graph['node_feat'].to(device), query_dataset.graph['edge_index'].to(device))
    else:
        query_out = model(query_dataset.graph['node_feat'].to(device), query_dataset.graph['edge_index']) # which is None
    
    query_out = F.log_softmax(query_out, dim=1)

    test_pred = query_out.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    if not args.cross_platform:
        test_true = query_dataset.label.squeeze(1).detach().cpu().numpy()
        
    else:
        print(test_true)
        test_true = query_dataset.label.squeeze(0).detach().cpu().numpy()
    CM = confusion_matrix(test_true, test_pred)
    

    return CM, test_pred, test_true
