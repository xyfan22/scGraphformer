import numpy as np
import os, gc
import torch
import pickle as pkl
import scanpy as sc 
import pandas as pd
from sklearn import preprocessing
from utils.data_utils import rand_train_test_idx
from sklearn.neighbors import kneighbors_graph

class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        
        Usage after construction: 
        
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        
        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, train_prop=.6, valid_prop=.20):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        test_prop: will be the left proportion
        """

        ignore_negative = True
        train_idx, valid_idx, test_idx = rand_train_test_idx(
            self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
        split_idx = {'train': train_idx,
                        'valid': valid_idx,
                        'test': test_idx}
            
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):  
        return '{}({})'.format(self.__class__.__name__, len(self))
    
def get_random_indices(data_range, num_indices):
    return np.random.choice(data_range, num_indices, replace=False)

def get_split_indices(merge_adata, train_prop):
    total_train_range = range(merge_adata.graph['ref_index'])
    total_test_range = range(merge_adata.graph['ref_index'], merge_adata.graph['num_nodes'])

    valid_prop = 1 - train_prop
    train_size = int(len(total_train_range) * train_prop)
    valid_size = int(len(total_train_range) * valid_prop)

    train_valid_indices = get_random_indices(total_train_range, train_size + valid_size)
    train_idx = torch.as_tensor(train_valid_indices[:train_size]).long()
    valid_idx = torch.as_tensor(train_valid_indices[train_size:]).long()

    test_idx = torch.as_tensor(np.array(total_test_range)).long()

    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

def filter_data(X, highly_genes=4000):
    # we adopt xxx highly-variale expressed genes as HVGs
    X = np.ceil(X).astype(int)
    adata = sc.AnnData(X, dtype=np.float32)
    sc.pp.filter_genes(adata, min_counts=3)
    sc.pp.filter_cells(adata, min_counts=1)
    
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=4, flavor='cell_ranger', min_disp=0.5,
                                n_top_genes=highly_genes, subset=True)
    genes_idx = np.array(adata.var_names.tolist()).astype(int)
    cells_idx = np.array(adata.obs_names.tolist()).astype(int)

    return genes_idx, cells_idx

def filter_adata(adata, highly_genes=4000):
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=4, flavor='cell_ranger', min_disp=0.5,
                                n_top_genes=highly_genes, subset=True)
    return adata

def construct_knn_graph(features, dataset_name, n_neighbors = 8, query_dataset = None):
    graph_path = 'cache/graph'
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
    if query_dataset:
        graph_path = os.path.join(graph_path,'crossplatforms')
        if not os.path.exists(graph_path):
            os.mkdir(graph_path)
        knn_graph_path = '{}/{}_{}_N{}.pt'.format(graph_path, dataset_name, query_dataset, n_neighbors)
    else:
        knn_graph_path = '{}/{}_N{}.pt'.format(graph_path, dataset_name, n_neighbors)

    if os.path.exists(knn_graph_path):
        if query_dataset:
            print(f'{dataset_name}->{query_dataset} graph path exists!')
        else:
            print(f'{dataset_name} graph path exists!')
        edge_index = torch.load(knn_graph_path)
        return edge_index
    else:
        if query_dataset:
            print(f'{dataset_name}->{query_dataset} graph path not exists!')
        else:
            print(f'{dataset_name} graph path not exists!')
        print('Making KNN predefined graph...')
        adj = kneighbors_graph(features, n_neighbors = n_neighbors, include_self=True)
        edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
        torch.save(edge_index, knn_graph_path)
        print('Graph Done!')
        return edge_index
    
def normalize_adata(adata, size_factors=True, normalize_input=True, logtrans_input=True):
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata, min_counts=0)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    
    if logtrans_input:
        sc.pp.log1p(adata)
    
    if normalize_input:
        sc.pp.scale(adata)
    
    return adata

def load_dataset(data_dir, dataname, use_HVG, use_knn, query_dataset=None):
    """ Loader for NCDataset 
        Returns NCDataset 
    """
    if query_dataset:
        merge_dataset = merge_CP_dataset(data_dir, dataname, query_dataset, use_knn=use_knn, use_HVG=use_HVG)

        return merge_dataset
    else:
        dataset = load_cell_dataset(data_dir, dataname, use_knn=use_knn, use_HVG=use_HVG)
    return dataset 

def load_cell_dataset(data_dir, dataname, use_knn=False, use_HVG=False):
    data_path = os.path.join(data_dir, dataname+'.h5ad')
    dataset = NCDataset('scRNA-seq_'+dataname)

    # single-cell rna datasets processing
    adata = sc.read_h5ad(data_path)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    X = adata.X.toarray()
    # cell type in adata
    if 'cell' in adata.obs:
        y = adata.obs['cell']
    if adata.n_vars<=3500:
        HVG = adata.n_vars
    elif adata.n_vars>3500 and adata.n_vars<=20000:
        HVG = 3500
    elif adata.n_vars>20000:
        HVG = 4000
    del adata
    gc.collect()

    # if HVG selection is needed
    if use_HVG:
        print("HVG is adopted.")
        genes_idx, cells_idx = filter_data(X, highly_genes = HVG)
        X = X[cells_idx][:, genes_idx]
        y = y[cells_idx]
        print('X shape (with HVG selecting): ', X.shape)
    else:
        print("HVG is not adopted.")
        print('X shape (without HVG selecting): ', X.shape)

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = torch.as_tensor(le.transform(y))

    features = torch.as_tensor(X)
    labels = y
    if use_knn:
        edge_index = construct_knn_graph(features, dataset_name=dataname, n_neighbors = 8)
    else:
        edge_index = None
    num_nodes = features.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': features,
                     'num_nodes': num_nodes}
    dataset.label = torch.LongTensor(labels)
    return dataset

def merge_CP_dataset(data_dir, ref_data, query_data, use_knn=False, use_HVG=False):
    ref_path = os.path.join(data_dir, ref_data+'.h5ad')
    query_path = os.path.join(data_dir, query_data+'.h5ad')
    if not os.path.exists(query_path):
        raise ValueError('Better to put query and reference into the same folder')
    merge_dataset =  NCDataset(f'scRNA-seq_{ref_data}_{query_data}')

    ref_adata = sc.read_h5ad(ref_path)
    sc.pp.normalize_total(ref_adata, target_sum=1e4)
    sc.pp.log1p(ref_adata)

    query_adata = sc.read_h5ad(query_path)
    sc.pp.normalize_total(query_adata, target_sum=1e4)
    sc.pp.log1p(query_adata)

    common_genes = ref_adata.var_names.intersection(query_adata.var_names)
    if len(common_genes) == 0:
        raise ValueError('Reference and Query have no common genes.')
    ref_adata = ref_adata[:, common_genes]
    query_adata = query_adata[:, common_genes]

    merge_adata = sc.AnnData.concatenate(ref_adata, query_adata)

    if use_knn: # for better constructing the graph
        merge_X = merge_adata.X.toarray()
        merge_features = torch.as_tensor(merge_X)
        KNN_cell_number_ref = merge_adata.n_obs
        merge_edge_index = construct_knn_graph(merge_features, dataset_name=ref_data, n_neighbors = 8, query_dataset=query_data)

    else:
        merge_edge_index = None

    if use_HVG:
        HVG = 3500
        merge_vars = merge_adata.n_vars
        if merge_vars < 3500:
            HVG = merge_vars
        merge_adata = filter_adata(merge_adata, HVG)
        if use_knn:
            if merge_adata.n_obs != KNN_cell_number_ref:
                raise ValueError('The reference KNN graph node number not equal to number after HVG selection')

    if 'cell' in ref_adata.obs:
        y = merge_adata.obs['cell']
        cell = 'cell'
    elif 'cell_type' in ref_adata.obs:
        y = merge_adata.obs['cell_type']
        cell = 'cell_type'
    else:
        raise ValueError('Please mention the cell obs in adata')

    merge_adata.obs['cell_raw'] = y
    # proprecessing lable one-hot encoding
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    merge_adata.obs['cell'] = y
    merge_X = merge_adata.X.toarray()
    if use_HVG:
        print('Merged X shape (with HVG selecting): ', merge_X.shape)

    else:
        print('Merged X shape (without HVG selecting): ', merge_X.shape)

    # constructing the NCD datasets
    merge_features = torch.as_tensor(merge_X).to(dtype=torch.float32)
    merge_labels = torch.as_tensor(merge_adata.obs['cell'])
    num_nodes = merge_features.shape[0]

    merge_dataset.graph = {'edge_index': merge_edge_index,
                     'edge_feat': None,
                     'node_feat': merge_features,
                     'num_nodes': num_nodes,
                     'ref_index': ref_adata.n_obs,
                     'query_index': query_adata.n_obs}
    merge_dataset.label = torch.LongTensor(merge_labels)

    return merge_dataset