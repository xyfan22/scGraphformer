# scGraphformer: A Graph Transformer Network for Single-Cell RNA Sequencing Analysis

![scGraphformer Overview](overview.png)

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Usage](#usage)
   - 4.1 [Intra-experiment Analysis](#intra-experiment-analysis)
   - 4.2 [Inter-experiment Analysis](#inter-experiment-analysis)
5. [Model Architecture](#model-architecture)
6. [Parameter Tuning](#parameter-tuning)
7. [Citation](#citation)

## 1. Introduction

scGraphformer, a cutting-edge approach that integrates the transformative capabilities of the Transformer model with the relational inductive biases of GNNs. This graph transformer network abandons the dependence on pre-defined graphs and instead derives a cellular interaction network directly from scRNA-seq data. By treating cells as nodes within a graph and iteratively refining connections, scGraphformer captures the full spectrum of cellular relationships, allowing for a more nuanced understanding of cell type.

## 2. Installation

### 2.1 Requirements
- Scanpy 
- torch 
- Numpy > 1.19
- Pandas > 1.2

### 2.2 Setup
1. Clone the repository:
```bash
git clone https://github.com/xyfan22/scGraphformer.git
```
2. Set up the conda environment:
```bash
conda create -n scGraphformer python==3.8
# enter the env
conda activate scGraphformer
```
3. Install PyTorch:
Visit https://pytorch.org/ and follow the instructions to install the correct version of PyTorch for your system. For example:
```bash
conda install pytorch==1.13.1 torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```
4. Install PyTorch Geometric and related packages,
replace cu116 with your CUDA version if different (e.g., cu117 for CUDA 11.7):
```bash
pip install torch-geometric==2.4.0
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.1+cu116.html

# If the above installation doesn't work, follow these steps:
# a. Check your PyTorch version and CUDA version:
import torch
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
# b. Visit the PyTorch Geometric wheels page: https://data.pyg.org/whl/
# c. Find the wheels that match your PyTorch and CUDA versions. For example, if you have PyTorch 1.13.1 and CUDA 11.6, look for files containing cp38 (for Python 3.8), torch_1.13.1, and cu116.
# d. Download the appropriate .whl files for torch-scatter, torch-sparse, torch-cluster, and torch-spline-conv.
# e. Install the downloaded wheels using pip:
pip install path/to/downloaded/torch_scatter-*.whl
pip install path/to/downloaded/torch_sparse-*.whl
pip install path/to/downloaded/torch_cluster-*.whl
pip install path/to/downloaded/torch_spline_conv-*.whl
```


5. Install required packages:
```bash
pip install -r requirements.txt
```

## 3. Data Preparation
We have uploaded all of datasets within this folder:

https://mycuhk-my.sharepoint.com/:f:/g/personal/1155187720_link_cuhk_edu_hk/EqVfLiFZDApEtel_fLOX_8gBCC83cvuz7o4UgZrAfEtFyw

The scGraphformer model asj
- anndata stored in h5ad file
- Data preprocessing steps
- How to handle different types of scRNA-seq data]

## 4. Usage
ruuning the run_cell.sh:
```bash
bash run_cell.sh
```
### 4.1 Intra-experiment Analysis
In intra-dataset experiments, we partitioned the dataset into training, validation, and testing subsets. The model will be trained on the training set then annotate the testing set.  

To run intra-experiment analysis:

```bash
python main.py --dataset Zheng\ 68K --data_dir {dataset_path} \
     --train_prop 0.6 --valid_prop 0.2 \
     --use_graph --use_knn --use_HVG \
     --epochs 20 --batch_size 256 --runs 5 --device 0 --save_result
```
### 4.2 Intra-experiment Analysis
For cross-platform (inter-dataset) experiments, we used the reference dataset for training and the query dataset solely for testing. While training, the reference dataset is split into training data and valid data (0.8/0.2). We retained only the common genes between datasets to ensure a consistent input feature space. The trained model will annotate the cell type labels of query dataset.

```bash
python main.py --dataset 10Xv3 --data_dir {dataset_path} \
	--cross_platform --query_dataset 10Xv2 \
        --train_prop 0.8 --valid_prop 0.2 \
        --use_HVG --use_graph --use_knn \
        --epoch 20 --batch_size 256 \
        --runs 1 --device 0 --save_result
```

### 5. Model Architecture
The scGraphformer architecture employs transformer-based graph neural networks to provide accurate and scalable annotations of cell types.
It stands out in its ability to dynamically construct an inter-cell relationship network through a refinement process that enhances the biological topological structure inherent in the cell graph. 
This innovative approach leads to the discovery of latent cellular connections, which are then harnessed to achieve precise cell type annotations. 
The scGraphformer framework is composed of two key components: a specially designed Transformer module and a cell network learning module. The Transformer module is adept at discerning latent interactions among genes, which in turn influence cellular connectivity. The cell network learning module is responsible for constructing a nuanced cell relationship network. Unlike conventional methods that typically depend on predefined graph structures, scGraphformer is distinctive in its ability to learn the cell graph’s structure directly from the raw scRNA-seq data, allowing for the continuous refinement of cell-to-cell connections and leading to more accurate cell type annotations. 


### 6. Parameter Tuning
scGraphformer offers various parameters to customize its behavior. Here are some key parameters you can adjust:
```bash
# Data and Experiment Settings:
--dataset: Specify the name of your dataset.
--data_dir: Set the path to your dataset directory.
--epochs: Define the number of training epochs (default: 20).
--runs: Set the number of distinct runs for the experiment (default: 1).
--batch_size: Adjust the mini-batch size for training (default: 300).
--train_prop: Set the proportion of data used for training (default: 0.6).
--valid_prop: Set the proportion of data used for validation (default: 0.2).
--large_scale: Use this flag for large-scale datasets to manage memory usage.
--cross_platform: Enable this for cross-platform analysis.
--query_dataset: Specify the query dataset for cross-platform analysis.
--use_HVG: Use this flag to adopt highly variable genes. 
--use_graph: Use a predefined graph with input.  (default: False). 
--use_knn: Add K-Nearest Neighbors as relational bias (default: False). 

# Model Architecture:
--num_layers: Set the number of layers for deep methods (default: 1).
--num_heads: Set the number of attention heads (default: 2).
--alpha: Adjust the weight for the residual link (default: 0.5).
--use_bn: Enable batch normalization (default: True).
--use_residual: Use residual connections for each GNN layer (default: True).
--use_graph: Enable positional embeddings (default: False).

# Training Parameters:
--lr: Set the learning rate (default: 0.0005).
--weight_decay: Adjust the weight decay for regularization (default: 0.05).
--dropout: Set the dropout rate (default: 0.3).

# others
--save_result: Enable this to save the results of the experiment.
```
For more parameters, refer utils/parse.py.

To adjust these parameters, simply add them to your command when running the script. For example:
```bash
python main.py --dataset YourDataset --data_dir /path/to/data --epochs 20 --batch_size 256 
```

### 7. Citation
If you find our codes useful, please consider citing our work: 
```
Fan, X., Liu, J., Yang, Y. et al. scGraphformer: unveiling cellular heterogeneity and interactions in scRNA-seq data using a scalable graph transformer network. Commun Biol 7, 1463 (2024). https://doi.org/10.1038/s42003-024-07154-w
```
