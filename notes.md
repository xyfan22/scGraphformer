# scGraphformer

## Introduction

scGraphformer, a cutting-edge approach that integrates the transformative capabilities of the Transformer model with the relational inductive biases of GNNs. This graph transformer network abandons the dependence on pre-defined graphs and instead derives a cellular interaction network directly from scRNA-seq data. By treating cells as nodes within a graph and iteratively refining connections, scGraphformer captures the full spectrum of cellular relationships, allowing for a more nuanced understanding of cell type. 



## Requirement

- Scanpy 
- Pytorch 
- Numpy > 1.19
- Pandas > 1.2


## Usage
### 1. Setting environmernt
Setting the conda environment first.
```Python
pip install -r requirements.txt
```

### 2. Running the code
for intra-experiment:
```bash
python main.py --dataset Zheng\ 68K --data_dir /home/xyfan/data/Datasets/baseline_datasets \
        -rand_split --train_prop 0.6 --valid_prop 0.2 \
        --lr 0.0005 --weight_decay 0.1 --num_layers 1 \
        --use_residual --use_bn --use_graph --use_weight --use_knn --use_HVG --alpha 0.5 \
        --epochs 256 --seed 123 --runs 0 --device 0 --save_result
```
for inter-experiments:
```bash
python main.py --dataset 10Xv3 --data_dir /home/xyfan/data/Datasets/cross_platforms\
				--cross_platform --query_dataset 10Xv2 \
        --rand_split --train_prop 0.8 --valid_prop 0.2 \
        --lr 0.0005 --weight_decay 0.1 --num_layers 1 \
        --use_residual --use_bn --use_weight \
        --use_HVG --alpha 0.4 --use_graph --use_knn --large_scale \
        --epoch 50 --batch_size 1000 --seed 123 --device 0 --runs 0 --save_result
```

## Citation
