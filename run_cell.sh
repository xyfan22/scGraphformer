# intra-experiments
python main.py --dataset Zheng\ 68K --rand_split --data_dir /home/xyfan/data/Datasets/baseline_datasets \
        --train_prop 0.6 --valid_prop 0.2 \
        --lr 0.0005 --weight_decay 0.1 --num_layers 1 \
        --use_residual --use_bn --use_graph --use_weight --use_knn --use_HVG --alpha 0.5 \
        --epochs 5 --seed 123 --runs 5 --device 0 --save_result

# cross-platforms
# python main.py --dataset 10Xv3 --data_dir /home/xyfan/data/Datasets/cross_platforms\
#         --cross_platform --query_dataset 10Xv2 \
#         --rand_split --train_prop 0.8 --valid_prop 0.2 \
#         --lr 0.0005 --weight_decay 0.1 --num_layers 1 \
#         --use_residual --use_bn --use_weight \
#         --use_HVG --alpha 0.4 --use_graph --use_knn \
#         --epoch 5 --batch_size 512 --seed 123 --device 0 --runs 5 --save_result