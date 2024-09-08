# intra-experiments
python main.py --dataset Zheng\ 68K --data_dir /home/xyfan/data/Datasets/baseline_datasets \
        --train_prop 0.6 --valid_prop 0.2 \
        --use_graph --use_knn --use_HVG \
        --epochs 20 --batch_size 256 \
        --runs 1 --device 0 --save_result

# cross-platforms 
# python main.py --dataset 10Xv3 --data_dir /home/xyfan/data/Datasets/cross_platforms \
# 	--cross_platform --query_dataset 10Xv2 \
#       --train_prop 0.8 --valid_prop 0.2 \
#       --use_HVG --use_graph --use_knn \
#       --epoch 20 --batch_size 256 \
#       --runs 1 --device 0 --save_result