from utils.scGraphformer import *


def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='Baron Human')
    parser.add_argument('--data_dir', type=str, default='/home/xyfan/data/Datasets/baseline_datasets')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.6,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.2,
                        help='validation label proportion')
    
    # cross-platforms
    parser.add_argument('--cross_platform', action='store_true', help='use random splits')
    parser.add_argument('--query_dataset', type=str, default='10Xv2')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    
    # scGraphformer model
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='number of heads for attention')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual link')
    parser.add_argument('--use_HVG', action='store_true', help='adopted HVG')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--use_residual', action='store_true', help='use residual link for each GNN layer')
    parser.add_argument('--use_graph', action='store_true', help='use pos emb')
    parser.add_argument('--use_knn', action='store_true', help='if add KNN as relational bias')
    parser.add_argument('--use_weight', action='store_true', help='use weight for GNN convolution')
    parser.add_argument('--large_scale', action='store_true', help='large scale para for evaluate on cpu')

    # training
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=300, help='mini batch training for scRNA-seq data')

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--save_result', action='store_true',
                        help='save result')
    parser.add_argument('--confusion_matrix', action='store_true', help='whether to save Confusion Matrix')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='../../model/')
    parser.add_argument('--get_attn', action='store_true', help='attention matrix')


    
    