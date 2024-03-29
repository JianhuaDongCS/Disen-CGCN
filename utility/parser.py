'''
Disen-CGCN
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument('--weights_path', nargs='?', default='output/weights-Tmall',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='Tmall/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')
    parser.add_argument('--dataset', nargs='?', default='buy',
                        help='Choose a dataset from {buy, cart, view}')
    parser.add_argument('--dataset1', nargs='?', default='buy',
                        help='Choose a dataset from {buy, cart, view}') #
    parser.add_argument('--dataset2', nargs='?', default='cart',
                        help='Choose a dataset from {buy, cart, view}') #
    parser.add_argument('--dataset3', nargs='?', default='view',
                        help='Choose a dataset from {buy, cart, view}') #                   
    parser.add_argument('--pretrain', type=int, default=1,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.') 
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')  
    parser.add_argument('--is_norm', type=int, default=1,
                    help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--meta_embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64,64,64]',
                        help='Output sizes of every layer') 
    parser.add_argument('--layer_size2', nargs='?', default='[64, 64, 64]',
                        help='Output sizes of every layer') #buy 
    parser.add_argument('--layer_size3', nargs='?', default='[64, 64, 64, 64]',
                        help='Output sizes of every layer') #cart
    parser.add_argument('--layer_size4', nargs='?', default='[64, 64]',
                        help='Output sizes of every layer') #view

    parser.add_argument('--n_factors', type=int, default=2,
                        help='Number of factors.')

    parser.add_argument('--cor_flag3', type=int, default=1,
                        help='Correlation matrix flag')


    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size.')
    parser.add_argument('--attention_enlarge_num', type=int, default=4,
                        help='Attention_enlarge_num.')


    parser.add_argument('--reg_W_l2', nargs='?', default='[1e-4]',
                        help='Regularizations.')
    parser.add_argument('--reg_W_PFT_l2', nargs='?', default='[1e-4]',
                        help='Regularizations.')
    parser.add_argument('--reg_c', nargs='?', default='[1e0]',
                        help='Regularizations.')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--regs', nargs='?', default='[1e-4]',
                        help='Regularizations.')


    parser.add_argument('--model_type', nargs='?', default='Disen-CGCN',
                        help='Specify the name of model (Disen-CGCN).')
    parser.add_argument('--adj_type', nargs='?', default='pre',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--alg_type', nargs='?', default='Disen-CGCN',
                        help='Specify the type of the graph convolutional layer from {ngcf, gcn, gcmc}.') 

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--node_dropout_flag', type=int, default=0,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[10,20,50]',
                        help='Top k(s) recommend')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver') 

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')

    return parser.parse_args()
