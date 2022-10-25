from argparse import ArgumentParser


def parse_train_args():

    # General arguments
    parser = ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./test_run', help='Folder in which to save model and logs')
    parser.add_argument('--restart_dir', type=str, default='', help='Folder of previous training model from which to restart')
    parser.add_argument('--cache', type=str, default='data/QM9/cache', help='Folder from where to load/restore cached dataset')
    parser.add_argument('--data_dir', type=str, default='data/QM9/qm9/', help='Folder containing original conformers')
    parser.add_argument('--std_pickles', type=str, default='data/QM9/standardized_pickles/', help='Folder in which the pickle are put after standardisation/matching')
    parser.add_argument('--split_path', type=str, default='data/QM9/split.npy', help='Path of file defining the split')
    parser.add_argument('--dataset', type=str, default='qm9', help='drugs or qm9')
    parser.add_argument('--device', type=str, default='cuda', help='Whether GPU is available')

    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=250, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for preprocessing')
    parser.add_argument('--optimizer', type=str, default='adam', help='Adam optimiser only one supported')
    parser.add_argument('--scheduler', type=str, default='plateau', help='LR scehduler: plateau or none')
    parser.add_argument('--scheduler_patience', type=int, default=20, help='Patience of plateau scheduler')
    parser.add_argument('--sigma', type=float, default=50, help='Maximum sigma used for training')
    parser.add_argument('--limit_train_mols', type=int, default=100, help='Limit to the number of molecules in dataset, 0 uses them all')
    parser.add_argument('--boltzmann_weight', action='store_true', default=False, help='Whether to sample conformers based on B.w.')
    parser.add_argument('--eps', type=float, default=1e-3, help='Epsilon')

    # Feature arguments
    parser.add_argument('--in_node_features', type=int, default=44, help='Dimension of node features: 74 for drugs and xl, 44 for qm9')
    parser.add_argument('--in_edge_features', type=int, default=4, help='Dimension of edge feature (do not change)')
    parser.add_argument('--sigma_embed_dim', type=int, default=32, help='Dimension of sinusoidal embedding of sigma')
    
    # Model arguments
    parser.add_argument('--num_conv_layers', type=int, default=1, help='Number of GINEConv layers')
    parser.add_argument('--scale_by_sigma', action='store_true', default=True, help='Whether to normalise the score')
    parser.add_argument('--no_residual', action='store_true', default=False, help='If set, it removes residual connection')
    

    args = parser.parse_args()
    return args