import argparse
from utils import utils


def parse_encoder(parser, arg_str=None):
    enc_parser = parser.add_argument_group()
    # utils.parse_optimizer(parser)

    enc_parser.add_argument('--conv_type', type=str,
                            help='type of convolution')
    enc_parser.add_argument('--method_type', type=str,
                            help='type of embedding')
    enc_parser.add_argument('--batch_size', type=int,
                            help='Training batch size')
    enc_parser.add_argument('--n_layers', type=int,
                            help='Number of graph conv layers')
    enc_parser.add_argument('--hidden_dim', type=int,
                            help='Training hidden size')
    enc_parser.add_argument('--skip', type=str,
                            help='"all" or "last"')
    enc_parser.add_argument('--dropout', type=float,
                            help='Dropout rate')
    enc_parser.add_argument('--n_batches', type=int,
                            help='Number of training minibatches')
    enc_parser.add_argument('--margin', type=float,
                            help='margin for loss')
    enc_parser.add_argument('--dataset', type=str,
                            help='Dataset')
    enc_parser.add_argument('--test_set', type=str,
                            help='test set filename')
    enc_parser.add_argument('--eval_interval', type=int,
                            help='how often to eval during training')
    enc_parser.add_argument('--val_size', type=int,
                            help='validation set size')
    enc_parser.add_argument('--model_path', type=str,
                            help='path to save/load model')
    enc_parser.add_argument('--opt_scheduler', type=str,
                            help='scheduler name')
    enc_parser.add_argument('--node_anchored', action="store_true",
                            help='whether to use node anchoring in training')
    enc_parser.add_argument('--test', action="store_true")
    enc_parser.add_argument('--n_workers', type=int)
    enc_parser.add_argument('--tag', type=str,
                            help='tag to identify the run')
    


    enc_parser.add_argument('--repeat', type=int, default=1, help='number of training instances to repeat')

    enc_parser.set_defaults(conv_type='SAGE',
                            method_type='gnn',
                            dataset='scene',     # syn
                            n_layers=8,
                            batch_size=64,  # 64, batch 개수
                            # feature_dim = 1, # rpe = 3, f0 = 1 # 기존 데이터 사용 시
                            feature_dim = 4, # rpe = 3, f0 = 1 새로운 데이터(rpe )
                            #hidden_dim=64,
                            hidden_dim= 4,
                            skip="learnable",
                            dropout=0.0,
                            n_batches=10,  # 1000000, total 반복
                            opt='adam',     # opt_enc_parser
                            opt_scheduler='none',
                            opt_restart=10,
                            weight_decay=0.0,
                            lr=1e-3,
                            margin=0.1,
                            test_set='',
                            eval_interval=10,   # 1000, batch 반복횟수
                            n_workers=4,        # 4
                            # model_path="ckpt/final/rpe_only/rpe_v3_0511_newdatase_e100_e991.pt",
                            # model_path="ckpt/final/rpe_only/rpe_v3_0512_newdatase_1_e991.pt",
                            # model_path="ckpt/final/rpe_v3_imgId100_e1.pt",
                            model_path="ckpt/final/rpe_v3_ged_diff_all___e1000.pt",
                            # model_path="ckpt/final/rpe_v3_imgId100_ged_e1000.pt",
                            tag='',
                            val_size=64,         # 4096,
                            node_anchored=False, # True
                            num_walks = 4,
                            num_steps =3, 
                            use_weight = False,
                            norm = 'all',
                            )    

    # return enc_parser.parse_args(arg_str)