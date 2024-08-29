'''
Document for console argument options.

Updated 2022/3/24 By Aurora Yin
'''
import os


# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py

class Options():
    def __init__(self):
        pass

    def init(self, parser):
        ### global settings ###
        parser.add_argument('--mode', type=str, default ='denoising',  help='image restoration mode')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        parser.add_argument('--epoch', type=int, default=200, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=56, help='train dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=56, help='evaluation dataloader workers')
        parser.add_argument('--dataset', type=str, default='SIDD')
        parser.add_argument('--pretrain_weights', type=str, default='./log_deblur/AtUNet_logs/model_log/model_best.pth', help='path of pretrained_weights')
        parser.add_argument('--lr_init', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--gpu', type=str, default='0,1,2,3', help='GPUs')
        parser.add_argument('--arch', type=str, default ='AtUNet',  help='archtechture')

        ### args for saving ###
        # parser.add_argument('--save_dir', type=str, default='/home/ma-user/work/deNoTr/log', help='save dir')
        parser.add_argument('--save_images', action='store_true', default=False)
        parser.add_argument('--env', type=str, default='_', help='env')
        parser.add_argument('--checkpoint', type=int, default=10, help='checkpoint')

        ### args for AtUNet ###
        parser.add_argument('--norm_layer', type=str, default='nn.LayerNorm', help='normalize layer in transformer')
        parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
        parser.add_argument('--win_size', type=int, default=8, help='window size of self-attention')
        parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
        parser.add_argument('--token_mlp', type=str, default='FFN', help='ffn/FFN token mlp')
        parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')

        ### args for training ###
        parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
        parser.add_argument('--resume', action='store_true', default=False)
        parser.add_argument('--train_dir', type=str, default='../dataset/denoising/cbsd/train', help='dir of train data')
        #../dataset/denoising/sidd/train   ../dataset/derain/rain100L/train
        parser.add_argument('--val_dir', type=str, default='../dataset/denoising/cbsd/val', help='dir of val data')
        parser.add_argument('--loss', type=str, default='Charbonnier', help='Loss function, choose one: Charbonnier/MSE/SSIM')
        parser.add_argument("--local_rank", type=int, default=0)

        return parser