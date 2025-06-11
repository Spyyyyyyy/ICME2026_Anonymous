import argparse


def process_args():

    parser = argparse.ArgumentParser(description='Configurations for TANGLE pretraining')
    
    #-----> model args 
    parser.add_argument('--config_path', type=str, default="/data1/sunpengyu/Task_VirtualStain/Code/GAN/i2i_config.yml")
    parser.add_argument('--i2i_patches_path', type=str, default="/data2/sunpengyu/Task_VirtualStain/Data/szph_preprocessed_v1_reg_patch_1024-4")
    parser.add_argument('--src_data_split_path', type=str, default="/data2/sunpengyu/Task_VirtualStain/Data/szph_preprocessed_v1_reg_patch_1024/splits.csv")
    parser.add_argument('--save_path', type=str, default="/data1/sunpengyu/Task_VirtualStain/Result/CUT_test")
    parser.add_argument("--src_marker", type=str, help="name of the source marker", default='HE')
    parser.add_argument("--dst_marker", type=str, help="name of the destination/target marker", default='PR')
    parser.add_argument("--is_train", type=eval, default=True)
    parser.add_argument("--is_test", type=eval, default=False)
    parser.add_argument('--cuda', type=eval, default=True, help='use GPU computation')
    parser.add_argument('--gpu_ids', type=int, default=[0,1,2,3,4,5,6,7], nargs='+', help='gpu ids')

    #----> training args
    parser.add_argument('--timm_model', type=str, default='swinv2_base_window16_256.ms_in1k', help='Vision Encoder Type.')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Tensor dtype. Defaults to bfloat16 for increased batch size.')
    parser.add_argument('--warmup', type=bool, default=True, help='If doing warmup.')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs.')
    parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs to train (default: 2)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--end_learning_rate', type=float, default=1e-8, help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1234, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--temperature', type=float, default=0.01, help='InfoNCE temperature.')
    parser.add_argument('--intra_modality_mode_wsi', type=str, default='reconstruct_masked_emb', help='Type of Intra loss. Options are: reconstruct_avg_emb, reconstruct_masked_emb.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--symmetric_cl', type=bool, default=True, help='If use symmetric contrastive objective.')
    parser.add_argument('--method', type=str, default='tangle', help='Train recipe. Options are: tangle, tanglerec, intra.')
    parser.add_argument('--num_workers', type=int, default=20, help='number of cpu workers')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay.')

    #---> model inference 
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path of checkpoint.')

    args = parser.parse_args()

    return args