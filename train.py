#!/usr/bin/python3
import argparse
import yaml
import torch
import logging
import time
import os
import torch

import dataset
import dataset_cond
from models.cut_models import CUTModel
from models.cpt_models import CPTModel
from models.concut_models import ConcutModel
from models.cond_cpt_models import cond_CPTModel
from utils import flatten


def train(opt, device):
    if not os.path.exists(opt["save_path"]):
        os.makedirs(opt["save_path"])

    logging.basicConfig(
            filename=f'{opt["save_path"]}/training.log',
            filemode='w',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
    )

    ###### 打印参数 ######
    logging.info("========== Training Options ==========")
    for k, v in opt.items():
        logging.info(f"{k}: {v}")
    logging.info("======================================")
    
    ###### 模型定义 ######
    if opt['gan_type'] == 'cut':
        model = CUTModel(opt, device)
    elif opt['gan_type'] == 'cpt':
        model = CPTModel(opt, device)
    elif opt['gan_type'] == 'concut':
        model = ConcutModel(opt, device)
    elif opt['gan_type'] == 'cond_cpt':
        model = cond_CPTModel(opt, device)
    else:
        raise ValueError(f"Invalid GAN type: {opt['gan_type']}")
    
    ###### 数据集定义 ######
    if opt['gan_type'] == 'cond_cpt':   
        dataset = dataset_cond.BaseDataset(opt)
        dataloader = dataset_cond.prepare_image_patch_dataloader(args=opt, dataset=dataset, shuffle=True)
    else:
        dataset = dataset.BaseDataset(opt)
        dataloader = dataset.prepare_image_patch_dataloader(args=opt, dataset=dataset, shuffle=True)

    # Dataloader : src_patches, src_core_names, src_coords, dst_patches, dst_core_names, dst_coords
    
    ###### Training ######
    for epoch in range(opt['train.params.epoch'], opt['train.params.n_epochs']):
        epoch_start_time = time.time()
        for i, batch in enumerate(dataloader):

            if len(opt['gpu_ids']) > 0:
                torch.cuda.synchronize()

            if epoch == 0 and i == 0:
                model.data_dependent_initialize(batch)
                model.setup()               # regular setup: load and print networks; create schedulers
                model.parallelize()

            if opt['gan_type'] == 'cond_cpt':
                model_input = {'src': batch[0], 'dst': batch[3], 'cond': batch[6], 'current_epoch': epoch}
            else:
                model_input = {'src': batch[0], 'dst': batch[3], 'current_epoch': epoch}

            model.set_input(model_input)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if len(opt['gpu_ids']) > 0:
                torch.cuda.synchronize()

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            if i % opt["train.params.save.loss_logging_freq"] == 0:
                losses = model.get_current_losses()
                epoch_str = f"[Epoch {epoch}/{opt['train.params.n_epochs']}] [Batch {i}/{len(dataloader)}] [Time: {epoch_duration:.2f}]"
                loss_str = " ".join([f"[{k}: {v:.4f}]" for k, v in losses.items()])
                log_str = f"{epoch_str} {loss_str}"
                logging.info(log_str)
                print(log_str)

        model.update_learning_rate()
        model.save_networks(epoch)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="/data1/sunpengyu/Task_VirtualStain/Code/GAN/i2i_config.yml")
    parser.add_argument('--data_path', type=str, default="/data2/sunpengyu/Task_VirtualStain/Data/szph_preprocessed_v1_reg_patch_1024-4", help='path of data')
    parser.add_argument('--split_csv_path', type=str, default="/data2/sunpengyu/Task_VirtualStain/Data/szph_preprocessed_v1_reg_patch_1024/splits.csv", help='path of data split csv')
    parser.add_argument('--save_path', type=str, default="/data1/sunpengyu/Task_VirtualStain/Result/CUT_test", help='save path')
    parser.add_argument("--src_marker", type=str, help="source marker", default='HE')
    parser.add_argument("--dst_marker", type=str, help="target marker", default='PR')
    parser.add_argument("--mode", type=str, default='train', help='train / test')
    parser.add_argument("--gan_type", type=str, default='cut', help='cut / cpt / concut / cond_cpt')
    parser.add_argument('--cuda', type=eval, default=True)
    parser.add_argument('--gpu_ids', type=int, default=[0,1,2,3,4,5,6,7], nargs='+')
    parser.add_argument('--dst_encoder', type=str, default=None, help='Path of dst encoder checkpoint.')

    opt = parser.parse_args()
    opt = vars(opt)
    with open(opt['config_path']) as f:
        config = yaml.safe_load(f)
    opt.update(config)
    opt = flatten(opt)

    if opt['gan_type'] == 'concut' and opt['mode'] == 'train' and not opt['dst_encoder']:
        raise ValueError("ckpt path of dst encoder is required for concut.")

    device = f'cuda:{opt["gpu_ids"][0]}' if torch.cuda.is_available() and opt['cuda'] else 'cpu'

    train(opt, device)

