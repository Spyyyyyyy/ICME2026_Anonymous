#!/usr/bin/python3

import argparse
import yaml
import torch
import os
import torch

from dataset import BaseDataset, prepare_image_patch_dataloader
from models.cut_models import CUTModel
from models.cond_cpt_models import cond_CPTModel
from utils import load_state_dict_strip_module, save_fake_image, save_stack_fake_image, flatten

def test(opt, device):
    print("device : ", device)

    ###### Definition of variables ######
    save_path_A2B = os.path.join(opt['save_path'], f'prediction/{opt["src_marker"]}2{opt["dst_marker"]}')
    if not os.path.exists(save_path_A2B):
        os.makedirs(save_path_A2B)

    if opt['gan_type'] == 'cond_cpt':
        model = cond_CPTModel(opt, device)
    else: # 其余模型均未改变生成器架构
        model = CUTModel(opt, device)

    netG = model.netG
    state_dict_netG_B2A =load_state_dict_strip_module(os.path.join(opt['save_path'], 'best_net_G.pth'))
    netG.load_state_dict(state_dict_netG_B2A)

    # Set model's test mode
    netG.eval()

    # Dataset loader
    dataset = BaseDataset(opt)
    dataloader = prepare_image_patch_dataloader(args=opt, dataset=dataset, shuffle=False)

    ###### Testing######
    with torch.no_grad():

        if opt['test.save_stack']:
            dataloader_iter = iter(dataloader)

            for i in range(opt['test.n_img']):
                batch = next(dataloader_iter)

                real_A = batch[0].to(device)
                real_B = batch[3].to(device)

                if opt['gan_type'] == 'cond_cpt':
                    cond = batch[6].to(device)
                    real = torch.cat((real_A, cond), dim=1)
                    fake_B = netG(real)
                else:
                    fake_B = netG(real_A)

                save_stack_fake_image(real_A, real_B, fake_B, save_path_A2B, i)
        else:
            for i, batch in enumerate(dataloader):

                real_A = batch[0].to(device)
                real_B = batch[3].to(device)

                if opt['gan_type'] == 'cond_cpt':
                    cond = batch[6].to(device)
                    real = torch.cat((real_A, cond), dim=1)
                    fake_B = netG(real)
                else:
                    fake_B = netG(real_A)

                A_core_name = batch[1]
                A_coords = batch[2]

                # 创建patch保存路径
                A_core_name_unique = list(dict.fromkeys(A_core_name))
                if len(A_core_name_unique) == 1:
                    save_path = os.path.join(save_path_A2B, A_core_name_unique[0])
                else:
                    save_path = os.path.join(save_path_A2B, A_core_name_unique[1])
                    
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_fake_image(fake_B, A_core_name, A_coords, save_path)


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

    test(opt, device)
