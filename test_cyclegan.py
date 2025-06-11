#!/usr/bin/python3

import argparse
import sys
import os


from torchvision.utils import save_image
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from dataset import BaseDataset, prepare_image_patch_dataloader
from models.cyclegan_networks import Generator

def test(opt, device):
    print("device : ", device)

    ###### Definition of variables ######
    save_path_A2B = os.path.join(opt['save_path'], f'prediction/{opt["src_marker"]}2{opt["dst_marker"]}')
    save_path_B2A = os.path.join(opt['save_path'], f'prediction/{opt["dst_marker"]}2{opt["src_marker"]}')
    if not os.path.exists(save_path_A2B):
        os.makedirs(save_path_A2B)
    if not os.path.exists(save_path_B2A):
        os.makedirs(save_path_B2A)

    # Networks
    netG_A2B = Generator(opt['train.data.input_nc'], opt['train.data.output_nc']).to(device)
    netG_B2A = Generator(opt['train.data.output_nc'], opt['train.data.input_nc']).to(device)

    state_dict_netG_A2B =load_state_dict_strip_module(os.path.join(opt['save_path'], 'netG_A2B.pth'))
    state_dict_netG_B2A =load_state_dict_strip_module(os.path.join(opt['save_path'], 'netG_B2A.pth'))
    # Load state dicts
    netG_A2B.load_state_dict(state_dict_netG_A2B)
    netG_B2A.load_state_dict(state_dict_netG_B2A)

    # Set model's test mode
    netG_A2B.eval()
    netG_B2A.eval()

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt['cuda'] else torch.Tensor
    input_A = Tensor(opt['test.patch_batch_size'], opt['train.data.input_nc'],
                      int(opt['train.data.patch_size']/opt['train.data.downsample']), int(opt['train.data.patch_size']/opt['train.data.downsample']))
    input_B = Tensor(opt['test.patch_batch_size'], opt['train.data.input_nc'],
                      int(opt['train.data.patch_size']/opt['train.data.downsample']), int(opt['train.data.patch_size']/opt['train.data.downsample']))

    # Dataset loader
    dataset = BaseDataset(opt)
    dataloader = prepare_image_patch_dataloader(args=opt, dataset=dataset, shuffle=False)

    ###### Testing ######

    # batch = next(iter(dataloader))
    dataloader_iter = iter(dataloader)
    for i in range(opt['test.n_img']):
        batch = next(dataloader_iter)
        # Set model input
        real_A = Variable(input_A.copy_(batch[0]))
        real_B = Variable(input_B.copy_(batch[1]))

        fake_B = netG_A2B(real_A).data
        fake_A = netG_B2A(real_B).data

        save_fake_image(real_A, real_B, fake_B, save_path_A2B, i)
        save_fake_image(real_B, real_A, fake_A, save_path_B2A, i)

    ###################################

def save_fake_image(src, dst, fake, save_path, png_id):
    stacked_images = torch.stack((src, fake, dst), dim=1)
    stacked_images = 0.5*(stacked_images + 1.0)
    mixed_images = stacked_images.view(-1, *src.shape[1:])
    mixed_images = F.interpolate(mixed_images, size=(256, 256), mode='bilinear', align_corners=False)
    grid = vutils.make_grid(mixed_images, nrow=6, padding=2, normalize=True, scale_each=True)
    save_image(grid, f"{save_path}/fake_{png_id}.png", normalize=True)
    print(f"save fake image to {save_path}/fake_{png_id}.png")

def load_state_dict_strip_module(pth_path):
    """
    加载pth文件，自动去除state_dict中参数名的'module.'前缀，返回处理后的state_dict。
    :param pth_path: pth文件路径
    :return: 处理后的state_dict
    """
    import torch
    from collections import OrderedDict
    state_dict = torch.load(pth_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # 去掉'module.'前缀
        new_state_dict[name] = v
    return new_state_dict

    