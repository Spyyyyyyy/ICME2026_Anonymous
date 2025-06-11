#!/usr/bin/python3
import logging
import argparse
import itertools
import time
import os
import yaml

import torch
from torch.autograd import Variable
from torch.nn import DataParallel

from models.cyclegan_networks import Generator
from models.cyclegan_networks import Discriminator
from utils import ReplayBuffer, LambdaLR, weights_init_normal, flatten
from dataset import BaseDataset, prepare_image_patch_dataloader


def train(opt, device):
    if not os.path.exists(opt["save_path"]):
        os.makedirs(opt["save_path"])

    logging.basicConfig(
            filename=f'{opt["save_path"]}/training.log',
            filemode='w',  # 新建/覆盖日志文件
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 写入参数文件
    logging.info("========== Training Options ==========")
    for k, v in opt.items():
        logging.info(f"{k}: {v}")
    logging.info("======================================")
    
    ###### Definition of variables ######
    # Networks
    netG_A2B = Generator(opt['train.data.input_nc'], opt['train.data.output_nc']).to(device)
    netG_B2A = Generator(opt['train.data.output_nc'], opt['train.data.input_nc']).to(device)
    netD_A = Discriminator(opt['train.data.input_nc']).to(device)
    netD_B = Discriminator(opt['train.data.output_nc']).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        netG_A2B = DataParallel(netG_A2B)
        netG_B2A = DataParallel(netG_B2A)
        netD_A = DataParallel(netD_A)
        netD_B = DataParallel(netD_B)

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                    lr=opt['train.params.optimizer.lr.lr_G'], betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt['train.params.optimizer.lr.lr_D'], betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt['train.params.optimizer.lr.lr_D'], betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt['train.params.n_epochs'], opt['train.params.epoch'], opt['train.params.n_epochs_decay']).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt['train.params.n_epochs'], opt['train.params.epoch'], opt['train.params.n_epochs_decay']).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt['train.params.n_epochs'], opt['train.params.epoch'], opt['train.params.n_epochs_decay']).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    input_A = Tensor(opt['train.params.patch_batch_size'], opt['train.data.input_nc'], int(opt['train.data.patch_size']/opt['train.data.downsample']), int(opt['train.data.patch_size']/opt['train.data.downsample']))
    input_B = Tensor(opt['train.params.patch_batch_size'], opt['train.data.input_nc'], int(opt['train.data.patch_size']/opt['train.data.downsample']), int(opt['train.data.patch_size']/opt['train.data.downsample']))
    target_real = Variable(Tensor(opt['train.params.patch_batch_size']).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt['train.params.patch_batch_size']).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset loader
    dataset = BaseDataset(opt)
    dataloader = prepare_image_patch_dataloader(args=opt, dataset=dataset, shuffle=True)

    # Loss plot
    # logger = Logger(opt['n_epochs'], len(dataloader))
    ###################################

    best_loss_G = float('inf') 
    ###### Training ######
    for epoch in range(opt['train.params.epoch'], opt['train.params.n_epochs']):
        epoch_start_time = time.time()
        for i, batch in enumerate(dataloader):
            # Set model input
            real_A = Variable(input_A.copy_(batch[0])) # src
            real_B = Variable(input_B.copy_(batch[1])) # dst

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B)*5.0
            # G_B2A(A) should equal A if real A is fed
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)*5.0

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            
            optimizer_G.step()
            ###################################

            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)
            
            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()

            optimizer_D_B.step()

            ###################################

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            if i % opt["train.params.save.loss_logging_freq"] == 0:
                logging.info("[Epoch %d/%d] [Batch %d/%d] [Time: %.2f] [D loss: %f] [G loss: %f] [G identity loss: %f] [G GAN loss: %f] [G cycle loss: %f]"
                    % (epoch, opt['train.params.n_epochs'], i, len(dataloader), epoch_duration, (loss_D_A + loss_D_B), loss_G, (loss_identity_A + loss_identity_B), \
                    (loss_GAN_A2B + loss_GAN_B2A), (loss_cycle_ABA + loss_cycle_BAB))
                    )
                print("[Epoch %d/%d] [Batch %d/%d] [Time: %.2f] [D loss: %f] [G loss: %f] [G identity loss: %f] [G GAN loss: %f] [G cycle loss: %f]"
                    % (epoch, opt['train.params.n_epochs'], i, len(dataloader), epoch_duration, (loss_D_A + loss_D_B), loss_G, (loss_identity_A + loss_identity_B), \
                    (loss_GAN_A2B + loss_GAN_B2A), (loss_cycle_ABA + loss_cycle_BAB))
                    )

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()


        if loss_G.item() <= best_loss_G and epoch >= 10:
            best_loss_G = loss_G.item()
            state = {
                'netG_A2B': netG_A2B.state_dict(),
                'netG_B2A': netG_B2A.state_dict(),
                'netD_A': netD_A.state_dict(),
                'netD_B': netD_B.state_dict(),
                'best_loss_G': best_loss_G,
                'epoch': epoch,
                'batch': i
            }
            torch.save(state, f'{opt["save_path"]}/model/best_model.pth')
            logging.info(f"Best model saved with G loss: {best_loss_G} at epoch {epoch}")
            print(f"Best model saved with G loss: {best_loss_G} at epoch {epoch}")

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