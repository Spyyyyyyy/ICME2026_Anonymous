import torch
from torch import nn
from torch.nn import DataParallel
from torch.optim import lr_scheduler
import models.cut_networks as cut_networks
from models.neighborhood_objectives import GANLoss, PatchNCELoss, AdaptiveSupervisedPatchNCELoss
from utils import LambdaLR
from collections import OrderedDict
from models.gauss_pyramid import Gauss_Pyramid_Conv
import os
import logging

class CPTModel(nn.Module):
    def __init__(self, opt, device):
        super(CPTModel, self).__init__()

        self.save_dir = opt['save_path']
        self.lambda_NCE = opt['train.params.loss.gan_loss.lambda_NCE']
        self.nce_idt = opt['train.params.loss.gan_loss.nce_idt']
        self.lr_G = opt['train.params.optimizer.lr.lr_G']
        self.lr_D = opt['train.params.optimizer.lr.lr_D']
        self.beta1 = opt['train.params.optimizer.params.beta1']
        self.beta2 = opt['train.params.optimizer.params.beta2']
        self.nce_num_patches = opt['train.params.loss.gan_loss.nce_num_patches']
        self.lambda_gp = opt['train.params.loss.gan_loss.lambda_gp']
        self.lambda_asp = opt['train.params.loss.gan_loss.lambda_asp']
        self.nce_layers = [int(i) for i in opt['train.params.loss.gan_loss.nce_layers'].split(',')]
        self.epoch = opt['train.params.epoch']
        self.n_epochs = opt['train.params.n_epochs']
        self.decay_epoch = opt['train.params.n_epochs_decay']
        self.netF_type = opt['train.model.projector.netF']
        self.num_patches = opt['train.model.projector.num_patches']

        self.gpu_ids = opt['gpu_ids']
        self.device = device

        self.loss_names = ['G', 'G_GAN', 'D_real', 'D_fake', 'NCE_A']
        self.model_names = ['G', 'F', 'D']
        self.best_loss_G = float('inf')
        self.optimizers = []
        if self.nce_idt:
            self.loss_names.append('NCE_B')

        self.netG = cut_networks.define_G(args=opt)
        self.netF = cut_networks.define_F(args=opt) # netF用于提取netG不同层特征
        self.netD = cut_networks.define_D(args=opt)

        self.criterionGAN = GANLoss(opt['train.params.loss.gan_loss.gan_mode'], device=self.device)
        self.criterionIdt = torch.nn.L1Loss().to(self.device)
        self.criterionNCE = PatchNCELoss(opt, device=self.device)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=self.lr_G,
                                            betas=(self.beta1, self.beta2))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=self.lr_D,
                                            betas=(self.beta1, self.beta2))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        # 金字塔卷积损失
        if self.lambda_gp > 0:
            self.P = Gauss_Pyramid_Conv(num_high=5)
            self.criterionGP = torch.nn.L1Loss().to(self.device)
            # if self.opt.gp_weights == 'uniform':
            #     self.gp_weights = [1.0] * 6
            # else:
            #     self.gp_weights = eval(self.opt.gp_weights)
            self.gp_weights = [1.0] * 6
            self.loss_names += ['GP']

        if self.lambda_asp > 0:
            self.criterionASP = AdaptiveSupervisedPatchNCELoss(opt).to(self.device)
            self.loss_names += ['ASP']

    # 预定义 net_F
    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data[0].size(0) // max(len(self.gpu_ids), 1)
        model_input = {'src': data[0], 'dst': data[3], 'current_epoch': 0}
        self.set_input(model_input)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        self.compute_D_loss().backward()                  # calculate gradients for D
        self.compute_G_loss().backward()                   # calculate graidents for G
        if self.lambda_NCE > 0.0:
            self.optimizer_F = torch.optim.Adam(
                self.netF.parameters(),
                lr=self.lr_G,
                betas=(self.beta1, self.beta2))
            self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.netF_type == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.netF_type == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        self.real_A = input['src'].to(self.device)
        self.real_B = input['dst'].to(self.device)
        self.current_epoch = input['current_epoch']

    # 设置学习率调度器
    def setup(self):
        def get_scheduler(optim):
            scheduler = lr_scheduler.LambdaLR(optim,
                                              lr_lambda=LambdaLR(self.n_epochs, self.epoch, self.decay_epoch).step)
            return scheduler
        self.schedulers = [get_scheduler(optimizer) for optimizer in self.optimizers]
        print("Schedulers:", self.schedulers)
    

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        # Real
        self.loss_D_real = self.criterionGAN(self.netD(self.real_B), True)
        # Fake; stop backprop to the generator by detaching fake_B
        self.loss_D_fake = self.criterionGAN(self.netD(self.fake_B.detach()), False)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        # GAN loss: D(G(A))
        self.loss_G_GAN = self.criterionGAN(self.netD(self.fake_B), True)

        # NCE loss: NCE(A, G(A))
        if self.lambda_NCE > 0.0:
            self.loss_NCE_A = self.compute_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE_A = 0.0

        # NCE-IDT loss: NCE(B, G(B))
        if self.nce_idt and self.lambda_NCE > 0.0:
            self.loss_NCE_B = self.compute_NCE_loss(self.real_B, self.idt_B)
        else:
            self.loss_NCE_B = 0

        # NCE between the noisy pairs (fake_B and real_B)
        if self.lambda_asp > 0:
            self.loss_ASP = self.compute_NCE_loss(self.real_B, self.fake_B, paired=True)
        else:
            self.loss_ASP = 0.0

        loss_NCE = self.lambda_NCE * (self.loss_NCE_A + self.loss_NCE_B) * 0.5 + self.lambda_asp * self.loss_ASP

        # FDL: compute loss on Gaussian pyramids
        if self.lambda_gp > 0:
            p_fake_B = self.P(self.fake_B)
            p_real_B = self.P(self.real_B)
            loss_pyramid = [self.criterionGP(pf, pr) for pf, pr in zip(p_fake_B, p_real_B)]
            weights = self.gp_weights
            loss_pyramid = [l * w for l, w in zip(loss_pyramid, weights)]
            self.loss_GP = torch.mean(torch.stack(loss_pyramid)) * self.lambda_gp
        else:
            self.loss_GP = 0

        self.loss_G = self.loss_G_GAN + loss_NCE + self.loss_GP

        return self.loss_G
    
    def compute_NCE_loss(self, src, tgt, paired=False):
        feat_q, patch_ids_q = self.netG(tgt, num_patches=self.nce_num_patches, encode_only=True)
        feat_k, _ = self.netG(src, num_patches=self.nce_num_patches, encode_only=True, patch_ids=patch_ids_q)
        feat_k_pool = self.netF(feat_k)
        feat_q_pool= self.netF(feat_q)
        # feat_k_pool, sample_ids = self.netF(feat_k, self.num_patches, None)
        # feat_q_pool, _ = self.netF(feat_q, self.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            if paired:
                loss = self.criterionASP(f_q, f_k, self.current_epoch)
            else:
                loss = self.criterionNCE(f_q, f_k)
            # loss = self.criterionNCE(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / len(self.nce_layers)
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.nce_idt else self.real_A

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def set_requires_grad(self, model, requires_grad=True):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']

        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret
    
    def save_networks(self, epoch):
        if self.loss_G.item() <= self.best_loss_G:
            self.best_loss_G = self.loss_G.item()
            for name in self.model_names:
                if isinstance(name, str):
                    save_path = os.path.join(self.save_dir, f'best_net_{name}.pth')
                    net = getattr(self, 'net' + name)

                    if torch.cuda.is_available():
                        torch.save(net.cpu().state_dict(), save_path)
                        net.to(self.device)
                    else:
                        torch.save(net.cpu().state_dict(), save_path)

            logging.info(f"Best model saved with G loss: {self.best_loss_G} at epoch {epoch}")
            print(f"Best model saved with G loss: {self.best_loss_G} at epoch {epoch}")

    def parallelize(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                setattr(self, 'net' + name, torch.nn.DataParallel(net, self.gpu_ids))


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