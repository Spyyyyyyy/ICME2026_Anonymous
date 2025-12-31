import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from collections import OrderedDict

import models.networks as cut_networks
from models.neighborhood_objectives import GANLoss, PatchNCELoss
from utils import LambdaLR
from models.gauss_pyramid import Gauss_Pyramid_Conv

# --- Loss Function Definitions ---

class OrthogonalLoss(nn.Module):
    """
    Soft Orthogonal Constraint: || W^T * W - I ||_F^2
    Used to ensure that learned prototypes or features remain distinct.
    """
    def __init__(self, device):
        super(OrthogonalLoss, self).__init__()
        self.device = device

    def forward(self, features):
        if isinstance(features, list):
            total_loss = sum(self._compute_ortho(x) for x in features)
            return total_loss / len(features)
        return self._compute_ortho(features)

    def _compute_ortho(self, x):
        # Normalize to unit sphere
        x = F.normalize(x, dim=1)
        n = x.size(0)
        # Compute Gram Matrix: G = x * x^T
        gram = torch.mm(x, x.t())
        eye = torch.eye(n, device=x.device)
        # Penalize non-diagonal elements to encourage orthogonality
        return torch.mean((gram - eye) ** 2)

class DispersionLoss(nn.Module):
    """
    Dispersion Loss based on von Mises-Fisher (vMF) kernel.
    Encourages features to be uniformly distributed on the hypersphere.
    """
    def __init__(self, opt, device):
        super(DispersionLoss, self).__init__()
        self.kappa = opt.get('train.params.dispersion.kappa', 5.0)
        self.device = device

    def forward(self, features):
        total_loss = 0.0
        for x in features:
            x = F.normalize(x, dim=1)
            sim_matrix = torch.mm(x, x.t())
            vmf_kernel = torch.exp(self.kappa * sim_matrix)
            
            # Mask out self-similarity
            mask = torch.eye(x.size(0), device=x.device).bool()
            vmf_kernel = vmf_kernel.masked_fill(mask, 0)
            
            u_size = x.size(0)
            if u_size > 1:
                row_mean = torch.sum(vmf_kernel, dim=1) / (u_size - 1 + 1e-6)
                total_loss += torch.mean(torch.log(row_mean + 1e-6))
            
        return total_loss / len(features)

# --- Main PID Model Implementation ---

class PIDModel(nn.Module):
    def __init__(self, opt, device):
        super(PIDModel, self).__init__()
        self.device = device
        self.save_dir = opt['save_path']
        self.gpu_ids = opt['gpu_ids']

        # Hyperparameters
        self.lambda_R = opt['train.params.loss.gan_loss.lambda_NCE'] 
        self.lambda_S = opt['train.params.loss.gan_loss.lambda_asp'] 
        self.T_R = opt.get('train.params.loss.gan_loss.T_R', 0.07) 
        self.T_S = opt.get('train.params.loss.gan_loss.T_S', 0.05) 
        self.capacity_factor = opt.get('train.params.loss.gan_loss.capacity_factor', 2.0)
        self.nce_idt = opt['train.params.loss.gan_loss.nce_idt']
        self.nce_num_patches = opt['train.params.loss.gan_loss.nce_num_patches']
        self.nce_layers = [int(i) for i in opt['train.params.loss.gan_loss.nce_layers'].split(',')]
        
        # Dispersion parameters
        self.lambda_dfl = opt.get('train.params.dispersion.lambda_dfl', 0.1)
        self.lambda_dpl = opt.get('train.params.dispersion.lambda_dpl', 0.1)
        self.dispersion_type = opt.get('train.params.dispersion.type', 'dispersion')

        # Prototypes parameters
        self.n_prototypes = opt.get('train.params.proto.n_prototypes', 10)
        self.proto_dim = opt.get('train.params.proto.proto_dim', 256)
        self.warm_up_epochs = opt.get('train.params.proto.warm_up_epochs', 5)

        # Network setup
        self.loss_names = ['G', 'G_GAN', 'D_real', 'D_fake', 'NCE_R']
        self.model_names = ['G', 'F_R', 'F_S', 'D']
        self.best_loss_G = float('inf')

        self.netG = cut_networks.define_G(args=opt)
        self.netF_R = cut_networks.define_F(args=opt) 
        self.netF_S = cut_networks.define_F(args=opt) 
        self.netD = cut_networks.define_D(args=opt)

        # Loss criteria
        self.criterionGAN = GANLoss(opt['train.params.loss.gan_loss.gan_mode'], device=self.device)
        self.criterionNCE = PatchNCELoss(opt, device=self.device)

        if opt.get('train.params.loss.gan_loss.lambda_gp', 0) > 0:
            self.lambda_gp = opt['train.params.loss.gan_loss.lambda_gp']
            self.P = Gauss_Pyramid_Conv(num_high=5)
            self.criterionGP = nn.L1Loss().to(self.device)
            self.gp_weights = [1.0] * 6
            self.loss_names.append('GP')

        if self.lambda_dfl > 0 or self.lambda_dpl > 0:
            if self.dispersion_type == 'orthogonal':
                self.criterionDispersion = OrthogonalLoss(device=self.device)
            else:
                self.criterionDispersion = DispersionLoss(opt, device=self.device)

        # Initialize Prototypes as learnable parameters
        self.prototypes = nn.ParameterList([
            nn.Parameter(F.normalize(torch.randn(self.n_prototypes, self.proto_dim, device=self.device), dim=1))
            for _ in range(len(self.nce_layers))
        ])

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            list(self.netG.parameters()) + list(self.prototypes.parameters()), 
            lr=opt['train.params.optimizer.lr.lr_G'], betas=(opt['train.params.optimizer.params.beta1'], 0.999)
        )
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt['train.params.optimizer.lr.lr_D'], betas=(0.5, 0.999))
        self.optimizers = [self.optimizer_G, self.optimizer_D]
        
        self.netF_type = opt['train.model.projector.netF']
        self.optimizer_F_R = None
        self.optimizer_F_S = None
        self.is_protos_initialized = False

    def _init_f_optimizers(self):
        """Initializes MLP projector optimizers only if they contain parameters."""
        if self.netF_type == 'mlp_sample' and len(list(self.netF_R.parameters())) > 0:
            print("Initializing projectors optimizers (netF_R, netF_S)...")
            self.optimizer_F_R = torch.optim.Adam(self.netF_R.parameters(), lr=2e-4, betas=(0.5, 0.999))
            self.optimizer_F_S = torch.optim.Adam(self.netF_S.parameters(), lr=2e-4, betas=(0.5, 0.999))
            self.optimizers.extend([self.optimizer_F_R, self.optimizer_F_S])

    def _check_and_init_prototypes(self):
        """Data-driven lazy initialization of prototypes using real features."""
        if not self.is_protos_initialized and self.netF_type == 'mlp_sample':
            with torch.no_grad():
                feat_real, _ = self.netG(self.real_B, num_patches=self.nce_num_patches, encode_only=True)
                z_real_list = self.netF_S(feat_real)
                for i, z_real in enumerate(z_real_list):
                    if z_real.size(0) >= self.n_prototypes:
                        perm = torch.randperm(z_real.size(0))[:self.n_prototypes]
                        self.prototypes[i].data.copy_(F.normalize(z_real[perm], dim=1))
            self.is_protos_initialized = True
            print("Prototypes seeded from data features.")

    def expert_choice_matching(self, sim_matrix):
        """
        Uncertainty-Gated Expert Choice Routing (PAMoE).
        Prototypes (Experts) actively choose the most confident and similar samples.
        """
        tokens_per_batch, n_experts = sim_matrix.shape
        
        with torch.no_grad():
            # Estimate Uncertainty via Shannon Entropy
            probs = F.softmax(sim_matrix / self.T_S, dim=1) 
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
            max_entropy = torch.log(torch.tensor(n_experts, dtype=torch.float32, device=probs.device))
            
            # Confidence gating: high entropy -> low confidence
            norm_entropy = entropy / (max_entropy + 1e-9)
            confidence = (1.0 - norm_entropy).pow(1.0)
            
        # Modulate similarity by confidence to filter out background noise
        weighted_sim = sim_matrix * confidence.unsqueeze(1)
        
        # Experts select Top-K tokens
        top_k = max(1, int(self.capacity_factor * tokens_per_batch / n_experts))
        _, topk_indices = torch.topk(weighted_sim, k=top_k, dim=0)
        
        mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
        mask.scatter_(0, topk_indices, True)
        return mask

    def compute_NCE_loss_S_Guided_Proto(self, fake, real, warm_up=False):
        """Calculates Prototype-Guided PatchNCE Loss."""
        feat_fake, ids = self.netG(fake, num_patches=self.nce_num_patches, encode_only=True)
        feat_real, _ = self.netG(real, num_patches=self.nce_num_patches, encode_only=True, patch_ids=ids)
        z_fake_list = self.netF_S(feat_fake)
        z_real_list = self.netF_S(feat_real)

        total_loss = 0.0
        for i, (z_fake, z_real) in enumerate(zip(z_fake_list, z_real_list)):
            curr_proto = self.prototypes[i].to(z_real.device)
            z_fake, z_real = F.normalize(z_fake, dim=1), F.normalize(z_real, dim=1)
            curr_proto_norm = F.normalize(curr_proto, dim=1)

            # Routing
            sim_real = torch.mm(z_real.detach(), curr_proto_norm.detach().t())
            valid_mask = self.expert_choice_matching(sim_real)
            
            # Real to Proto similarity
            sim_real_grad = torch.mm(z_real.detach(), curr_proto_norm.t())
            loss_real = -torch.sum(valid_mask * F.log_softmax(sim_real_grad / self.T_R, dim=1)) / (valid_mask.sum() + 1e-9)

            # Fake to Proto similarity (Generation constraint)
            if not warm_up:
                sim_fake = torch.mm(z_fake, curr_proto_norm.detach().t())
                loss_fake = -torch.sum(valid_mask * F.log_softmax(sim_fake / self.T_S, dim=1)) / (valid_mask.sum() + 1e-9)
            else:
                loss_fake = 0.0
            
            total_loss += (loss_real + loss_fake) * 0.5

        return total_loss / len(self.nce_layers), z_fake_list, z_real_list

    def optimize_parameters(self):
        self.forward()
        self._check_and_init_prototypes()
        if self.optimizer_F_R is None: self._init_f_optimizers()

        # Update Discriminator
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.compute_D_loss().backward()
        self.optimizer_D.step()

        # Update Generator and Projectors
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.optimizer_F_R:
            self.optimizer_F_R.zero_grad()
            self.optimizer_F_S.zero_grad()
            
        self.compute_G_loss().backward()
        self.optimizer_G.step()
        if self.optimizer_F_R:
            self.optimizer_F_R.step()
            self.optimizer_F_S.step()
        
        # Maintain prototype normalization
        with torch.no_grad():
            for proto in self.prototypes:
                proto.data = F.normalize(proto.data, dim=1)

    def forward(self):
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.nce_idt else self.real_A
        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def set_input(self, input):
        self.real_A = input['src'].to(self.device) 
        self.real_B = input['dst'].to(self.device) 
        self.current_epoch = input['current_epoch']

    def compute_D_loss(self):
        self.loss_D_real = self.criterionGAN(self.netD(self.real_B), True)
        self.loss_D_fake = self.criterionGAN(self.netD(self.fake_B.detach()), False)
        return (self.loss_D_fake + self.loss_D_real) * 0.5

    def compute_G_loss(self):
        self.loss_G_GAN = self.criterionGAN(self.netD(self.fake_B), True)
        self.loss_G = self.loss_G_GAN
        
        # Semantic Guided Loss
        self.loss_NCE_S, _, feat_S_real = self.compute_NCE_loss_S_Guided_Proto(
            self.fake_B, self.real_B, warm_up=(self.current_epoch < self.warm_up_epochs)
        )
        self.loss_G += self.loss_NCE_S * self.lambda_S

        # Feature Dispersion Losses
        if self.lambda_dfl > 0:
            self.loss_DFL = self.criterionDispersion(feat_S_real) * self.lambda_dfl
            self.loss_G += self.loss_DFL
        
        if self.lambda_dpl > 0:
            self.loss_DPL = self.criterionDispersion(list(self.prototypes)) * self.lambda_dpl
            self.loss_G += self.loss_DPL

        return self.loss_G

    def save_networks(self, epoch):
        # Best model saving
        if self.loss_G.item() < self.best_loss_G:
            self.best_loss_G = self.loss_G.item()
            for name in self.model_names:
                self._save_pth(name, f'best_net_{name}.pth')

        # Latest model saving
        for name in self.model_names:
            self._save_pth(name, f'latest_net_{name}.pth')

    def _save_pth(self, name, filename):
        save_path = os.path.join(self.save_dir, filename)
        net = getattr(self, 'net' + name)
        model_state = net.module.state_dict() if isinstance(net, DataParallel) else net.state_dict()
        torch.save({k: v.cpu() for k, v in model_state.items()}, save_path)

    def parallelize(self):
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            setattr(self, 'net' + name, DataParallel(net, self.gpu_ids))

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_current_losses(self):
        return OrderedDict({name: float(getattr(self, 'loss_' + name)) for name in self.loss_names})