import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from typing import Dict, Any
from copy import deepcopy
import numpy as np
import functools
import torch.nn.functional as F

# Normalizer configuration
NORM_CFG = {
    'instance': {
        'type': 'in'
    }
}

###############################################################################
# Helper Functions
###############################################################################

def make_norm_layer(norm_cfg: Dict[str, Any], **kwargs: Any):
    """
    Create normalization layer based on given config and arguments.

    Args:
        norm_cfg (Dict[str, Any]): A dict of keyword arguments of normalization layer.
    Returns:
        nn.Module: A layer object.
    """
    norm_cfg = deepcopy(norm_cfg)
    norm_cfg.update(kwargs)
    norm_type = norm_cfg['type']
    del norm_cfg['type']

    if norm_type == 'in':
        return nn.InstanceNorm2d(**norm_cfg)
    else:
        raise ValueError(f'Unknown norm type: {norm_type}.')
    
class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler
    Parameters:
        optimizer       -- the optimizer of the network
        args            -- stores all the experiment information
    For 'linear', we keep the same learning rate for the first <args.n_epochs> epochs
    and linearly decay the rate to zero over the next <args.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    lr_policy = args['train.params.optimizer.lr.lr_policy']
    lr_decay_iters = args['train.params.optimizer.lr.lr_decay_iters']
    n_epochs_decay = args['train.params.n_epochs_decay']
    n_epochs = args['train.params.n_epochs']
    epoch_count = args['train.params.save.epoch_count']

    if lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain)
    return net

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>
    return net


def define_G(args):
    """Create a generator
    The generator has been initialized by <init_weights>. It uses RELU for non-linearity.
    """
    netG = args['train.model.generator.netG']
    init_type = args['train.model.init_type']
    init_gain = args['train.model.init_gain']
    gpu_ids = args['gpu_ids']
    if netG == 'resnet_9blocks':
        n_blocks = 9
    elif netG == 'resnet_6blocks':
        n_blocks = 6
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    # Unet架构
    net = Generator(
        input_nc=args['train.data.input_nc'],
        output_nc=args['train.data.output_nc'],
        ngf=args['train.model.generator.ngf'], # 网络基础通道数
        n_blocks=n_blocks,
        norm_type=args['train.model.generator.normG'],
        no_antialias=args['train.model.no_antialias'],
        no_antialias_up=args['train.model.no_antialias_up']
    )

    # return init_weights(net, init_type, init_gain)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(args):
    """Create a discriminator
    PatchGAN discriminator: n_layers_D=3
    The discriminator has been initialized by <init_weights>. It uses Leakly RELU for non-linearity.
    """
    netD = args['train.model.discriminator.netD']
    init_type = args['train.model.init_type']
    init_gain = args['train.model.init_gain']
    gpu_ids = args['gpu_ids']

    if netD == 'n_layers':
        net = Discriminator(
            # input_nc=args['train.data.input_nc'],
            input_nc=args['train.model.discriminator.input_nc'] if args['train.model.discriminator.input_nc'] else 3,
            ndf=args['train.model.discriminator.ndf'],
            n_layers=args['train.model.discriminator.n_layers_D'],
            norm_type=args['train.model.discriminator.normD'],
            no_antialias=args['train.model.no_antialias']
        )
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)

    # return init_weights(net, init_type, init_gain)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_F(args):
    netF_type = args['train.model.projector.netF']
    netF_nc = args['train.model.projector.netF_nc']
    init_type = args['train.model.init_type']
    init_gain = args['train.model.init_gain']
    gpu_ids = args['gpu_ids']

    if netF_type == 'mlp_sample':
        net = PatchSampleF(nc=netF_nc, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netF_type)

    # return init_weights(net, init_type, init_gain)
    return init_net(net, init_type, init_gain, gpu_ids)

###############################################################################
# Nuclei Density Map Estimator
###############################################################################

class UnetEstimator(nn.Module):
    def __init__(self, input_nc, output_nc, features=32):
        super(UnetEstimator, self).__init__()
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_nc, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(features, features*2, 3, padding=1),
            nn.BatchNorm2d(features*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features*2, features*2, 3, padding=1),
            nn.BatchNorm2d(features*2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*2, features*4, 3, padding=1),
            nn.BatchNorm2d(features*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(features*4, features*4, 3, padding=1),
            nn.BatchNorm2d(features*4),
            nn.ReLU(inplace=True)
        )
        # 解码器
        self.up2 = nn.ConvTranspose2d(features*4, features*2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(features*4, features*2, 3, padding=1),
            nn.BatchNorm2d(features*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features*2, features*2, 3, padding=1),
            nn.BatchNorm2d(features*2),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.ConvTranspose2d(features*2, features, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(features*2, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(features, output_nc, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, input_nc, H, W]
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        up2 = self.up2(bottleneck)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        out = self.final(dec1)
        return self.out_act(out)


def define_E(input_nc, output_nc, args):
    features = args.get('train.model.estimator.nef', 32)
    init_type = args['train.model.init_type']
    init_gain = args['train.model.init_gain']
    gpu_ids = args['gpu_ids']

    net = UnetEstimator(input_nc, output_nc, features)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################

class Downsample(nn.Module):
    def __init__(self, channels):
        super(Downsample, self).__init__()
        self.reflectionpad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.reflectionpad(x)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = [
            nn.ReplicationPad2d(1),
            nn.ConvTranspose2d(
                channels,
                channels,
                kernel_size=4,
                stride=2,
                padding=3,
            ),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x, dim=1):
        # norm = x.pow(self.power).sum(dim, keepdim=True).pow(1. / self.power)
        # out = x.div(norm + 1e-7)
        # FDL: To avoid sqrting 0s, which causes nans in grad
        norm = (x + 1e-7).pow(self.power).sum(dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class PatchSampleF(nn.Module):
    def __init__(self, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        # self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, features, power=2):
        if not self.mlp_init:
            self.create_mlp(features)

        return_features = []
        for feature_id, feature in enumerate(features):
            mlp = getattr(self, f"mlp_{feature_id}")
            feature = mlp(feature)
            norm = feature.pow(power).sum(1, keepdim=True).pow(1.0 / power)
            feature = feature.div(norm + 1e-7)
            return_features.append(feature)
        return return_features

    # def forward(self, feats, num_patches=64, patch_ids=None):
    #     return_ids = []
    #     return_feats = []
    #     if not self.mlp_init:
    #         self.create_mlp(feats)
    #     for feat_id, feat in enumerate(feats):
    #         # print(f"PatchSampleF: feat_id={feat_id}, feat.shape={feat.shape}")
    #         if num_patches > 0:
    #             if patch_ids is not None:
    #                 patch_id = patch_ids[feat_id]
    #             else:
    #                 patch_id = np.random.permutation(feat.shape[1])
    #                 patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
    #             patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
    #             x_sample = feat[patch_id, :]
    #         else:
    #             x_sample = feat
    #             patch_id = []
    #         mlp = getattr(self, 'mlp_%d' % feat_id)
    #         x_sample = mlp(x_sample)

    #         return_ids.append(patch_id)
    #         x_sample = self.l2norm(x_sample)

    #         return_feats.append(x_sample)
    #     return return_feats, return_ids


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, norm_cfg, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.norm_cfg = norm_cfg
        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
            make_norm_layer(norm_cfg, num_features=dim),
            nn.ReLU(False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
            make_norm_layer(norm_cfg, num_features=dim),
        )

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.model(x)  # add skip connections
        return out

    def forward_with_anchor(self, x, y_anchor, x_anchor, mode):
        assert self.norm_cfg['type'] == "kin"
        x_residual = x
        x = self.model[:2](x)
        x = self.model[2](
            x, y_anchor=y_anchor, x_anchor=x_anchor, mode=mode,
        )
        x = self.model[3:6](x)
        x = self.model[6](
            x, y_anchor=y_anchor, x_anchor=x_anchor, mode=mode,
        )
        return x_residual + x


class GeneratorBasicBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size=3,
        stride=1,
        padding=1,
        use_bias=False,
        do_upsample=False,
        no_antialias_up=False,
        do_downsample=False,
        norm_cfg=None,
    ):
        super().__init__()

        self.do_upsample = do_upsample
        self.do_downsample = do_downsample
        self.no_antialias_up = no_antialias_up
        self.norm_cfg = norm_cfg

        if self.do_upsample:
            if self.no_antialias_up:
                self.upsample = nn.ConvTranspose2d(in_features, out_features,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, output_padding=1, bias=use_bias)
            else:
                self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        if not self.no_antialias_up:
            self.conv = nn.Conv2d(
                in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias
            )

        self.norm = make_norm_layer(self.norm_cfg, num_features=out_features)
        self.relu = nn.ReLU(False)
        if self.do_downsample:
            self.downsample = Downsample(out_features)

    def forward_hook(self, x):
        if self.do_upsample:
            x = self.upsample(x)
        x_hook = self.conv(x)
        x = self.norm(x_hook)
        x = self.relu(x)
        if self.do_downsample:
            x = self.downsample(x)
        return x_hook, x

    def forward(self, x):
        if self.do_upsample:
            x = self.upsample(x)

        if not self.no_antialias_up:
            x = self.conv(x)

        x = self.norm(x)
        x = self.relu(x)

        if self.do_downsample:
            x = self.downsample(x)
        return x

    def forward_with_anchor(self, x, y_anchor, x_anchor, mode):
        assert self.norm_cfg['type'] == "kin"
        if self.do_upsample:
            x = self.upsample(x)

        if not self.no_antialias_up:
            x = self.conv(x)

        x = self.norm(
            x, y_anchor=y_anchor, x_anchor=x_anchor, mode=mode,
        )
        x = self.relu(x)

        if self.do_downsample:
            x = self.downsample(x)
        return x


class Generator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf, n_blocks, norm_type='instance', no_antialias=False, no_antialias_up=False):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            n_blocks (int)      -- the number of ResNet blocks
        """
        assert(n_blocks >= 0)
        super(Generator, self).__init__()

        self.norm_cfg = NORM_CFG[norm_type]
        use_bias = True if norm_type in ['instance'] else False

        # initial block
        self.reflectionpad = nn.ReflectionPad2d(3)
        self.block1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=7, bias=use_bias),
            make_norm_layer(self.norm_cfg, num_features=ngf),
            nn.ReLU(False),
        )

        # downsampling
        stride = 2 if no_antialias else 1
        do_downsample = False if no_antialias else True
        self.downsampleblock2 = GeneratorBasicBlock(
            ngf, ngf * 2, stride=stride, use_bias=use_bias,
            do_downsample=do_downsample, do_upsample=False, norm_cfg=self.norm_cfg
        )
        self.downsampleblock3 = GeneratorBasicBlock(
            ngf * 2, ngf * 4, stride=stride, use_bias=use_bias,
            do_downsample=do_downsample, do_upsample=False, norm_cfg=self.norm_cfg
        )

        # residual blocks
        self.resnetblocks4 = nn.Sequential(
            *[
                ResnetBlock(ngf * 4, norm_cfg=self.norm_cfg, use_bias=use_bias)
                for _ in range(n_blocks)
            ]
        )

        # upsampling
        stride = 2 if no_antialias_up else 1
        self.upsampleblock5 = GeneratorBasicBlock(
            ngf * 4, ngf * 2, stride=stride, use_bias=use_bias, no_antialias_up=no_antialias_up,
            do_upsample=True, do_downsample=False, norm_cfg=self.norm_cfg
        )
        self.upsampleblock6 = GeneratorBasicBlock(
            ngf * 2, ngf, stride=stride, use_bias=use_bias, no_antialias_up=no_antialias_up,
            do_upsample=True, do_downsample=False, norm_cfg=self.norm_cfg
        )

        # final block
        self.block7 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7),
            nn.Tanh()
        )

    def append_sample_feature(
            self,
            feature,
            return_ids,
            return_feats,
            mlp_id=0,
            num_patches=256,
            patch_ids=None,
    ):
        feature_reshape = feature.permute(0, 2, 3, 1).flatten(1, 2)  # B, F, C
        if patch_ids is not None:
            patch_id = patch_ids[mlp_id]
            if patch_id.device != feature.device:
                patch_id = patch_id.to(feature.device)
        else:
            patch_id = torch.randperm(feature_reshape.shape[1], device=feature.device)
            patch_id = patch_id[: int(min(num_patches, patch_id.shape[0]))]
        x_sample = feature_reshape[:, patch_id, :].flatten(0, 1)

        return_ids.append(patch_id)
        return_feats.append(x_sample)

    def forward_with_anchor(self, x, y_anchor, x_anchor, mode):
        assert self.norm_cfg['type'] == "kin"
        x = self.reflectionpad(x)
        x = self.block1[0](x)
        x = self.block1[1](
            x, y_anchor=y_anchor, x_anchor=x_anchor, mode=mode,
        )
        x = self.block1[2](x)
        x = self.downsampleblock2.forward_with_anchor(
            x, y_anchor=y_anchor, x_anchor=x_anchor, mode=mode,
        )
        x = self.downsampleblock3.forward_with_anchor(
            x, y_anchor=y_anchor, x_anchor=x_anchor, mode=mode,
        )
        for resnetblock in self.resnetblocks4:
            x = resnetblock.forward_with_anchor(
                x, y_anchor=y_anchor, x_anchor=x_anchor, mode=mode,
            )
        x = self.upsampleblock5.forward_with_anchor(
            x, y_anchor=y_anchor, x_anchor=x_anchor, mode=mode,
        )
        x = self.upsampleblock6.forward_with_anchor(
            x, y_anchor=y_anchor, x_anchor=x_anchor, mode=mode,
        )
        x = self.block7(x)
        return x

    # def print_memory_usage(self, message=""):
    #     """
    #     Prints the current memory usage of the GPU.
    #     """
    #     allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    #     reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
    #     print(f"{message}: Allocated {allocated:.2f} MB, Reserved {reserved:.2f} MB")

    def forward(self, x, encode_only=False, num_patches=256, patch_ids=None, spatial_shape=False):
        if not encode_only:
            x = self.reflectionpad(x)
            x = self.block1(x)
            x = self.downsampleblock2(x)
            x = self.downsampleblock3(x)
            x = self.resnetblocks4(x)
            x = self.upsampleblock5(x)
            x = self.upsampleblock6(x)
            x = self.block7(x)
            # self.print_memory_usage("After forward pass")  # Check memory usage after the forward pass
            return x
        else:
            return_ids = []
            return_feats = []
            return_shapes = []  # <--- 新增
            mlp_id = 0

            # 0
            x = self.reflectionpad(x)
            # 不在RGB通道提取特征
            # return_shapes.append(x.shape[2:])  # <--- 新增: 记录 (H, W)
            # self.append_sample_feature(
            #     x,
            #     return_ids,
            #     return_feats,
            #     mlp_id=mlp_id,
            #     num_patches=num_patches,
            #     patch_ids=patch_ids,
            # )
            # mlp_id += 1

            # 4
            x = self.block1(x)
            x_hook, x = self.downsampleblock2.forward_hook(x)
            return_shapes.append(x_hook.shape[2:])  # <--- 新增: 记录 (H, W)
            self.append_sample_feature(
                x_hook,
                return_ids,
                return_feats,
                mlp_id=mlp_id,
                num_patches=num_patches,
                patch_ids=patch_ids,
            )
            mlp_id += 1

            # 8
            x_hook, x = self.downsampleblock3.forward_hook(x)
            return_shapes.append(x_hook.shape[2:])  # <--- 新增: 记录 (H, W)
            self.append_sample_feature(
                x_hook,
                return_ids,
                return_feats,
                mlp_id=mlp_id,
                num_patches=num_patches,
                patch_ids=patch_ids,
            )
            mlp_id += 1

            # 12, 16
            for resnet_layer_id, resnet_layer in enumerate(self.resnetblocks4):
                x = resnet_layer(x)
                if resnet_layer_id in [0, 4]:
                    return_shapes.append(x.shape[2:])  # <--- 新增: 记录 (H, W)
                    self.append_sample_feature(
                        x,
                        return_ids,
                        return_feats,
                        mlp_id=mlp_id,
                        num_patches=num_patches,
                        patch_ids=patch_ids,
                    )
                    mlp_id += 1

            # 修改返回值，使其包含空间形状
            if not spatial_shape:
                return return_feats, return_ids
            else:
                return_shapes_tensor = torch.tensor(return_shapes, device=x.device, dtype=torch.long)
                return return_feats, return_ids, return_shapes_tensor


class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_type='instance', no_antialias=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_type      -- normalization layer type
        """
        super(Discriminator, self).__init__()

        self.norm_cfg = NORM_CFG[norm_type]
        use_bias = True if norm_type in ['instance', 'kin'] else False

        if no_antialias:
            sequence = [
                nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, False)
            ]
        else:
            sequence = [
                nn.Conv2d(input_nc, ndf, kernel_size=4, stride=1, padding=1),
                nn.LeakyReLU(0.2, False),
                Downsample(ndf)
            ]

        # gradually increase the number of filters
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if no_antialias:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                    make_norm_layer(self.norm_cfg, num_features=ndf * nf_mult),
                    nn.LeakyReLU(0.2, False)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias),
                    make_norm_layer(self.norm_cfg, num_features=ndf * nf_mult),
                    nn.LeakyReLU(0.2, False),
                    Downsample(ndf * nf_mult)
                ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias),
            make_norm_layer(self.norm_cfg, num_features=ndf * nf_mult),
            nn.LeakyReLU(0.2, False)
        ]

        # output 1 channel prediction map
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

