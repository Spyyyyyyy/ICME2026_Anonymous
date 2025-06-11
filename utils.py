import random
import time
import datetime
import sys
import collections

from torch.autograd import Variable
import torch
from visdom import Visdom
import numpy as np
import os

from torchvision.utils import save_image
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.utils as vutils

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

def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        # if isinstance(v, collections.MutableMapping):
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def set_seed(device, seed=0):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].data[0]
            else:
                self.losses[loss_name] += losses[loss_name].data[0]

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), 
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

        

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

def save_fake_image(fake_B, A_core_names, A_coords, save_dir):
    """
    fake_B: torch.Tensor, shape [N, C, H, W]
    A_core_names: list/tuple/array, 长度为N
    A_coords: list/tuple/array, shape [N, 2] 或 [(y1, x1), (y2, x2), ...]
    save_dir: str, 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    to_pil = T.ToPILImage()
    N = fake_B.shape[0]
    for idx in range(N):
        img = fake_B[idx].cpu().detach()
        img_pil = to_pil(img)
        core_name = A_core_names[idx]
        coords = A_coords[idx]
        save_path = os.path.join(save_dir, f"{core_name}_y_{coords[0]}_x_{coords[1]}.png")
        img_pil.save(save_path)
        print(f"已保存: {save_path}")

def save_stack_fake_image(src, dst, fake, save_path, png_id):
    stacked_images = torch.stack((src, fake, dst), dim=1)
    stacked_images = 0.5*(stacked_images + 1.0)
    mixed_images = stacked_images.view(-1, *src.shape[1:])
    mixed_images = F.interpolate(mixed_images, size=(256, 256), mode='bilinear', align_corners=False)
    grid = vutils.make_grid(mixed_images, nrow=6, padding=2, normalize=True, scale_each=True)
    save_image(grid, f"{save_path}/fake_{png_id}.png", normalize=True)
    print(f"save fake image to {save_path}/fake_{png_id}.png")
