import random
import time
import datetime
import sys
import collections
import os
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict

import torchvision.transforms as T
import torchvision.utils as vutils
from torchvision.utils import save_image
from visdom import Visdom

# --- State Dict & Config Utilities ---

def load_state_dict_strip_module(pth_path):
    """
    Loads a .pth file and automatically removes the 'module.' prefix 
    from state_dict keys (common when saving from DataParallel).
    """
    state_dict = torch.load(pth_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    return new_state_dict

def flatten(d, parent_key='', sep='.'):
    """
    Flattens a nested dictionary into a single-level dictionary 
    with dot-separated keys.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# --- Training Setup Utilities ---

def set_seed(device, seed=0):
    """
    Sets random seeds for reproducibility across Python, Numpy, and PyTorch.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def weights_init_normal(m):
    """
    Initializes network weights using a normal distribution.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class LambdaLR():
    """
    Learning rate scheduler following a linear decay pattern.
    """
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before training ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

# --- Visualization & Logging ---

def tensor2image(tensor):
    """
    Converts a PyTorch tensor (range -1 to 1) to a NumPy image (0 to 255).
    """
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)

class Logger():
    """
    Tracks training progress, calculates ETA, and updates Visdom windows.
    """
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
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=int(batches_left * self.mean_period / batches_done))))

        # Update Visdom Images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title': image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title': image_name})

        # Logic for end of epoch
        if (self.batch % self.batches_epoch) == 0:
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), 
                                                                opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

# --- Buffer & Image Saving ---

class ReplayBuffer():
    """
    Maintains a buffer of previously generated images to update discriminators (CycleGAN style).
    """
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Buffer size must be greater than 0.'
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
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

def save_fake_image(fake_B, A_core_names, A_coords, save_dir):
    """
    Saves generated patches as individual image files.
    """
    os.makedirs(save_dir, exist_ok=True)
    to_pil = T.ToPILImage()
    N = fake_B.shape[0]
    for idx in range(N):
        img = (fake_B[idx].cpu().detach() + 1.0) / 2.0
        img_pil = to_pil(img)
        core_name = A_core_names[idx]
        save_path = os.path.join(save_dir, core_name)
        img_pil.save(save_path)
        print(f"Saved: {save_path}")

def save_stack_fake_image(src, dst, fake, save_path, png_id):
    """
    Saves a comparison grid: [Source, Fake, Target].
    """
    stacked_images = torch.stack((src, fake, dst), dim=1)
    stacked_images = 0.5 * (stacked_images + 1.0)
    mixed_images = stacked_images.view(-1, *src.shape[1:])
    mixed_images = F.interpolate(mixed_images, size=(256, 256), mode='bilinear', align_corners=False)
    grid = vutils.make_grid(mixed_images, nrow=6, padding=2, normalize=True, scale_each=True)
    save_image(grid, f"{save_path}/fake_{png_id}.png", normalize=True)
    print(f"Saving comparison grid to {save_path}/fake_{png_id}.png")

def save_stack_fake_image_cond(src, cond, dst, fake, save_path, png_id):
    """
    Saves a conditional comparison grid: [Source, Condition, Fake, Target].
    """
    if cond.shape[1] == 1 and src.shape[1] == 3:
        cond = cond.repeat(1, 3, 1, 1)
    stacked_images = torch.stack((src, cond, fake, dst), dim=1)
    stacked_images = 0.5 * (stacked_images + 1.0)
    mixed_images = stacked_images.view(-1, *src.shape[1:])
    mixed_images = F.interpolate(mixed_images, size=(256, 256), mode='bilinear', align_corners=False)
    grid = vutils.make_grid(mixed_images, nrow=8, padding=2, normalize=True, scale_each=True)
    save_image(grid, f"{save_path}/fake_{png_id}.png", normalize=True)
    print(f"Saving conditional grid to {save_path}/fake_{png_id}.png")