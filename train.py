#!/usr/bin/python3
import argparse
import yaml
import torch
import logging
import time
import os
import random
import numpy as np
import torch

import dataset as dataset_module
from models.pid_models_11 import PIDModel
from utils import flatten

def setup_seed(seed=42):
    """
    Set random seed for reproducibility across Python, Numpy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Use deterministic algorithms for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[Info] Random seed set to: {seed}")

def train(opt, device):
    """
    Main training loop for PIDModel.
    """
    if not os.path.exists(opt["save_path"]):
        os.makedirs(opt["save_path"])

    # Configure logging
    logging.basicConfig(
            filename=f'{opt["save_path"]}/training.log',
            filemode='w',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Log training options
    logging.info("========== Training Options ==========")
    for k, v in opt.items():
        logging.info(f"{k}: {v}")
    logging.info("======================================")
    
    # Model definition (PIDModel only)
    if opt['gan_type'] == 'pid':
        model = PIDModel(opt, device)
    else:
        raise ValueError(f"Invalid GAN type: {opt['gan_type']}. This script only supports 'pid'.")
    
    # Dataset and Dataloader setup
    dataset = dataset_module.BaseDataset(opt)
    dataloader = dataset_module.prepare_image_patch_dataloader(args=opt, dataset=dataset, shuffle=True)

    # Main Training Loop
    for epoch in range(opt['train.params.epoch'], opt['train.params.n_epochs']):
        epoch_start_time = time.time()
        for i, batch in enumerate(dataloader):

            if len(opt['gpu_ids']) > 0:
                torch.cuda.synchronize()

            # Data-dependent initialization (for the first batch of the first epoch)
            if epoch == 0 and i == 0:
                model.data_dependent_initialize(batch)
                model.setup()      # Load/print networks and create schedulers
                model.parallelize()

            # Prepare input: batch[0] is source, batch[3] is target
            model_input = {
                'src': batch[0], 
                'dst': batch[3], 
                'current_epoch': epoch
            }

            model.set_input(model_input)  # Unpack data and apply preprocessing
            model.optimize_parameters()   # Update weights

            if len(opt['gpu_ids']) > 0:
                torch.cuda.synchronize()

            epoch_duration = time.time() - epoch_start_time

            # Log losses
            if i % opt["train.params.save.loss_logging_freq"] == 0:
                losses = model.get_current_losses()
                epoch_str = f"[Epoch {epoch}/{opt['train.params.n_epochs']}] [Batch {i}/{len(dataloader)}] [Time: {epoch_duration:.2f}s]"
                loss_str = " ".join([f"[{k}: {v:.4f}]" for k, v in losses.items()])
                log_result = f"{epoch_str} {loss_str}"
                logging.info(log_result)
                print(log_result)

        # Update learning rate and save model at the end of each epoch
        model.update_learning_rate()
        model.save_networks(epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Path configuration
    parser.add_argument('--config_path', type=str, default="configs/pid_config.yml")
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save results')
    
    # Experiment parameters
    parser.add_argument("--src_marker", type=str, default='HE', help="Source stain (e.g., HE)")
    parser.add_argument("--dst_marker", type=str, default='PR', help="Target stain (e.g., PR)")
    parser.add_argument("--mode", type=str, default='train', help='train / test')
    parser.add_argument("--gan_type", type=str, default='pid', help='Model type (pid)')
    
    # Hardware configuration
    parser.add_argument('--cuda', type=eval, default=True)
    parser.add_argument('--gpu_ids', type=int, default=[0], nargs='+', help='GPU IDs to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    opt = parser.parse_args()
    opt = vars(opt)
    
    # Initialize seed before model creation
    setup_seed(opt['seed'])

    # Load and merge yaml config
    with open(opt['config_path']) as f:
        config = yaml.safe_load(f)
    opt.update(config)
    opt = flatten(opt)

    device = f'cuda:{opt["gpu_ids"][0]}' if torch.cuda.is_available() and opt['cuda'] else 'cpu'

    train(opt, device)