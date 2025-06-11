import timm
import torch 
import yaml
import os
import numpy as np
import time
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from ContraVisEnc.utils import InfoNCE, smooth_rank_measure, flatten
from ContraVisEnc.process_args import process_args

from dataset import BaseDataset, prepare_image_patch_dataloader


def train_loop(config, loss_fn, src_model, dst_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler, device, logging):

    src_model.eval()
    dst_model.train()

    ep_loss = 0.
    fb_time = 0.
    all_embeds = []
    
    for i, batch in enumerate(dataloader):
        
        s_fb = time.time()

        # set data on device and set to float-16. 
        src_imgs = batch[0].to(device)
        dst_imgs = batch[3].to(device)
                
        # forward pass 
        src_features = src_model(src_imgs)
        dst_features = dst_model(dst_imgs)

        # inter modality loss src <-> dst
        # symmetric 表示计算对称损失，以增强模型对齐能力
        loss = loss_fn(query=src_features, positive_key=dst_features, symmetric=config["symmetric_cl"])
        
        # accumate loss
        loss.backward() 
        
        optimizer.step()
        optimizer.zero_grad()
                
        e_fb = time.time()
        fb_time += e_fb - s_fb

        # step scheduler
        if epoch <= config["warmup_epochs"]:
            scheduler_warmup.step()
        else:
            scheduler.step()  
            
        # if (i % 5) == 0:
        #     print(f"Loss for batch: {i} = {loss}")
            
        ep_loss += loss.item()
        
        # save the wsi_emb 
        all_embeds.extend(dst_features.float().cpu().detach().numpy())

        if i % config["train.params.save.loss_logging_freq"] == 0:
            logging.info("[Epoch %d/%d] [Batch %d/%d] [Time: %.2f] [Lr: %.6f] [loss: %f] [Sum loss: %f]"
                % (epoch, config["epochs"], i, len(dataloader), fb_time, scheduler.get_last_lr()[0], loss.item(), ep_loss)
                )
            print("[Epoch %d/%d] [Batch %d/%d] [Time: %.2f] [Lr: %f] [loss: %.6f] [Sum loss: %f]"
                % (epoch, config["epochs"], i, len(dataloader), fb_time, scheduler.get_last_lr()[0], loss.item(), ep_loss)
                )

    # track rank
    all_embeds_tensor = torch.Tensor(np.array(all_embeds))
    rank = smooth_rank_measure(all_embeds_tensor)  
    return ep_loss, rank

def write_dict_to_config_file(config_dict, json_file_path):
    """
    Write a dictionary to a configuration file.

    Args:
        config_dict (dict): The dictionary to be written to the config file.
        config_file_path (str): The path to the configuration file.

    Returns:
        None
    """
    config_dict_dump = {}
    for key in config_dict:
        config_dict_dump[key] = str(config_dict[key])
    
    with open(json_file_path, 'w') as jsonfile:
        json.dump(config_dict_dump, jsonfile, indent=4)


def get_args():
    args = process_args()
    args = vars(args)

    with open(args['config_path']) as f:
        config = yaml.safe_load(f)
    args.update(config)
    args = flatten(args)
    
    # get dtype 
    if args['dtype'] == "float64":
        args['dtype'] = torch.float64
    elif args['dtype'] == "float32":
        args['dtype'] = torch.float32
    elif args['dtype'] == "float16":
        args['dtype'] = torch.float16
    elif args['dtype'] == "bfloat16":
        args['dtype'] = torch.bfloat16
    
    return args

def main():
    # setup args
    args = get_args()

    if not os.path.exists(args["save_path"]):
        os.makedirs(args["save_path"])

    logging.basicConfig(
            filename=f'{args["save_path"]}/training.log',
            filemode='w',  # 新建/覆盖日志文件
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info("========== Training Options ==========")
    for k, v in args.items():
        logging.info(f"{k}: {v}")
    logging.info("======================================")

    os.makedirs(args['save_path'], exist_ok=True)
    
    # make the datasets: Multimodal, Slide train and Slide external to derive the embeddings. 
    print("* Setup dataset...")
        
    dataset = BaseDataset(args)
    dataloader = prepare_image_patch_dataloader(args=args, dataset=dataset, shuffle=True)

    # set up model
    print("* Setup model...")
    src_model = timm.create_model(args['timm_model'], pretrained=True) # 预训练权重dtype为float32
    dst_model = timm.create_model(args['timm_model'], pretrained=False)

    # 禁止src_model梯度更新
    for param in src_model.parameters():
        param.requires_grad = False

    if len(args["gpu_ids"]) > 1 and torch.cuda.is_available():
        print(f"* Using {len(args['gpu_ids'])} GPUs.")
        src_model = src_model.to(f'cuda:{args["gpu_ids"][0]}')
        dst_model = dst_model.to(f'cuda:{args["gpu_ids"][0]}')
        src_model = nn.DataParallel(src_model, device_ids=args["gpu_ids"])
        dst_model = nn.DataParallel(dst_model, device_ids=args["gpu_ids"])
        device = 'cuda'
    else:
        device = 'cpu'
    
    # set up optimizers
    print("* Setup optimizer...")
    optimizer = optim.AdamW(dst_model.parameters(), lr=args["learning_rate"], weight_decay=args['weight_decay'])
    
    # set up schedulers
    print("* Setup schedulers...")
    T_max = (args["epochs"] - args["warmup_epochs"]) * len(dataloader) if args["warmup"] else args["epochs"] * len(dataloader)
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=T_max,
        eta_min=args["end_learning_rate"]
    )
    
    if args["warmup"]:
        scheduler_warmup = LinearLR(
            optimizer, 
            start_factor=0.00001,
            total_iters=args["warmup_epochs"] * len(dataloader)
        )
    else:
        scheduler_warmup = None
    
    # set up losses
    print("* Setup losses...")
    loss_fn = InfoNCE(temperature=args["temperature"])
    
    # main training loop
    for epoch in range(args["epochs"]):
        
        # print(f"Training for epoch {epoch}...")
        
        # train
        start = time.time()
        ep_loss, train_rank = train_loop(args, loss_fn, src_model, dst_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler, device, logging)
        last_lr = scheduler.get_last_lr()[0]
        end = time.time()

        logging.info("[Epoch %d/%d] [Total loss: %.6f] [Train rank: %s] [Last lr: %.6f] [Total time: %.3f seconds]"
            % (epoch, args["epochs"], ep_loss, train_rank, last_lr[0] if isinstance(last_lr, list) else last_lr, end - start)
        )
        logging.info("===========================================================================================")
        print("[Epoch %d/%d] [Total loss: %.6f] [Train rank: %s] [Last lr: %.6f] [Total time: %.3f seconds]"
            % (epoch, args["epochs"], ep_loss, train_rank, last_lr[0] if isinstance(last_lr, list) else last_lr, end - start)
        )

        if epoch % 25 == 0:
            torch.save(dst_model.state_dict(), os.path.join(args['save_path'], f"{args['dst_marker']}_VisEnc_epoch_{epoch}.pt"))