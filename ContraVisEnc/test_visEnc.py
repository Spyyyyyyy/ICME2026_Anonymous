import torch
import timm
import numpy as np
from PIL import Image
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from ContraVisEnc.process_args import process_args
import yaml
from ContraVisEnc.utils import InfoNCE, flatten
from dataset import BaseDataset, prepare_image_patch_dataloader
from utils import load_state_dict_strip_module

class TripletImageDataset(Dataset):
    def __init__(self, pred_dir, he_dir, pr_dir, transform=None):
        self.pred_dir = pred_dir
        self.he_dir = he_dir
        self.pr_dir = pr_dir
        self.transform = transform

        # list of filenames without extension
        self.filenames = [os.path.splitext(f)[0] for f in os.listdir(pred_dir) if f.endswith('.png')]
        self.filenames.sort()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        pred_path = os.path.join(self.pred_dir, fname + '.png')
        he_path = os.path.join(self.he_dir, fname + '.png')
        pr_path = os.path.join(self.pr_dir, fname.replace('_y', '-PR_y') + '.png') if not os.path.exists(os.path.join(self.pr_dir, fname + '.png')) else os.path.join(self.pr_dir, fname + '.png')

        # load images
        pred_img = Image.open(pred_path).convert('RGB')
        he_img = Image.open(he_path).convert('RGB')
        pr_img = Image.open(pr_path).convert('RGB')

        if self.transform:
            pred_img = self.transform(pred_img)
            he_img = self.transform(he_img)
            pr_img = self.transform(pr_img)

        return {'prediction': pred_img, 'HE': he_img, 'PR': pr_img}

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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 加载模型
    src_model = timm.create_model(args['timm_model'], pretrained=True)
    dst_model = timm.create_model(args['timm_model'], pretrained=False)
    dst_model_woCon = timm.create_model(args['timm_model'], pretrained=True)
    
    dst_state_dict = load_state_dict_strip_module(args['ckpt_path'])
    dst_model.load_state_dict(dst_state_dict)

    src_model.eval()
    dst_model.eval()
    dst_model_woCon.eval()

    src_model.to(device)
    dst_model.to(device)
    dst_model_woCon.to(device)

    # 准备测试图片
    # dataset = BaseDataset(args)
    # dataloader = prepare_image_patch_dataloader(args=args, dataset=dataset, shuffle=True)

    # 构建图片对
    pred_dir = '/data1/sunpengyu/Task_VirtualStain/Result/20May2025/CPT-2048_8/prediction/HE2PR/24-00456A07'
    he_dir   = '/data6/sunpengyu/Task_VirtualStain/Data/szph_preprocessed_v1_reg_patch_2048-8/HE/24-00456A07'
    pr_dir   = '/data6/sunpengyu/Task_VirtualStain/Data/szph_preprocessed_v1_reg_patch_2048-8/PR/24-00456A07-PR'

    # define any transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])

    dataset = TripletImageDataset(pred_dir, he_dir, pr_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    loss_fn = InfoNCE(temperature=args["temperature"])

    # 3. 前向推理
    with torch.no_grad():
        for i, batch in enumerate(dataloader):

            # src_imgs = batch[0].to(device)
            # dst_imgs = batch[3].to(device)
            pred_batch = batch['prediction'].to(device)
            he_batch = batch['HE'].to(device)
            pr_batch = batch['PR'].to(device)

            # print("src_imgs.dtype: ", src_imgs.dtype)
            # print("dst_imgs.dtype: ", dst_imgs.dtype)
            # exit()

            src_features = src_model(he_batch)
            dst_features = dst_model(pr_batch)
            dst_features_woCon = dst_model_woCon(pr_batch)
            dst_features_pred = dst_model(pred_batch)

            # loss_nce = loss_fn(query=src_features, positive_key=dst_features, symmetric=args["symmetric_cl"])
            loss_nce = loss_fn(query=src_features, positive_key=dst_features)
            loss_sim = F.cosine_similarity(src_features, dst_features).mean()
            loss_l2 = F.mse_loss(src_features, dst_features).mean()

            # loss_nce_woCon = loss_fn(query=src_features, positive_key=dst_features_woCon, symmetric=args["symmetric_cl"])
            loss_nce_woCon = loss_fn(query=src_features, positive_key=dst_features_woCon)
            loss_sim_woCon = F.cosine_similarity(src_features, dst_features_woCon).mean()
            loss_l2_woCon = F.mse_loss(src_features, dst_features_woCon).mean()

            loss_nce_pred = loss_fn(query=src_features, positive_key=dst_features_pred)
            loss_sim_pred = F.cosine_similarity(src_features, dst_features_pred).mean()
            loss_l2_pred = F.mse_loss(src_features, dst_features_pred).mean()

            print("[loss_nce: %.4f] [loss_sim: %.4f] [loss_l2: %.4f]"
                % (loss_nce.item(), loss_sim.item(), loss_l2.item())
                )
            print("[loss_nce_woCon: %.4f] [loss_sim_woCon: %.4f] [loss_l2_woCon: %.4f]"
                % (loss_nce_woCon.item(), loss_sim_woCon.item(), loss_l2_woCon.item())
                )
            print("[loss_nce_pred: %.4f] [loss_sim_pred: %.4f] [loss_l2_pred: %.4f]"
                % (loss_nce_pred.item(), loss_sim_pred.item(), loss_l2_pred.item())
                )
            print("--------------------------------")