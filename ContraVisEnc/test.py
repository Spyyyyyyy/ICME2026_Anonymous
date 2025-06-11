import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

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
        pr_path = os.path.join(self.pr_dir, fname + '-PR.png') if not os.path.exists(os.path.join(self.pr_dir, fname + '.png')) else os.path.join(self.pr_dir, fname + '.png')

        # load images
        pred_img = Image.open(pred_path).convert('RGB')
        he_img = Image.open(he_path).convert('RGB')
        pr_img = Image.open(pr_path).convert('RGB')

        if self.transform:
            pred_img = self.transform(pred_img)
            he_img = self.transform(he_img)
            pr_img = self.transform(pr_img)

        return {'prediction': pred_img, 'HE': he_img, 'PR': pr_img}

# example usage:
if __name__ == '__main__':
    # directories
    pred_dir = '/data1/sunpengyu/Task_VirtualStain/Result/20May2025/CPT-2048_8/prediction/HE2PR/24-00456A07'
    he_dir   = '/data6/sunpengyu/Task_VirtualStain/Data/szph_preprocessed_v1_reg_patch_2048-8/HE/24-00456A07'
    pr_dir   = '/data6/sunpengyu/Task_VirtualStain/Data/szph_preprocessed_v1_reg_patch_2048-8/PR/24-00456A07-PR'

    # define any transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = TripletImageDataset(pred_dir, he_dir, pr_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    # iterate
    for batch in dataloader:
        pred_batch = batch['prediction']
        he_batch = batch['HE']
        pr_batch = batch['PR']
        print(pred_batch.shape, he_batch.shape, pr_batch.shape)
        break
