from typing import Optional, Callable, Any, List, Tuple, Dict
import os
import glob
import random
from tqdm import tqdm
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.utils as vutils

# Allow loading of truncated images and disable decompression bomb checks for large histology/WSI tiles
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class BaseDataset(Dataset):
    """
    Dataset for paired image-to-image translation.
    Supports synchronized cropping and resizing for training.
    """
    def __init__(
        self,
        args,
        image_fmt: str = 'png',
        **kwargs
    ) -> None:
        self.mode = args['mode']
        self.data_path = args.get('data_path')
        self.dst_marker = args.get('dst_marker') or args.get('data.dst_marker')
        
        # Configuration for patch sizes and downsampling
        self.patch_size = args.get('train.data.patch_size') or args.get('data.patch_size')
        self.downsample = args.get('train.data.downsample') or args.get('data.downsample')
        self.resize_size = args.get('train.data.resize_size') or args.get('data.resize_size')

        # Fallback defaults for missing config keys
        if self.patch_size is None:
            print("Warning: 'patch_size' not found. Defaulting to 256.")
            self.patch_size = 256
        if self.downsample is None:
            print("Warning: 'downsample' not found. Defaulting to 1.")
            self.downsample = 1
            
        self.downsampled_size = (int(self.resize_size), int(self.resize_size))
        
        self.resize_transform = None
        self.crop_size_train = None

        if self.mode == 'train':
            # Setup Training transforms
            self.resize_size_train = args.get('train.data.resize_size')
            self.crop_size_train = args.get('train.data.crop_size')
            
            if self.crop_size_train is None:
                print("Warning: 'train.data.crop_size' not found. Defaulting to 256.")
                self.crop_size_train = 256
            if self.resize_size_train is None:
                # Default resize to 110% of crop size to allow for cropping overhead
                self.resize_size_train = int(self.crop_size_train * 1.1) 
                print(f"Warning: 'train.data.resize_size' not found. Defaulting to {self.resize_size_train}.")

            # 1. Base Resize (Applied to both A and B)
            self.resize_transform = transforms.Resize(
                (self.resize_size_train, self.resize_size_train), 
                interpolation=transforms.InterpolationMode.BILINEAR
            )
            # 2. Post-crop transforms (Tensor conversion and Normalization)
            self.final_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            self.dir_A = os.path.join(self.data_path, self.dst_marker, 'TrainValAB', 'trainA')
            self.dir_B = os.path.join(self.data_path, self.dst_marker, 'TrainValAB', 'trainB')
        
        else: # 'test' or 'val' mode
            self.dir_A = os.path.join(self.data_path, self.dst_marker, 'TrainValAB', 'valA')
            self.dir_B = os.path.join(self.data_path, self.dst_marker, 'TrainValAB', 'valB')

            self.final_transform = transforms.Compose([
                transforms.Resize(self.downsampled_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        # Pair verification logic
        self.image_pairs = []
        paths_A = sorted(glob.glob(os.path.join(self.dir_A, '*.*')))
        
        print(f"Scanning for paired images in {self.dir_A} and {self.dir_B}...")
        for path_A in tqdm(paths_A):
            basename = os.path.basename(path_A)
            path_B = os.path.join(self.dir_B, basename)
            if os.path.exists(path_B):
                self.image_pairs.append((path_A, path_B))

        self.dataset_len = len(self.image_pairs)
        print(f"Found {self.dataset_len} valid paired images.")

    def __getitem__(self, index: int):
        path_A, path_B = self.image_pairs[index]
        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')

        if self.mode == 'train':
            # Synchronized transformations
            img_A = self.resize_transform(img_A)
            img_B = self.resize_transform(img_B)
            
            # Generate random crop parameters once and apply to both images
            i, j, h, w = transforms.RandomCrop.get_params(
                img_A, output_size=(self.crop_size_train, self.crop_size_train)
            )
            
            img_A = F.crop(img_A, i, j, h, w)
            img_B = F.crop(img_B, i, j, h, w)
            
            src_t = self.final_transform(img_A)
            dst_t = self.final_transform(img_B)
            
        else:
            # Deterministic transforms for Val/Test
            src_t = self.final_transform(img_A)
            dst_t = self.final_transform(img_B)

        src_name = os.path.basename(path_A)
        dst_name = os.path.basename(path_B)
        dummy_coords = [0, 0]
        
        return (
            src_t, src_name, dummy_coords,
            dst_t, dst_name, dummy_coords,
        )

    def __len__(self) -> int:
        return self.dataset_len

def collate_images_patches_batch(batch):
    """Custom collate to stack tensors and collect metadata lists."""
    src_ts, src_names, src_coords, dst_ts, dst_names, dst_coords = zip(*batch)
    return (
        torch.stack(src_ts),
        list(src_names),
        list(src_coords),
        torch.stack(dst_ts),
        list(dst_names),
        list(dst_coords),
    )

def prepare_image_patch_dataloader(
    args,
    dataset: Dataset,
    shuffle: bool = False,
    sampler: Optional[Any] = None,
    collate_fn: Callable = collate_images_patches_batch,
    **kwargs
) -> DataLoader:
    batch_size = args['train.params.patch_batch_size'] if args['mode']=='train' else args['test.patch_batch_size']
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args['train.params.num_workers'],
        collate_fn=collate_fn,
        drop_last=(args['mode']=='train')
    )

if __name__ == "__main__":
    # Test configuration
    args = {
        'mode': 'train',
        'dst_marker': 'PR',
        'train.data.patch_size': 1024,
        'train.data.downsample': 2,
        'data.patch_size': 512,
        'data.downsample': 2,
        'train.data.resize_size': 512,
        'train.data.crop_size': 256,
        'data_path': '/path/to/your/data', # Update this to your local path
        'train.params.patch_batch_size': 16,
        'train.params.num_workers': 4,
        'test.patch_batch_size': 8,
    }

    print(f"--- Starting Dataset Test (Mode: {args['mode']}) ---")
    
    dataset = BaseDataset(args)
    dataloader = prepare_image_patch_dataloader(args, dataset, shuffle=True)
    
    if len(dataset) > 0:
        batch = next(iter(dataloader))
        expected_size = args['train.data.crop_size']
        
        print("\n--- Batch Statistics ---")
        print(f"Source (A) shape: {batch[0].shape}")
        print(f"Target (B) shape: {batch[3].shape}")
        print(f"Filename A: {batch[1][0]}")
        print(f"Filename B: {batch[4][0]}")
        
        if batch[1][0] == batch[4][0]:
            print("Pair Consistency: PASSED")
        else:
            print("Pair Consistency: FAILED")

        # Visual verification: Save a sample pair
        try:
            # De-normalize from [-1, 1] to [0, 1]
            img_a_sample = (batch[0][0] + 1) / 2.0
            img_b_sample = (batch[3][0] + 1) / 2.0

            vutils.save_image(img_a_sample, "verify_A.png")
            vutils.save_image(img_b_sample, "verify_B.png")
            print("\nVerification images saved: verify_A.png and verify_B.png")

        except Exception as e:
            print(f"\nError saving verification images: {e}")
    else:
        print("\nError: No image pairs found. Check your paths.")