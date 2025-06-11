from typing import Optional, Callable, Any, List, Tuple, Dict
import os
import glob
import random
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd

def get_split(splits_path: str, split_key: str):
    all_splits = pd.read_csv(splits_path)
    split = all_splits[split_key]
    split = split.dropna().reset_index(drop=True)
    return split

class BaseDataset(Dataset):
    def __init__(
            self,
            args,
            image_fmt: str='png',
            **kwargs
    ) -> None:
        self.src_marker = args['src_marker']
        self.dst_marker = args['dst_marker']
        self.patch_size = args['train.data.patch_size']
        self.patch_norm = args['train.params.patch_norm']

        # 最大样本数，dst表示目标图片
        self.max_src_samples = args['train.data.max_src_samples']
        self.downsample = args['train.data.downsample']
        self.data_path = args['data_path']
        self.image_fmt = image_fmt

        # 用于训练的patch_size
        self.downsampled_size = (
            int(self.patch_size // self.downsample),
            int(self.patch_size // self.downsample),
        )
        self.mode = args['mode']

        print('Source marker: ', self.src_marker)
        print('Destination marker: ', self.dst_marker)

        self.split = get_split(splits_path=os.path.join(args['split_csv_path']), split_key=f'{self.mode}_cores')

        # define src and dst splits
        self._get_splits()

        # load image and patch paths for src and dst splits
        self._base_load()

        self.transform = self._get_transform()

    def _get_splits(self):
        self.split = self.split.values.tolist()

        # select subset of src cores for training
        # if self.is_train and (self.max_src_samples != -1):
        if (self.max_src_samples != -1):
            assert self.max_src_samples <= len(self.split), "ERROR: max src samples is greater than src split"
            np.random.seed(0)
            idx = np.random.choice(len(self.split), size=min(self.max_src_samples, len(self.split)), replace=False)
            self.split = [self.split[x] for x in idx]

        # select common src and dst cores
        src_split = []
        dst_split = []

        # for x in self.split:
        #     y = x + f'-{self.dst_marker}'
        #     src_split.append(x)
        #     dst_split.append(y)

        # dst数目多于src，按照目标集进行split
        for x in self.split:
            y = x.replace(f"-{self.dst_marker}", "")
            src_split.append(y)
            dst_split.append(x)

        self.src_split = src_split
        self.dst_split = dst_split

    def _base_load(self):

        # if self.is_train:
        print('Loading Paths...')
        # get patch names
        def _get_patch_paths(marker: str, split: List) -> Dict:
            paths_ = dict()
            # for core_name in split:
            for core_name in tqdm(split):
                paths_[core_name] = glob.glob(f'{self.data_path}/{marker}/{core_name}/{core_name}_*.{self.image_fmt}')
            return paths_
        src_patch_paths = _get_patch_paths(self.src_marker, self.src_split)
        dst_patch_paths = _get_patch_paths(self.dst_marker, self.dst_split)

        print('Loading Patches...')
        self.src_patch_coords, self.src_paths, self.src_patch_core_names, self.dst_patch_coords, self.dst_paths, self.dst_patch_core_names = self._load_patch_paths(src_patch_paths, dst_patch_paths, self.src_split)

        print("len(src_patch_coords) :", len(self.src_paths), "len(dst_patch_coords) :", len(self.dst_paths))


    def _get_transform(self):

        transform_list = []
        transform_list.append(transforms.Resize(self.downsampled_size))
        transform_list.append(transforms.ToTensor())
        if self.patch_norm:
            transform_list.append(
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5)
                )
            )

        return transforms.Compose(transform_list)

    def _load_patch_paths(
            self,
            src_patch_paths_ori: [],
            dst_patch_paths_ori: [],
            split: []
) -> Tuple[List[List[int]], List[str], List[str], List[List[int]], List[str], List[str]]:        # load pre-extracted patches and corresponding information
        src_patch_coords = list()
        src_patch_paths = list()
        dst_patch_coords = list()
        dst_patch_paths = list()
        src_patch_core_names = list()
        dst_patch_core_names = list()

        for core_name in split:
            src_paths = src_patch_paths_ori[core_name]
            dst_paths = dst_patch_paths_ori[f"{core_name}-{self.dst_marker}"] # 通过HE构建其他染色索引

            for src_path_ in tqdm(src_paths):
                
                dst_path_ = src_path_.replace(self.src_marker, self.dst_marker).replace(core_name, f"{core_name}-{self.dst_marker}")
                
                if dst_path_ in dst_paths:

                    src_basename = os.path.basename(src_path_)
                    src_y = int(src_basename.rpartition('_y_')[2].rpartition('_x_')[0])
                    src_x = int(src_basename.rpartition('_x_')[2].rpartition('.png')[0])
                    
                    dst_basename = os.path.basename(dst_path_)
                    dst_y = int(dst_basename.rpartition('_y_')[2].rpartition('_x_')[0])
                    dst_x = int(dst_basename.rpartition('_x_')[2].rpartition('.png')[0])
                    
                    if src_y != dst_y or src_x != dst_x:
                        continue
                    
                    # 不对坐标进行下采样
                    # if self.downsample != 1:
                    #     src_y = int(src_y/self.downsample)
                    #     src_x = int(src_x/self.downsample)
                    #     dst_y = int(dst_y/self.downsample)
                    #     dst_x = int(dst_x/self.downsample)

                    src_patch_coords.append([src_y, src_x])
                    dst_patch_coords.append([dst_y, dst_x])

                    # load patch paths
                    src_patch_paths.append(src_path_)
                    dst_patch_paths.append(dst_path_)
                    src_patch_core_names.append(core_name)
                    dst_patch_core_names.append(f"{core_name}-{self.dst_marker}")

        return src_patch_coords, src_patch_paths, src_patch_core_names,  dst_patch_coords, dst_patch_paths, dst_patch_core_names
    
    # src : source, dst : destination(HE)
    def _pair_patches(self, src_patch_coords, dst_patch_coords, src_paths, dst_paths):

        # print("len(src_patch_coords) :", len(src_patch_coords), "len(dst_patch_coords) :", len(dst_patch_coords))

        if len(src_patch_coords) != len(dst_patch_coords):

            dst_patch_coords_set = set(map(tuple, dst_patch_coords))

            valid_indices = [i for i, coord in enumerate(src_patch_coords) if tuple(coord) in dst_patch_coords_set]

            src_patch_coords = [src_patch_coords[i] for i in valid_indices]
            src_paths = [src_paths[i] for i in valid_indices]

        return src_paths, dst_paths


    def __getitem__(self, index: int):

        src_patches = Image.open(self.src_paths[index]).convert('RGB')
        if self.transform:
            src_patches = self.transform(src_patches)

        dst_patches = Image.open(self.dst_paths[index]).convert('RGB')
        if self.transform:
            dst_patches = self.transform(dst_patches)

        return src_patches, self.src_patch_core_names[index], self.src_patch_coords[index], \
            dst_patches, self.dst_patch_core_names[index], self.dst_patch_coords[index]


    def __len__(self) -> int:
        return len(self.dst_paths)

# def collate_images_patches_batch(batch):
#     src_patches = [item[0] for item in batch]
#     src_patch_core_names = [item[1] for item in batch]
#     src_patch_coords = [item[2] for item in batch]
#     dst_patches = [item[3] for item in batch]
#     dst_patch_core_names = [item[4] for item in batch]
#     dst_patch_coords = [item[5] for item in batch]

#     # 堆叠成 [batch, 3, 256, 256]
#     src_patches = torch.stack(src_patches, dim=0)
#     dst_patches = torch.stack(dst_patches, dim=0)

def collate_images_patches_batch(batch):
    # batch 是一个列表，每个元素是 __getitem__ 返回的 6 元组
    src_patches = [item[0] for item in batch]
    src_patch_core_names = [item[1] for item in batch]
    src_patch_coords = [item[2] for item in batch]
    dst_patches = [item[3] for item in batch]
    dst_patch_core_names = [item[4] for item in batch]
    dst_patch_coords = [item[5] for item in batch]

    # 堆叠成 [batch, 3, 256, 256]
    src_patches = torch.stack(src_patches, dim=0)
    dst_patches = torch.stack(dst_patches, dim=0)

    # core_names 依然是 list[str]，如需编码成 tensor 可进一步处理
    return src_patches, src_patch_core_names, src_patch_coords, dst_patches, dst_patch_core_names, dst_patch_coords

def prepare_image_patch_dataloader(
        args,
        dataset: Dataset,
        shuffle: bool = False,
        sampler: Optional = None,
        collate_fn: Optional[Callable] = collate_images_patches_batch,
        **kwargs
) -> DataLoader:
    dataloader = DataLoader(
        dataset,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        batch_size=args['train.params.patch_batch_size'] if args['mode'] == 'train' else args['test.patch_batch_size'],
        num_workers=args['train.params.num_workers'],
        drop_last=True if args['mode'] == 'train' else False
    )
    return dataloader