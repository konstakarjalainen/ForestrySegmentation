import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from einops import rearrange


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, is_unet: bool,
                 mask_suffix: str = '', forest: bool = False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mask_suffix = mask_suffix
        self.forest = forest
        self.is_unet = is_unet
        if not self.forest:
            self.colormap = [[0, 0, 0],[108, 64, 20],[255, 229, 204],[0, 102, 0],[0, 255, 0],[0, 153, 153],[0, 128, 255],[0, 0, 255],
                        [255, 255, 0],[255, 0, 127],[64, 64, 64],[255, 128, 0],[255, 0, 0],[153, 76, 0],[102, 102, 0],[102, 0, 0],
                        [0, 255, 128],[204, 153, 255],[255, 153, 204],[0, 102, 102],[153, 204, 255],[102, 255, 255],[101, 101, 11],[114, 85, 47]]
        else:
            self.colormap = [[170, 170, 170], [0, 255, 0], [102, 102, 51], [0, 60, 0], [0, 120, 255], [0, 0, 0]]

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, colormap, is_mask, is_forest, is_unet):
        # ViT requires 384x384 input
        if is_unet:
            if is_forest:
                newW = 768
                newH = 384
            else:
                newW = 680
                newH = 550
        else:
            newW = 384
            newH = 384

        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)
        if is_mask:

            output_mask = np.zeros((newH,newW),dtype=int)
            for i, color in enumerate(colormap):
                truthmap = np.all(np.equal(img_ndarray,color),axis=-1).astype(int)
                output_mask = np.where(truthmap == 0, output_mask, i)

            img_ndarray = output_mask
        else:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray/255
        if not is_unet:
            if is_mask:
                img_ndarray = rearrange(img_ndarray, 'b (h p1) (w p2) -> (b h w) (p1 p2)', p1=16, p2=16)
            else:
                img_ndarray = rearrange(img_ndarray, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=16, p2=16)

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]

        img_file = list(self.images_dir.glob(name + '.*'))
        # 2 kinds of names in masks
        if self.forest:
            nm = name.split('_')
            mask_file = list(self.masks_dir.glob(nm[0] + self.mask_suffix + '.*'))
            if len(mask_file) == 0:
                mask_file = list(self.masks_dir.glob(name + '.*'))
        else:
            mask_file = list(self.masks_dir.glob(name + '.*'))
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.colormap, is_mask=False, is_forest=self.forest, is_unet=self.is_unet)
        mask = self.preprocess(mask, self.colormap, is_mask=True, is_forest=self.forest, is_unet=self.is_unet)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class ForestDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, is_unet):
        super().__init__(images_dir, masks_dir, is_unet, mask_suffix='_mask', forest=True)
