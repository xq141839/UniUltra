import os
from skimage import io, transform, color, img_as_ubyte
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import torchvision.transforms as pytorch_transforms
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensor 
import albumentations as A

class BinaryLoader(Dataset):
        def __init__(self, data_name, jsfiles, transforms, pixel_mean=[123.675, 116.280, 103.530], pixel_std=[58.395, 57.12, 57.375]):
            self.path = f'/home/***/FTSAM/datasets/{data_name}'
            self.data_name = data_name
            self.jsfiles = jsfiles
            self.img_tesnor = pytorch_transforms.Compose([pytorch_transforms.ToTensor(), ])
            self.transforms = transforms
            self.img_size = 1024
            self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
            self.pixel_std = torch.Tensor(pixel_mean).view(-1, 1, 1)
            
        
        def __len__(self):
            return len(self.jsfiles)
              
        
        def __getitem__(self,idx):
            image_id = list(self.jsfiles[idx].split('.'))[0]

            image_path = os.path.join(self.path,'image_1024/',image_id)
            mask_path = os.path.join(self.path,f'mask_1024/',image_id)
            # pos_path = os.path.join(self.path,f'mask_pos/',image_id)
 
    
            img = io.imread(image_path+'.png')[:,:,:3].astype('float32')
            mask = io.imread(mask_path+'.png', as_gray=True).astype(np.uint8)
            # pos = io.imread(pos_path+'.png', as_gray=True).astype(np.uint8)

            mask[mask>0]=255
            # mask[mask<255]=0
            

            data_group = self.transforms(image=img, mask=mask)
            img_resized = data_group['image']
            mask = data_group['mask']
            # pos = data_group['mask2']

            img = self.img_tesnor(img)
            img = self.preprocess(img)

   
            return (img_resized, img, mask, image_id)
        
        def preprocess(self, x):
            """Normalize pixel values and pad to a square input."""
            # Normalize colors
            x = (x - self.pixel_mean) / self.pixel_std

            # Pad
            h, w = x.shape[-2:]
            padh = self.img_size - h
            padw = self.img_size - w
            x = F.pad(x, (0, padw, 0, padh))

            return x
        