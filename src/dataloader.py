from torch.utils.data import Subset,Dataset
import torch
import os
import numpy as np
import cv2

def collate_fn(batch):
    imgs = [img for img, _ in batch]
    labels = torch.tensor([label for _, label in batch])
    return imgs, labels


class ImageDataset(Dataset):
    def __init__(self,root_path : str,img_size=(256,256)):
        classes = os.listdir(root_path)
        self.img_size = img_size
        self.classes = classes
        data = []
        for idx,class_name in enumerate(classes):
            class_path = os.path.join(root_path,class_name)
            files = os.listdir(class_path)
            for file in files:
                filepath = os.path.join(class_path,file)
                data.append({"image_path":filepath,"label":class_name,"id":idx})
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        curr = self.data[idx]
        label = curr['id']
        img_path = curr['image_path']
        img = cv2.imread(img_path)
        img = cv2.resize(img,(self.img_size))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img,label

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    if len(img.shape) == 2:
        return clahe.apply(img)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_hist_equalization(img):
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def simple_augment(img,
                   p_flip=0.5,
                   p_rotate=1.0,
                   p_he=0.3,
                   p_clahe=0.3):

    if np.random.rand() < p_flip:
        img = cv2.flip(img, 1)

    if np.random.rand() < p_rotate:
        angle = np.random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(
            img, M, (w, h),
            borderMode=cv2.BORDER_REFLECT
        )

    if np.random.rand() < p_he:
        img = apply_hist_equalization(img)

    if np.random.rand() < p_clahe:
        img = apply_clahe(img)

    return img


class AugmentedSubset(Subset):
    def __init__(self, subset, augment_fn=None):
        super().__init__(subset.dataset, subset.indices)
        self.augment_fn = augment_fn

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        if self.augment_fn:
            img = self.augment_fn(img)
        return img, label
