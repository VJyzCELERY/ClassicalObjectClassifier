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

def simple_augment(img):
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)

    angle = np.random.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

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
