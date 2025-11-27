from torch.utils.data import Dataset
import torch
import os
import numpy as np
import cv2

def waste_collate_fn(batch):
    imgs = [img for img, _ in batch]
    labels = torch.tensor([label for _, label in batch])
    return imgs, labels


class RealWasteDataset(Dataset):
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


