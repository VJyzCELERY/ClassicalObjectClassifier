from src.model import Classifier
from src.dataloader import RealWasteDataset,waste_collate_fn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import torch.nn as nn

class RealWasteTrainer:
    def __init__(self,model : Classifier,train_set : RealWasteDataset,val_set : RealWasteDataset = None, batch_size=32,lr = 1e-3,device='cpu'):
        self.train_loader = DataLoader(train_set,batch_size,shuffle=True,collate_fn=waste_collate_fn)
        self.device = device
        if val_set is not None:
            self.val_loader = DataLoader(val_set,batch_size,shuffle=False,collate_fn=waste_collate_fn)
        else:
            self.val_loader=None
        self.class_names = model.classes
        self.model = model
        self.lr = lr
        self.optim = optim.Adam(model.parameters(),lr=lr)
        self.criterion = nn.CrossEntropyLoss()
    def visualize_batch(self, imgs, preds, labels, class_names=None, max_samples=4):
        # Convert list of images to tensor if needed
        if isinstance(imgs, list):
            imgs = np.stack(imgs, axis=0)
            imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).float()  # (B,H,W,C) -> (B,C,H,W)

        imgs = imgs.cpu().numpy()
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()

        # Choose random indices to visualize
        batch_size = imgs.shape[0]
        indices = random.sample(range(batch_size), min(max_samples, batch_size))

        plt.figure(figsize=(16, 8))
        for i, idx in enumerate(indices):
            plt.subplot(1, len(indices), i + 1)
            plt.imshow(imgs[idx].transpose(1, 2, 0))
            title = f"P:{preds[idx]} | T:{labels[idx]}"
            if class_names:
                title = f"P:{class_names[preds[idx]]} | T:{class_names[labels[idx]]}"
            plt.title(title)
            plt.axis("off")

        plt.show()


    def train_one_epoch(self):
        """Train the model for a single epoch with tqdm."""
        self.model.train()
        total_loss = 0
        train_pbar = tqdm(self.train_loader, desc="Training",leave=False)
        correct = 0
        total = 0
        for imgs, labels in train_pbar:
            labels = labels.to(self.device)

            # Forward
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)

            # Backward
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
            train_pbar.set_postfix(acc=correct/total,loss=loss.item())

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = correct / total
        return avg_loss,avg_acc
    def train(self, epochs=10, visualize_every=5):
        train_losses=[]
        train_accuracies=[]
        val_losses=[]
        val_accuracies=[]
        for epoch in range(1, epochs + 1):
            train_loss,train_acc = self.train_one_epoch()
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            if self.val_loader is not None:
                val_loss,val_acc=self.validate(epoch, visualize=(epoch % visualize_every == 0))
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                print(f"Epoch {epoch} Train Loss: {train_loss:.4f} | Train Acc : {train_acc:.4f} | Val Loss : {val_loss:.4f} | Val Acc : {val_acc:.4f}")
            else:
                print(f"Epoch {epoch} Train Loss: {train_loss:.4f} | Train Acc : {train_acc:.4f}")
        return train_losses,train_accuracies,val_losses,val_accuracies

    def validate(self,epoch, visualize=False):
        """Validate the model for a single epoch with tqdm."""
        if self.val_loader is None:
            return

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        val_imgs_display = None
        val_preds_display = None
        val_labels_display = None

        val_pbar = tqdm(self.val_loader, desc="Validation",leave=False)

        with torch.no_grad():
            for imgs, labels in val_pbar:
                labels = labels.to(self.device)

                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                if visualize and val_imgs_display is None:
                    val_imgs_display = imgs
                    val_preds_display = preds
                    val_labels_display = labels

                val_pbar.set_postfix(loss=loss.item(), acc=correct / total)

        avg_loss = total_loss / len(self.val_loader)
        acc = correct / total

        if visualize and val_imgs_display is not None:
            self.visualize_batch(val_imgs_display, val_preds_display, val_labels_display, self.class_names)

        self.model.train()
        return avg_loss, acc