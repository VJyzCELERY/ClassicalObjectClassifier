from src.model import Classifier
from src.dataloader import ImageDataset,collate_fn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import torch.nn as nn

class ModelTrainer:
    def __init__(self,model : Classifier,train_set : ImageDataset,val_set : ImageDataset = None, batch_size=32,lr = 1e-3,device='cpu',return_fig=False):
        self.train_loader = DataLoader(train_set,batch_size,shuffle=True,collate_fn=collate_fn)
        self.device = device
        if val_set is not None:
            self.val_loader = DataLoader(val_set,batch_size,shuffle=False,collate_fn=collate_fn)
        else:
            self.val_loader=None
        self.class_names = model.classes
        self.model = model
        self.lr = lr
        self.optim = optim.Adam(model.parameters(),lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.return_fig=return_fig
    
    def visualize_batch(self, imgs, preds, labels, class_names=None, max_samples=4):
        
        first_image = imgs
        if isinstance(imgs, list):
            imgs = np.stack(imgs, axis=0)
            imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).float()

        imgs_np = imgs.cpu().numpy()
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()

        batch_size = imgs_np.shape[0]
        indices = random.sample(range(batch_size), min(max_samples, batch_size))
        first_image = first_image[indices[0]]
        fig_pred = plt.figure(figsize=(6 * len(indices), 5))
        grid = fig_pred.add_gridspec(1, len(indices))

        for col, idx in enumerate(indices):
            ax = fig_pred.add_subplot(grid[0, col])
            ax.imshow(imgs_np[idx].transpose(1, 2, 0))

            if class_names:
                title = f"P: {class_names[preds[idx]]} | T: {class_names[labels[idx]]}"
            else:
                title = f"P: {preds[idx]} | T: {labels[idx]}"

            ax.set_title(title)
            ax.axis("off")

        fig_pred.tight_layout()
        raw_features = self.model.visualize_feature(first_image,show=False)
        feature_figs = []

        for f in raw_features:

            if isinstance(f, plt.Figure):
                feature_figs.append(f)
                continue

            if hasattr(f, "mode"):
                f = np.array(f)

            fig = plt.figure(figsize=(16, 16))
            ax = fig.add_subplot(111)
            ax.imshow(f)
            ax.axis("off")
            feature_figs.append(fig)
            

        all_figs = [fig_pred] + feature_figs
        plt.show()
        plt.close(fig_pred)
        for i,fig in enumerate(feature_figs):
            plt.figure(fig)
            plt.show()
            plt.close(fig) 
        if self.return_fig:
            return all_figs
        else:
            return None


    def train_one_epoch(self):
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
                val_loss,val_acc,fig=self.validate(epoch, visualize=(epoch % visualize_every == 0))
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                print(f"Epoch {epoch} Train Loss: {train_loss:.4f} | Train Acc : {train_acc:.4f} | Val Loss : {val_loss:.4f} | Val Acc : {val_acc:.4f}")
                yield train_loss,train_acc,val_loss,val_acc,fig
            else:
                print(f"Epoch {epoch} Train Loss: {train_loss:.4f} | Train Acc : {train_acc:.4f}")
                yield train_loss,train_acc,None,None,None
        yield train_losses,train_accuracies,val_losses,val_accuracies,None

    def validate(self,epoch, visualize=False):
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
        fig = None
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
            fig = self.visualize_batch(val_imgs_display, val_preds_display, val_labels_display, self.class_names)

        self.model.train()
        return avg_loss,acc,fig