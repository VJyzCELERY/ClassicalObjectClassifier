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
import time
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def model_evaluation(model, val_set, device,batch_size=32,num_workers=0, class_names=None):

    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    with torch.no_grad():
        for images, labels in val_loader:
            labels = labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    num_classes = y_prob.shape[1]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    cm = confusion_matrix(y_true, y_pred)

    cm_fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=75)
    ax.set_yticklabels(class_names)

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )

    cr_fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    table_data = []
    headers = ["Class", "Precision", "Recall", "F1", "Support"]

    for cls in class_names:
        row = report[cls]
        table_data.append([
            cls,
            f"{row['precision']:.3f}",
            f"{row['recall']:.3f}",
            f"{row['f1-score']:.3f}",
            int(row['support'])
        ])

    accuracy = report["accuracy"]
    macro_avg = report["macro avg"]
    weighted_avg = report["weighted avg"]

    table_data.append([
        "Accuracy",
        f"{accuracy:.3f}",
        "",
        "",
        ""
    ])

    table_data.append([
        "Macro Avg",
        f"{macro_avg['precision']:.3f}",
        f"{macro_avg['recall']:.3f}",
        f"{macro_avg['f1-score']:.3f}",
        f"{int(macro_avg['support'])}" if 'support' in macro_avg else ""
    ])

    table_data.append([
        "Weighted Avg",
        f"{weighted_avg['precision']:.3f}",
        f"{weighted_avg['recall']:.3f}",
        f"{weighted_avg['f1-score']:.3f}",
        f"{int(weighted_avg['support'])}" if 'support' in weighted_avg else ""
    ])

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc="center"
    )

    table.scale(1, 2)
    ax.set_title("Classification Report")

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    roc_fig, ax = plt.subplots(figsize=(6, 6))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-AUC Curve")
    ax.legend()
    ax.grid(True)

    return cm_fig, cr_fig, roc_fig

class ModelTrainer:
    def __init__(self,model : Classifier,train_set : ImageDataset,val_set : ImageDataset = None, batch_size=32,lr = 1e-3,device='cpu',return_fig=False, seed=None):
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        
        self.train_loader = DataLoader(
            train_set, 
            batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            generator=g 
        )
        
        self.device = device
        
        if val_set is not None:
            self.val_loader = DataLoader(
                val_set, 
                batch_size, 
                shuffle=False, 
                collate_fn=collate_fn,
                worker_init_fn=seed_worker
            )
        else:
            self.val_loader = None
        self.class_names = model.classes
        self.model = model
        self.lr = lr
        self.optim = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.optim.zero_grad()
        self.criterion = nn.CrossEntropyLoss()
        self.return_fig=return_fig
        self.best_model_state = None
        self.best_val_acc = 0.0
        self.interrupt=False
    
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
            h, w = f.shape[:2]

            dpi = 100
            fig_w = max(4, w / dpi)
            fig_h = max(4, h / dpi)
            fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
            ax = fig.add_subplot(111)
            ax.imshow(f)
            ax.axis("off")
            feature_figs.append(fig)
            

        all_figs = [fig_pred] + feature_figs
        if not self.return_fig:
            plt.show()
        plt.close(fig_pred)
        if self.return_fig:
            return all_figs
        else:
            return None


    def train_one_epoch(self,epoch):
        self.model.train()
        total_loss = 0
        train_pbar = tqdm(self.train_loader, desc="Training",leave=False)
        correct = 0
        total = 0
        for imgs, labels in train_pbar:
            if self.interrupt:
                break
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
            train_loss,train_acc = self.train_one_epoch(epoch)
            if self.interrupt:
                return
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            if self.val_loader is not None:
                val_loss,val_acc,fig=self.validate(epoch, visualize=(epoch % visualize_every == 0 or epoch == 1))
                if self.interrupt:
                    return
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                print(f"Epoch {epoch} Train Loss: {train_loss:.4f} | Train Acc : {train_acc:.4f} | Val Loss : {val_loss:.4f} | Val Acc : {val_acc:.4f}")
                if val_acc > self.best_val_acc:
                    print(f"New best model found at epoch {epoch} (Val Acc: {val_acc:.4f})")
                    self.best_val_acc = val_acc
                    self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

                yield train_loss,train_acc,val_loss,val_acc,fig
            else:
                print(f"Epoch {epoch} Train Loss: {train_loss:.4f} | Train Acc : {train_acc:.4f}")
                yield train_loss,train_acc,None,None,None
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Best model (Val Acc: {self.best_val_acc:.4f}) loaded into trainer.model")
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
                if self.interrupt:
                    break
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

        return avg_loss,acc,fig