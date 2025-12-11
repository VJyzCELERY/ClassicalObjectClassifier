import torch
import torch.nn as nn
import cv2
import numpy as np
from dataclasses import dataclass
from skimage.feature import hog,local_binary_pattern
import matplotlib.pyplot as plt
import os
import io
from PIL import Image

@dataclass
class Config:
    img_size=(256,256)
    in_channels=3
    fc_hidden_dim=3
    conv_hidden_dim=3
    conv_kernel_size=3
    dropout=0.2
    classical_downsample=1
    # HOG
    hog_orientations = 9
    hog_pixels_per_cell = (16, 16)
    hog_cells_per_block = (2, 2)
    hog_block_norm = 'L2-Hys'

    # Canny
    canny_sigma = 1.0
    canny_low = 100
    canny_high = 200

    # Gaussian
    gaussian_ksize = (3, 3)
    gaussian_sigmaX = 1.0
    gaussian_sigmaY = 1.0

    # Harris corners
    harris_block_size = 2
    harris_ksize = 3
    harris_k = 0.04

    # Shi-Tomasi corners
    shi_max_corners = 100
    shi_quality_level = 0.01
    shi_min_distance = 10

    # LBP
    lbp_P = 8 
    lbp_R = 1  

    # Gabor filters
    gabor_ksize = 21
    gabor_sigma = 5
    gabor_theta = 0
    gabor_lambda = 10
    gabor_gamma = 0.5

class CNNFeatureExtractor(nn.Module):
    def __init__(self,config : Config):
        super().__init__()
        layers = []
        self.in_channels = config.in_channels
        in_channel = config.in_channels
        self.img_size = config.img_size
        out_channel = 32
        for i in range(config.conv_hidden_dim):
            layers.append(nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=config.conv_kernel_size,stride=1,padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channel=out_channel
            out_channel*=2
        layers.append(nn.Dropout(config.dropout))
        self.layers = nn.Sequential(*layers)
    def get_device(self):
        return next(self.parameters()).device
    def forward(self,x):
        if isinstance(x, list):
            if isinstance(x[0], np.ndarray):
                x = np.stack(x, axis=0) 
        if isinstance(x,np.ndarray):
            if len(x.shape) == 2:
                x = x[:, :, None]  
                x = np.expand_dims(x, 0)
                x = x.transpose(2, 0, 1)  
            elif len(x.shape) == 3:
                x = x.transpose(2, 0, 1)
                x = np.expand_dims(x, 0)
            elif x.ndim == 4:
                x = x.transpose(0, 3, 1, 2) # Change to (B,C,H,W)
            x = torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            if x.ndim == 3:
                x = x.unsqueeze(0)
        x=x.to(self.get_device())
        return self.layers(x) # Always expects (B,C,H,W)
    def output(self):
        self.eval()

        with torch.no_grad():
            x = torch.zeros(
                (1, self.in_channels, self.img_size[1], self.img_size[0]),
                device=self.get_device()
            )

            out = self(x)

        return out
    def visualize(self, input_image, max_channels=8,show=True):
        self.eval()
        device = self.get_device()

        if isinstance(input_image, np.ndarray):
            x = torch.from_numpy(input_image).permute(2, 0, 1).float().unsqueeze(0).to(device)  # HWC -> CHW -> B
        elif isinstance(input_image, torch.Tensor):
            x = input_image.unsqueeze(0).to(device) if input_image.ndim == 3 else input_image.to(device)
        else:
            raise TypeError("input_image must be np.ndarray or torch.Tensor")

        conv_layers = [(name, module) for name, module in self.named_modules() if isinstance(module, nn.Conv2d)]
        all_layer_images = []

        for name, layer in conv_layers:
            activations = []

            def hook_fn(module, input, output):
                activations.append(output.cpu().detach())

            handle = layer.register_forward_hook(hook_fn)
            _ = self(x)
            handle.remove()

            act = activations[0][0]
            num_channels = min(act.shape[0], max_channels)

            fig, axes = plt.subplots(1, num_channels, figsize=(15, 3))
            if num_channels == 1:
                axes = [axes]

            for i in range(num_channels):
                axes[i].imshow(act[i], cmap='gray')
                axes[i].axis('off')

            fig.suptitle(f'Layer: {name}', fontsize=14)
            if show:
                plt.show()

            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            all_layer_images.append(np.array(img))
            plt.close(fig) 
        return all_layer_images
    
class ClassicalFeatureExtractor(nn.Module):
    def __init__(self, config : Config):
        super().__init__()
        self.img_size = config.img_size  # (H, W)
        self.hog_orientations = config.hog_orientations
        self.num_downsample = config.classical_downsample
        self.config = config
        self.feature_names = ['HoG','Canny Edge','Harris Corner','Shi-Tomasi corners','LBP','Gabor Filters']
        self.device = 'cpu'

    def get_device(self):
        return next(self.parameters()).device if len(list(self.parameters())) > 0 else self.device


    def extract_features(self, img):
        cfg = self.config

        # Convert to grayscale
        min_h = cfg.hog_pixels_per_cell[0] * cfg.hog_cells_per_block[0]
        min_w = cfg.hog_pixels_per_cell[1] * cfg.hog_cells_per_block[1]
        gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        for _ in range(self.num_downsample):
            h, w = gray.shape
            if h <= min_h or w <= min_w:
                break
            gray = cv2.pyrDown(gray)

        gray = cv2.GaussianBlur(gray, cfg.gaussian_ksize, sigmaX=cfg.gaussian_sigmaX, sigmaY=cfg.gaussian_sigmaY)

        feature_list = []

        # 1. HOG
        _, hog_image = hog(
            gray,
            orientations=cfg.hog_orientations,
            pixels_per_cell=cfg.hog_pixels_per_cell,
            cells_per_block=cfg.hog_cells_per_block,
            block_norm=cfg.hog_block_norm,
            visualize=True
        )
        feature_list.append(hog_image)

        # 2. Canny edges
        edges = cv2.Canny(gray, cfg.canny_low, cfg.canny_high) / 255.0
        feature_list.append(edges)

        # 3. Harris corners
        harris = cv2.cornerHarris(gray, blockSize=cfg.harris_block_size, ksize=cfg.harris_ksize, k=cfg.harris_k)
        harris = cv2.dilate(harris, None)
        harris = np.clip(harris, 0, 1)
        feature_list.append(harris)

        # 4. Shi-Tomasi corners
        shi_corners = np.zeros_like(gray, dtype=np.float32)
        keypoints = cv2.goodFeaturesToTrack(gray, maxCorners=cfg.shi_max_corners, qualityLevel=cfg.shi_quality_level, minDistance=cfg.shi_min_distance)
        if keypoints is not None:
            for kp in keypoints:
                x, y = kp.ravel()
                shi_corners[int(y), int(x)] = 1.0
        feature_list.append(shi_corners)

        # 5. LBP
        lbp = local_binary_pattern(gray, P=cfg.lbp_P, R=cfg.lbp_R, method='uniform')
        lbp = lbp / lbp.max() if lbp.max() != 0 else lbp
        feature_list.append(lbp)

        # 6. Gabor filter
        g_kernel = cv2.getGaborKernel((cfg.gabor_ksize, cfg.gabor_ksize), cfg.gabor_sigma, cfg.gabor_theta, cfg.gabor_lambda, cfg.gabor_gamma)
        gabor_feat = cv2.filter2D(gray, cv2.CV_32F, g_kernel)
        gabor_feat = (gabor_feat - gabor_feat.min()) / (gabor_feat.max() - gabor_feat.min() + 1e-8)
        feature_list.append(gabor_feat)

        # Stack all features along channel axis
        features = np.stack(feature_list, axis=2)
        return features.astype(np.float32)


    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(x, np.ndarray):
            if x.ndim == 3:
                x = np.expand_dims(x, 0)
            elif x.ndim != 4:
                raise ValueError(f"Expected input of shape HWC or BHWC, got {x.shape}")
        elif isinstance(x, list):
            x = np.stack(x, axis=0)

        batch_features = []
        for img in x:
            if img.ndim != 3 or img.shape[2] != 3:
                img = np.repeat(img[:, :, None], 3, axis=2)
            feat = self.extract_features(img)
            batch_features.append(feat)
        batch_features = np.stack(batch_features, axis=0)
        return torch.from_numpy(batch_features).float().to(self.get_device())
    
    def visualize(self, img, show_original=True,show=True):
        if img.ndim != 3 or img.shape[2] != 3:
            img = np.repeat(img[:, :, None], 3, axis=2)

        feature_stack = self.extract_features(img)
        num_channels = feature_stack.shape[2]

        outputs = []  

        def fig_to_pil(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)

            pil_img = Image.open(buf).copy()

            buf.close()
            plt.close(fig)

            return pil_img

        if show_original:
            fig = plt.figure(figsize=(4, 4))
            plt.imshow(img)
            plt.title("Original")
            plt.axis("off")
            if show:
                plt.show()                      
            outputs.append(fig_to_pil(fig)) 

        for c in range(num_channels):
            fig = plt.figure(figsize=(4, 4))

            plt.imshow(feature_stack[:, :, c], cmap="gray")
            plt.title(f"Feature {self.feature_names[c]}")
            plt.axis("off")
            if show:
                plt.show()                      
            outputs.append(fig_to_pil(fig)) 

        return outputs


    def output(self):
        """Return dummy output to compute in_features for FC head"""
        dummy_img = np.zeros((1, self.img_size[1],self.img_size[0], 3), dtype=np.float32)
        feat = self.forward(dummy_img)
        return feat



class FullyConnectedHead(nn.Module):
    def __init__(self,in_features,classes,config:Config):
        super().__init__()
        num_classes = len(classes)
        self.classes = classes
        layers = []
        out_features=512
        for i in range(config.fc_hidden_dim):
            layers.append(nn.Linear(in_features,out_features))
            layers.append(nn.ReLU())
            in_features=out_features
            out_features=out_features // 2
        layers.append(nn.Linear(in_features,num_classes))
        self.layers = nn.Sequential(*layers)
    def get_device(self):
        return next(self.parameters()).device
    def forward(self,x : torch.Tensor):
        x=x.to(self.get_device())
        return self.layers(x)
    
class Classifier(nn.Module):
    def __init__(self,backbone,classes,config : Config):
        super().__init__()
        self.config=config
        self.classes=classes
        self.backbone = backbone
        self.flatten = nn.Flatten()
        feat = backbone.output()
        flat = self.flatten(feat)
        in_features = flat.shape[1]
        self.head = FullyConnectedHead(in_features,classes,config)
    def get_device(self):
        return next(self.parameters()).device
    
    @torch.no_grad()
    def predict(self, x):
        self.eval()
        target_size = self.config.img_size
        x = cv2.resize(x, target_size)
        logits = self.forward(x)    
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

        return self.classes[pred_idx]

    def forward(self,x):
        feat = self.backbone(x)
        feat = self.flatten(feat)
        return self.head(feat)
    def visualize_feature(self,img,return_img=True,**kwargs):
        target_size = self.config.img_size
        img = cv2.resize(img, target_size)
        if return_img:
            return self.backbone.visualize(img,**kwargs)
        else:
            self.backbone.visualize(img,**kwargs)
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'classes': self.classes,
            'config': self.config
        }, path)
        print(f"Model saved to {path}")

@staticmethod
def load(path: str, backbone_class, device='cpu'):
    checkpoint = torch.load(path, map_location=device,weights_only=False)
    config = checkpoint['config']
    classes = checkpoint['classes']
    backbone = backbone_class(config).to(device)
    model = Classifier(backbone, classes, config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {path}")
    return model