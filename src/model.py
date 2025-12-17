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
    fc_num_layers=3
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

    # Sobel
    sobel_ksize=3

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
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channel=out_channel
            out_channel*=2
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
    def visualize(
        self,
        input_image,
        max_channels=8,
        couple=False,
        show=True,
        **kwargs
    ):
        self.eval()
        device = self.get_device()

        if isinstance(input_image, np.ndarray):
            x = torch.from_numpy(input_image).permute(2, 0, 1).float().unsqueeze(0).to(device)
        elif isinstance(input_image, torch.Tensor):
            x = input_image.unsqueeze(0).to(device) if input_image.ndim == 3 else input_image.to(device)
        else:
            raise TypeError("input_image must be np.ndarray or torch.Tensor")

        conv_layers = [
            (name, module)
            for name, module in self.named_modules()
            if isinstance(module, nn.Conv2d)
        ]

        all_layer_images = []

        for name, layer in conv_layers:
            activations = []

            def hook_fn(module, input, output):
                activations.append(output.detach().cpu())

            handle = layer.register_forward_hook(hook_fn)
            _ = self(x)
            handle.remove()

            act = activations[0][0]  # (C, H, W)
            C, H, W = act.shape

            # --------------------------------------------------
            # COUPLED RGB VISUALIZATION
            # --------------------------------------------------
            if couple:
                max_rgb = max_channels // 3
                num_rgb = min(C // 3, max_rgb)
                rem = min(C - num_rgb * 3, max_channels - num_rgb * 3)

                total_tiles = num_rgb + rem
                cols = min(4, total_tiles)
                rows = int(np.ceil(total_tiles / cols))

                fig, axes = plt.subplots(
                    rows, cols,
                    figsize=(3 * cols, 3 * rows)
                )

                axes = np.atleast_2d(axes)

                tile_idx = 0

                # ---------------------------
                # RGB COUPLED CHANNELS
                # ---------------------------
                for i in range(num_rgb):
                    r = tile_idx // cols
                    c = tile_idx % cols

                    rgb = act[i*3:(i+1)*3].clone()

                    for ch in range(3):
                        v = rgb[ch]
                        rgb[ch] = (v - v.min()) / (v.max() - v.min() + 1e-8)

                    rgb = rgb.permute(1, 2, 0).numpy()

                    axes[r, c].imshow(rgb)
                    axes[r, c].axis("off")
                    axes[r, c].set_title(f"RGB {i*3}-{i*3+2}", fontsize=9)

                    tile_idx += 1

                start = num_rgb * 3
                for j in range(rem):
                    r = tile_idx // cols
                    c = tile_idx % cols

                    ch = act[start + j]
                    ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)

                    axes[r, c].imshow(ch, cmap="gray")
                    axes[r, c].axis("off")
                    axes[r, c].set_title(f"Ch {start + j}", fontsize=9)

                    tile_idx += 1

                for idx in range(tile_idx, rows * cols):
                    r = idx // cols
                    c = idx % cols
                    axes[r, c].axis("off")

                fig.suptitle(f"Layer: {name} (Coupled RGB + Grayscale)", fontsize=14)
                plt.tight_layout()

            # --------------------------------------------------
            # STANDARD GRAYSCALE VISUALIZATION
            # --------------------------------------------------
            else:
                num_channels = min(C, max_channels)
                cols = min(8, num_channels)
                rows = int(np.ceil(num_channels / cols))

                fig, axes = plt.subplots(
                    rows, cols,
                    figsize=(3 * cols, 3 * rows)
                )

                axes = np.atleast_2d(axes)

                for idx in range(num_channels):
                    r = idx // cols
                    c = idx % cols
                    axes[r, c].imshow(act[idx], cmap="gray")
                    axes[r, c].axis("off")

                for idx in range(num_channels, rows * cols):
                    r = idx // cols
                    c = idx % cols
                    axes[r, c].axis("off")

                fig.suptitle(f"Layer: {name}", fontsize=14)
                plt.tight_layout()

            if show:
                plt.show()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
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
        self.device = 'cpu'

    def get_device(self):
        return next(self.parameters()).device if len(list(self.parameters())) > 0 else self.device


    def extract_features(self, img,visualize=False,**kwargs):
        cfg = self.config

        # Convert to grayscale
        gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        for _ in range(self.num_downsample):
            gray = cv2.pyrDown(gray)

        gray = cv2.GaussianBlur(gray, cfg.gaussian_ksize, sigmaX=cfg.gaussian_sigmaX, sigmaY=cfg.gaussian_sigmaY)
        valid_H, valid_W = gray.shape[:2]
        
        def render_subplots(items, max_cols=8, figsize_per_cell=3):
            n = len(items)
            cols = min(max_cols, n)
            rows = int(np.ceil(n / cols))

            fig, axes = plt.subplots(
                rows, cols,
                figsize=(cols * figsize_per_cell, rows * figsize_per_cell)
            )

            axes = np.atleast_2d(axes)

            for idx, (img, title, cmap) in enumerate(items):
                r = idx // cols
                c = idx % cols
                ax = axes[r, c]
                ax.imshow(img, cmap=cmap)
                ax.set_title(title, fontsize=9)
                ax.axis("off")

            # Hide unused axes
            for idx in range(n, rows * cols):
                r = idx // cols
                c = idx % cols
                axes[r, c].axis("off")

            plt.tight_layout()
            return fig

        feature_list = []
        vis_items=[]
        # figs = []
        H, W = gray.shape
        cell_h, cell_w = cfg.hog_pixels_per_cell
        block_h, block_w = cfg.hog_cells_per_block

        min_h = cell_h * block_h
        min_w = cell_w * block_w
        use_hog = (H > 2*min_h) and (W > 2*min_w)
        # 1. HOG
        if use_hog:
            hog_descriptors, hog_image = hog(
                gray,
                orientations=cfg.hog_orientations,
                pixels_per_cell=cfg.hog_pixels_per_cell,
                cells_per_block=cfg.hog_cells_per_block,
                block_norm=cfg.hog_block_norm,
                visualize=True,
                feature_vector=False
            )

            hog_cells = hog_descriptors.mean(axis=(2, 3))
            
            cell_h, cell_w = cfg.hog_pixels_per_cell
            hog_pixel = np.repeat(
                np.repeat(hog_cells, cell_h, axis=0),
                cell_w, axis=1
            )
            hog_pixel = hog_pixel[:gray.shape[0], :gray.shape[1]]
            hog_energy = np.sum(hog_pixel, axis=2)
            dominant_bin = np.argmax(hog_pixel, axis=2)
            dominant_strength = np.max(hog_pixel, axis=2)
            dominant_weighted = dominant_bin * dominant_strength
            valid_H, valid_W = hog_pixel.shape[:2]
            if visualize:
                # figs.append(plot_feature(hog_energy, "HOG Energy"))
                # figs.append(plot_feature(dominant_bin, "HOG Dominant Bin",cmap='hsv'))
                # figs.append(plot_feature(dominant_weighted, "HOG Weighted Dominant Bin"))
                # figs.append(plot_feature(hog_image[:valid_H, :valid_W], f"HoG"))
                vis_items.append((hog_energy, "HOG Energy",'gray'))
                vis_items.append((dominant_bin, "HOG Dominant Bin",'hsv'))
                vis_items.append((dominant_weighted, "HOG Weighted Dominant Bin",'gray'))
                vis_items.append((hog_image[:valid_H, :valid_W], f"HoG",'gray'))
            for b in range(hog_pixel.shape[2]):
                feature_list.append(hog_pixel[:, :, b])
        
        
        # 2. Canny edges
        edges = cv2.Canny(gray, cfg.canny_low, cfg.canny_high) / 255.0
        # feature_list.append(edges.ravel())
        feature_list.append(edges[:valid_H, :valid_W])
        if visualize:
            # figs.append(plot_feature(edges[:valid_H, :valid_W], "Canny Edge"))
            vis_items.append((edges[:valid_H, :valid_W], "Canny Edge", "gray"))
        # 3. Harris corners
        harris = cv2.cornerHarris(gray, blockSize=cfg.harris_block_size, ksize=cfg.harris_ksize, k=cfg.harris_k)
        harris = cv2.dilate(harris, None)
        harris = np.clip(harris, 0, 1)
        # feature_list.append(harris.ravel())
        feature_list.append(harris[:valid_H, :valid_W])
        if visualize:
            # figs.append(plot_feature(harris[:valid_H, :valid_W], "Harris Corner"))
            vis_items.append((harris[:valid_H, :valid_W], "Harris Corner", "gray"))
        # # 4. Shi-Tomasi corners
        # shi_corners = np.zeros_like(gray, dtype=np.float32)
        # keypoints = cv2.goodFeaturesToTrack(gray, maxCorners=cfg.shi_max_corners, qualityLevel=cfg.shi_quality_level, minDistance=cfg.shi_min_distance)
        # if keypoints is not None:
        #     for kp in keypoints:
        #         x, y = kp.ravel()
        #         shi_corners[int(y), int(x)] = 1.0
        # # feature_list.append(shi_corners.ravel())
        # feature_list.append(shi_corners[:valid_H, :valid_W])
        # if visualize:
        #     figs.append(plot_feature(shi_corners[:valid_H, :valid_W], "Shi-Tomasi Corner"))
        # 5. LBP
        lbp = local_binary_pattern(gray, P=cfg.lbp_P, R=cfg.lbp_R, method='uniform')
        lbp = lbp / lbp.max() if lbp.max() != 0 else lbp
        # feature_list.append(lbp.ravel())
        feature_list.append(lbp[:valid_H, :valid_W])
        if visualize:
            # figs.append(plot_feature(lbp[:valid_H, :valid_W], "LBP"))
            vis_items.append((lbp[:valid_H, :valid_W], "LBP", "gray"))
        # 6. Gabor filter
        # g_kernel = cv2.getGaborKernel((cfg.gabor_ksize, cfg.gabor_ksize), cfg.gabor_sigma, cfg.gabor_theta, cfg.gabor_lambda, cfg.gabor_gamma)
        # gabor_feat = cv2.filter2D(gray, cv2.CV_32F, g_kernel)
        # gabor_feat = (gabor_feat - gabor_feat.min()) / (gabor_feat.max() - gabor_feat.min() + 1e-8)
        # # feature_list.append(gabor_feat.ravel())
        # feature_list.append(gabor_feat[:valid_H, :valid_W])
        # if visualize:
        #     figs.append(plot_feature(gabor_feat[:valid_H, :valid_W], "Gabor Filter"))

        for theta in [0, np.pi/4, np.pi/2]:
            kernel = cv2.getGaborKernel(
                (cfg.gabor_ksize, cfg.gabor_ksize),
                cfg.gabor_sigma, theta,
                cfg.gabor_lambda, cfg.gabor_gamma
            )
            g = cv2.filter2D(gray, cv2.CV_32F, kernel)
            g = np.abs(g)
            g /= g.max() + 1e-8
            feature_list.append(g[:valid_H, :valid_W])
            if visualize:
                # figs.append(plot_feature(g[:valid_H, :valid_W], "Gabor Filter"))
                vis_items.append((g[:valid_H, :valid_W], f"Gabor θ={theta:.2f}", "gray"))
        # 7. Sobel
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=cfg.sobel_ksize)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=cfg.sobel_ksize)

        sobelx = np.abs(sobelx)
        sobely = np.abs(sobely)

        sobelx /= sobelx.max() + 1e-8
        sobely /= sobely.max() + 1e-8

        feature_list.append(sobelx[:valid_H, :valid_W])
        feature_list.append(sobely[:valid_H, :valid_W])
        if visualize:
            # figs.append(plot_feature(sobelx[:valid_H, :valid_W], "Sobel X"))
            # figs.append(plot_feature(sobely[:valid_H, :valid_W], "Sobel Y"))
            vis_items.append((sobelx[:valid_H, :valid_W], "Sobel X",'gray'))
            vis_items.append((sobely[:valid_H, :valid_W], "Sobel Y",'gray'))
        # 8. Laplacian
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap = np.abs(lap)
        lap /= lap.max() + 1e-8

        feature_list.append(lap[:valid_H, :valid_W])

        if visualize:
            # figs.append(plot_feature(lap[:valid_H, :valid_W], "Laplacian"))
            vis_items.append((lap[:valid_H, :valid_W], "Laplacian",'gray'))

        # 9. Gradient Magnitude
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=cfg.sobel_ksize)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=cfg.sobel_ksize)

        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_mag /= grad_mag.max() + 1e-8

        feature_list.append(grad_mag[:valid_H, :valid_W])

        if visualize:
            # figs.append(plot_feature(grad_mag[:valid_H, :valid_W], "Gradient Magnitude"))
            vis_items.append((grad_mag[:valid_H, :valid_W], "Gradient Magnitude",'gray'))

        # Stack all features along channel axis
        features = np.stack(feature_list, axis=0)
        # features = np.concatenate(feature_list).astype(np.float32)
        if visualize:
            return features.astype(np.float32),[render_subplots(vis_items, max_cols=8)]
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
        batch_features = torch.from_numpy(batch_features).float().to(self.get_device())
        return batch_features
    
    def visualize(self, img, show_original=True,show=True):
        if img.ndim != 3 or img.shape[2] != 3:
            img = np.repeat(img[:, :, None], 3, axis=2)

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
        feature_stack,figs = self.extract_features(img,visualize=True)
        if show:
            plt.show()      
        for fig in figs:
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
        out_features=256
        for i in range(config.fc_num_layers):
            layers.append(nn.Linear(in_features,out_features))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_features=out_features
            out_features=out_features // 2
            if out_features <= num_classes:
                break
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