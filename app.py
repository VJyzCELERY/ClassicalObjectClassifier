import gradio as gr
import zipfile
import os
import torch
from src.dataloader import ImageDataset,collate_fn,AugmentedSubset,simple_augment
from src.model import Classifier,Config,CNNFeatureExtractor,ClassicalFeatureExtractor,load
from torch.utils.data import Subset
from src.trainer import ModelTrainer,model_evaluation
import torch
import os
import numpy as np
import time
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def unzip_dataset(zip_file):
    base_name = os.path.splitext(os.path.basename(zip_file.name))[0]
    dataset_path = os.path.join(".", base_name)
    
    os.makedirs(dataset_path, exist_ok=True)
    
    with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)
        extracted_items = os.listdir(dataset_path)
        if len(extracted_items) == 1 and os.path.isdir(os.path.join(dataset_path, extracted_items[0])):
            dataset_path = os.path.join(dataset_path, extracted_items[0])
    
    print(f"Dataset extracted to: {dataset_path}")
    class_names = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"Detected classes: {class_names}")
    
    for cls in class_names:
        cls_path = os.path.join(dataset_path, cls)
        images = os.listdir(cls_path)
        print(f"Class '{cls}' has {len(images)} images. Sample: {images[:3]}")
    
    return dataset_path

cnn_history={
    "train_acc":[],
    "train_loss":[],
    "val_acc":[],
    "val_loss":[]
}

classic_history={
    "train_acc":[],
    "train_loss":[],
    "val_acc":[],
    "val_loss":[]
}

training_interrupt = False

def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img_array = np.array(img)
    plt.close(fig) 
    return img_array

def plot(datas, labels, xlabel, ylabel, title, figsize=(16, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    for data, label in zip(datas, labels):
        ax.plot(range(1, len(data) + 1), data, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig_to_image(fig)

class TrainingInterrupted(Exception):
    pass
cnntrainer=None
classictrainer=None
def stop_training():
    global training_interrupt
    training_interrupt = True
    if cnntrainer is not None:
        cnntrainer.interrupt=True
    if classictrainer is not None:
        classictrainer.interrupt=True
    return "Training stopped."



def train(cnn,classic,train_set,val_set,batch_size,lr,epochs,device="cpu",visualize_every=5):
    global training_interrupt
    global cnntrainer,classictrainer
    training_interrupt = False
    global cnn_history
    global classic_history
    cnn_done=False
    cnn_history={
        "train_acc":[],
        "train_loss":[],
        "val_acc":[],
        "val_loss":[]
    }

    classic_history={
        "train_acc":[],
        "train_loss":[],
        "val_acc":[],
        "val_loss":[]
    }
    try:
        if training_interrupt:
            raise TrainingInterrupted("Training was interrupted!")
        cnntrainer = ModelTrainer(cnn,train_set,val_set,batch_size,lr,device=device,return_fig=True)
        classictrainer = ModelTrainer(classic,train_set,val_set,batch_size,lr,device=device,return_fig=True)
        cnn_text=""
        classic_text=""
        cnn_fig=None
        all_cnn_fig = []
        all_classic_fig= []
        classic_fig=None
        start_time = time.time()
        for i,(cnn_train_loss,cnn_train_acc,cnn_val_loss,cnn_val_acc,cnn_fig) in enumerate(cnntrainer.train(epochs,visualize_every=visualize_every)):
            if training_interrupt:
                raise TrainingInterrupted("Training was interrupted!")
            if i == epochs:
                break
            if cnn_fig is not None:
                for fig in cnn_fig:
                    fig.suptitle(f"Epoch {i+1}", fontsize=16)
                    all_cnn_fig.append(fig_to_image(fig))
            cnn_text+= f"Epochs {i+1} : Train Loss: {cnn_train_loss:.4f}, Train Acc: {cnn_train_acc:.4f}, Val Loss: {cnn_val_loss:.4f}, Val Acc: {cnn_val_acc:.4f}\n"
            cnn_history['train_acc'].append(cnn_train_acc)
            cnn_history['train_loss'].append(cnn_train_loss)
            cnn_history['val_acc'].append(cnn_val_acc)
            cnn_history['val_loss'].append(cnn_val_loss)
            
            yield cnn_text,all_cnn_fig,classic_text,all_classic_fig,cnn_done
        cnn_done=True
        dt = time.time()-start_time
        cnn_fig=None
        cnn_text+=f'Time taken : {dt:.2f} seconds\n'
        yield cnn_text,all_cnn_fig,classic_text,all_classic_fig,cnn_done
        start_time = time.time()
        for i,(classic_train_loss,classic_train_acc,classic_val_loss,classic_val_acc,classic_fig) in enumerate(classictrainer.train(epochs,visualize_every=visualize_every)):
            if training_interrupt:
                raise TrainingInterrupted("Training was interrupted!")
            if i == epochs:
                break
            if classic_fig is not None:
                for fig in classic_fig:
                    fig.suptitle(f"Epoch {i+1}", fontsize=16)
                    all_classic_fig.append(fig_to_image(fig))
            classic_history['train_acc'].append(classic_train_acc)
            classic_history['train_loss'].append(classic_train_loss)
            classic_history['val_acc'].append(classic_val_acc)
            classic_history['val_loss'].append(classic_val_loss)
            classic_text+= f"Epochs {i+1} : Train Loss: {classic_train_loss:.4f}, Train Acc: {classic_train_acc:.4f}, Val Loss: {classic_val_loss:.4f}, Val Acc: {classic_val_acc:.4f}\n"
            yield cnn_text,all_cnn_fig,classic_text,all_classic_fig,cnn_done
        dt = time.time()-start_time
        classic_fig=None
        classic_text+=f'Time taken : {dt:.2f} seconds\n'
        yield cnn_text,all_cnn_fig,classic_text,all_classic_fig,cnn_done
    except TrainingInterrupted as e:
        print(e)
        return

def train_model(zip_file,batch_size,lr,epochs,seed,vis_every,use_augment,
                img_width,img_height,fc_num_layers,
                in_channels,conv_hidden_dim,dropout,
                classical_downsample,
                # hog_orientations,hog_pixels_per_cell,hog_cells_per_block,hog_block_norm,
                canny_sigma,canny_low,canny_high,
                gaussian_ksize,gaussian_sigmaX,gaussian_sigmaY,
                harris_block_size,harris_ksize,harris_k,
                lbp_P,lbp_R,
                gabor_ksize,gabor_sigma,gabor_theta,gabor_lambda,gabor_gamma,
                sobel_ksize):
    config = Config()
    global training_interrupt
    training_interrupt = False
    BATCH_SIZE = batch_size
    DATASET_PATH = unzip_dataset(zip_file)
    SEED = seed
    EPOCHS = epochs
    LR = lr
    config.img_size = (int(img_width),int(img_height))
    config.fc_num_layers = int(fc_num_layers)
    # CNN Config
    config.in_channels = int(in_channels)
    config.conv_hidden_dim=int(conv_hidden_dim)
    config.dropout=dropout
    # Classical Config
    config.classical_downsample=int(classical_downsample)
    # config.hog_orientations=int(hog_orientations)
    # config.hog_pixels_per_cell=(int(hog_pixels_per_cell),int(hog_pixels_per_cell))
    # config.hog_cells_per_block=(int(hog_cells_per_block),int(hog_cells_per_block))
    # config.hog_block_norm=hog_block_norm
    config.canny_sigma=int(canny_sigma)
    config.canny_low=canny_low
    config.canny_high=canny_high
    config.gaussian_ksize=(int(gaussian_ksize),int(gaussian_ksize))
    config.gaussian_sigmaX=gaussian_sigmaX
    config.gaussian_sigmaY=gaussian_sigmaY
    config.harris_block_size=int(harris_block_size)
    config.harris_ksize=int(harris_ksize)
    config.harris_k=harris_k
    config.lbp_P=int(lbp_P)
    config.lbp_R=int(lbp_R)
    config.gabor_ksize=int(gabor_ksize)
    config.gabor_sigma=int(gabor_sigma)
    config.gabor_theta=int(gabor_theta)
    config.gabor_lambda=int(gabor_lambda)
    config.gabor_gamma=gabor_gamma
    config.sobel_ksize=int(sobel_ksize)
    cnn_history_plots=[]
    classical_history_plots=[]
    cnn_plotted=False
    try:
        dataset = ImageDataset(DATASET_PATH,config.img_size)
        labels = [item['id'] for item in dataset.data]
        classes_name = dataset.classes
        train_idx, validation_idx = train_test_split(np.arange(len(dataset)),
                                                test_size=0.2,
                                                random_state=SEED,
                                                shuffle=True,
                                                stratify=labels)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, validation_idx)
        if use_augment:
            train_dataset = AugmentedSubset(train_dataset,simple_augment)
            val_dataset = AugmentedSubset(val_dataset,None)
        else:
            train_dataset = AugmentedSubset(train_dataset,None)
            val_dataset = AugmentedSubset(val_dataset,None)
        cnnbackbone = CNNFeatureExtractor(config).to(device)
        cnnmodel = Classifier(cnnbackbone,train_dataset.dataset.classes,config).to(device)
        classicbackbone = ClassicalFeatureExtractor(config)
        classicmodel = Classifier(classicbackbone,train_dataset.dataset.classes,config).to(device)
        for cnn_text,cnn_fig,classic_text,classic_fig,cnn_done in train(cnnmodel,classicmodel,train_dataset,val_dataset,BATCH_SIZE,LR,EPOCHS,device,visualize_every=vis_every):
            if cnn_done and not cnn_plotted:
                cnn_plotted=True
                cnn_history_plots.append(plot([cnn_history['train_acc'],cnn_history['val_acc']],['Training Accuracy','Validation Accuracy'],'Epochs','Accuracy (%)','Training vs Validation Accuracy'))
                cnn_history_plots.append(plot([cnn_history['train_loss'],cnn_history['val_loss']],['Training Loss','Validation Loss'],'Epochs','Loss','Training vs Validation Loss'))
                cm,cr,roc = model_evaluation(cnnmodel,val_dataset,device,BATCH_SIZE,0,classes_name)
                cnn_history_plots.append(fig_to_image(cm))
                cnn_history_plots.append(fig_to_image(cr))
                cnn_history_plots.append(fig_to_image(roc))
            yield cnn_text,cnn_fig,classic_text,classic_fig,cnn_history_plots,classical_history_plots
        classical_history_plots.append(plot([classic_history['train_acc'],classic_history['val_acc']],['Training Accuracy','Validation Accuracy'],'Epochs','Accuracy (%)','Training vs Validation Accuracy'))
        classical_history_plots.append(plot([classic_history['train_loss'],classic_history['val_loss']],['Training Loss','Validation Loss'],'Epochs','Loss','Training vs Validation Loss'))
        cm,cr,roc = model_evaluation(classicmodel,val_dataset,device,BATCH_SIZE,0,classes_name)
        classical_history_plots.append(fig_to_image(cm))
        classical_history_plots.append(fig_to_image(cr))
        classical_history_plots.append(fig_to_image(roc))
        yield cnn_text,cnn_fig,classic_text,classic_fig,cnn_history_plots,classical_history_plots

    except RuntimeError as e:
        print(e)
        yield cnn_text,cnn_fig,classic_text,classic_fig,cnn_history_plots,classical_history_plots
        return
    finally:
        if os.path.exists(DATASET_PATH):
            shutil.rmtree(DATASET_PATH)
            print(f"Temporary dataset folder '{DATASET_PATH}' removed.")
        
    cnnmodel.save(os.path.join('trained_model','cnn_model.pt'))
    classicmodel.save(os.path.join('trained_model','classic_model.pt'))
    

intro_html = """
<div style="
    border-left:6px solid #2563eb;
    border-right:6px solid #2563eb;
    padding:16px;
    border-radius:8px;
    font-size:16px;
    line-height:1.6;
    text-align: justify;
    text-justify: inter-word;
">
    <h1 style="margin-top:0;">Welcome to the Object Classifier Playground!</h1>
    <p>
    Object Classification is a field of computer vision where we train computer to learn to classify or identify what a model is.
    In traditional Object Classification, the task usually consist of feature extraction and classification model.
    For feature extraction there has been several methods of extracting a feature using certain algorithm. These algorithm consist of algorithm such as Corner Detection, Edge Detection, Local Binary Pattern (LBP) or even Histogram of Gradient (HoG).
    There are a lot of means of feature extraction. After feature extraction, the feature will be passed to machine learning algorithm specifically classifier model.
    One such model is the SVM, k Nearest Neighbor or Naive Bayes which will learn to distinguish object categories based on said features.
    </p>
    <p>
    With the advancement of deep learning, object classification task has been significantly simplified. Now with deep learning, we barely use feature extraction algorithm anymore.
    The reason is not because feature extraction has became obsolete in deep learning, instead the process itself has become part of the learning process. With deep learning, we use a model called Convolutional Neural Network (CNN).
    A convolutional network consist of two main layers, the convolution layer and the fully connected layer. The convolution layer apply filter on the image with a filter usually called convolutional kernel where the value of each cells in the convolutional kernel is random initially.
    </p>
    <img src="https://raw.githubusercontent.com/VJyzCELERY/ClassicalObjectClassifier/refs/heads/main/assets/conv-illus.jpg"></img>
    <p>
    For more detail on how convolutional neural network work, you can refer to this <a href="https://viso.ai/deep-learning/convolution-operations/">link</a>.
    </p>
    <p>
    In reality, what this convolution operation does is extract features to be processed on for a machine learning or another deep learning model. The Convolution by itself does not result in an object classification directly. So even deep learning model such as CNN
    still does the traditional feature extraction then classification pipeline. However, the strength in this model is the convolution layer learns what weight it needs to use to get the best feature possible. Usually in a single convolution layer could result in tens or hundreds of feature channels.
    </p>
    <p>
    In this program, although we will not discuss too deep about what traditional feature extraction is nor the fully inner workings of CNN, we will instead have a playground to demonstrate what feature extraction both perform and how they differ from
    one and another.
    </p>
    <h2 style="margin-top:0;">The Model Architecture!</h2>
    <p>
    The model architecture used in this program will follow a CNN architecture where it will consist of Convolution layer and Fully Connected Layer as a classifier. However, we will instead make it so that the feature extraction layer or the convolution layer be replacable with a traditional feature extraction algorithm.
    This is done because in theory they should be able to perform just as well or a little worse as it is basically what Convolution Layer does as convolution layer is able to extract a lot more features and trainable and specific features.
    </p>
    <p>
    For more detail you can refer to : https://github.com/VJyzCELERY/ClassicalObjectClassifier which will include a paper to explain the code and it's method.
    </p>
    
</div>
"""

with gr.Blocks(title="Object Classifier Playground") as demo:
    with gr.Tab("Introduction"):
        gr.HTML(intro_html)
    with gr.Tab("Training"):
        with gr.Row():
            zip_file = gr.File(label='Upload Dataset in Zip',file_types=['.zip'],file_count='single',interactive=True)
            batch_size = gr.Number(value=32,label='Batch Size',interactive=True,precision=0)
            lr = gr.Number(value=1e-3,label='Learning Rate',interactive=True)
            epochs= gr.Number(value=20,label="Epochs",interactive=True,precision=0)
            seed=gr.Number(value=42,label='Seed',interactive=True,precision=0)
            vis_every=gr.Number(value=5,label='Visualize Validation Every (Epochs)',interactive=True,precision=0)
            use_augment = gr.Checkbox(value=True,label='Use data augmentation for train data')
        with gr.Row():
            img_width=gr.Number(value=128,label='Image Width',interactive=True,precision=0)
            img_height=gr.Number(value=128,label='Image Height',interactive=True,precision=0)
            fc_num_layers = gr.Number(value=3,label="Fully Connected Layer Depth",interactive=True,precision=0)
            dropout = gr.Slider(minimum=0,maximum=1,value=0.2,step=0.05,label='Fully Connected Layer Dropout',interactive=True)
        gr.Markdown("# CNN Feature Extractor Configuration")
        with gr.Accordion(label="CNN Settings",open=False):
            with gr.Row():
                in_channels = gr.Number(value=3,label='Input Color Channel Amount',interactive=True,precision=0)
                conv_hidden_dim = gr.Number(value=3,label='Conv Hidden Dim',interactive=True,precision=0)
        gr.Markdown("# Classical Feature Extractor Configuration")
        with gr.Accordion(label='Classical Feature Extractor Settings',open=False):
            with gr.Row():
                classical_downsample = gr.Number(value=1,label='Classical Extractor Downsampling Amount',interactive=True,precision=0)
            # Deprecated
            # with gr.Row():
            #     hog_orientations = gr.Number(value=9,label='HoG Orientations',interactive=True,precision=0)
            #     hog_pixels_per_cell = gr.Number(value=16,label='HoG pixels per cell',interactive=True,precision=0)
            #     hog_cells_per_block = gr.Number(value=2,label='HoG cells per block',interactive=True,precision=0)
            #     hog_block_norm = gr.Dropdown(['L2-Hys'],value='L2-Hys',label='HoG Block Normalization Method',interactive=True)
            with gr.Row():
                canny_sigma = gr.Number(value=1.0,label='Canny Sigma Value',interactive=True)
                canny_low = gr.Number(value=100,label='Canny Low Threshold',interactive=True,precision=0)
                canny_high = gr.Number(value=200,label='Canny High Threshold',interactive=True,precision=0)
            with gr.Row():
                gaussian_ksize = gr.Number(value=3,label='Gaussian Kernel Size',interactive=True,precision=0)
                gaussian_sigmaX = gr.Number(value=1.0,label='Gaussian Sigma X Value',interactive=True)
                gaussian_sigmaY = gr.Number(value=1.0,label='Gaussian Sigma Y Value',interactive=True)
            with gr.Row():
                harris_block_size = gr.Number(value=2,label='Harris Corner Block Size',interactive=True,precision=0)
                harris_ksize = gr.Number(value=3,label='Harris Corner Kernel Size',interactive=True,precision=0)
                harris_k = gr.Slider(minimum=0.01, maximum=0.1, value=0.04, step=0.005, label='Harris Corner K value',interactive=True)
            with gr.Row():
                lbp_P = gr.Number(value=8,label='LBP P Value',interactive=True,precision=0)
                lbp_R = gr.Number(value=1,label='LBP R Value',interactive=True,precision=0)
            with gr.Row():
                gabor_ksize  = gr.Number(value=21,label="Gabor Kernel Size",interactive=True,precision=0)
                gabor_sigma  = gr.Number(value=5,label="Gabor Sigma",interactive=True,precision=0)
                gabor_theta  = gr.Number(value=0,label="Gabor Theta",interactive=True,precision=0,info="This current does nothing")
                gabor_lambda = gr.Number(value=10,label="Gabor Lambda",interactive=True,precision=0)
                gabor_gamma  = gr.Number(value=0.5,label="Gabor Gamma",interactive=True)
            with gr.Row():
                sobel_ksize = gr.Number(value=3,label="Sobel Kernel Size",interactive=True,precision=0)
        with gr.Column():
            train_btn = gr.Button("Train Model",variant='secondary',interactive=True)
            stop_btn = gr.Button("Stop Training")

        with gr.Column():
            with gr.Column():
                gr.Markdown("### CNN Training Log")
                cnn_log = gr.Textbox(label="CNN Log", interactive=False)
                cnn_fig = gr.Gallery(label="CNN Batch Visualization",interactive=False,object_fit='fill',columns=1)
                cnn_plots = gr.Gallery(label="CNN Training Performance",interactive=False,object_fit='fill',columns=1)
            with gr.Column():
                gr.Markdown("### Classical Training Log")
                classical_log = gr.Textbox(label="Classical Log", interactive=False)
                classical_fig = gr.Gallery(label="Classical Batch Visualization",interactive=False,object_fit='fill',columns=1)
                classical_plots = gr.Gallery(label="CNN Training Performance",interactive=False,object_fit='fill',columns=1)
        stop_btn.click(fn=stop_training, inputs=[], outputs=[])
        train_btn.click(fn=train_model,
                        inputs=[zip_file,batch_size,lr,epochs,seed,vis_every,use_augment,
                            img_width,img_height,fc_num_layers,
                            in_channels,conv_hidden_dim,dropout,
                            classical_downsample,
                            # hog_orientations,hog_pixels_per_cell,hog_cells_per_block,hog_block_norm,
                            canny_sigma,canny_low,canny_high,
                            gaussian_ksize,gaussian_sigmaX,gaussian_sigmaY,
                            harris_block_size,harris_ksize,harris_k,
                            lbp_P,lbp_R,
                            gabor_ksize,gabor_sigma,gabor_theta,gabor_lambda,gabor_gamma,
                            sobel_ksize],
                        outputs=[cnn_log,cnn_fig,classical_log,classical_fig,cnn_plots,classical_plots]
                        )
    def make_figure_from_image(img):
        fig, ax = plt.subplots(figsize=(8,8)) 
        ax.imshow(img)                        
        ax.axis("off")

        plt.show()  

        return fig 
    def predict_image(upload,show_original,max_channels,cnn_couple_channel):
        img = cv2.cvtColor(cv2.imread(upload),cv2.COLOR_BGR2RGB)
        model_base_path = "./trained_model"
        classic_model_path =os.path.join(model_base_path,'classic_model.pt')
        cnn_model_path = os.path.join(model_base_path,'cnn_model.pt')
        os.makedirs(model_base_path,exist_ok=True)
        if os.path.exists(classic_model_path):
            classic_model = load(classic_model_path,ClassicalFeatureExtractor,device=device)
        else:
            return "No Classical Model trained",None,None,None
        if os.path.exists(cnn_model_path):
            cnn_model = load(cnn_model_path,CNNFeatureExtractor,device=device)
        else:
            return "No CNN Model trained",None,None,None
        cnn_predict = cnn_model.predict(img)
        classic_predict = classic_model.predict(img)
        cnn_features = cnn_model.visualize_feature(img,max_channels=max_channels,couple=cnn_couple_channel)
        classical_features = classic_model.visualize_feature(img,show_original=show_original)
        return None,make_figure_from_image(img),cnn_predict,classic_predict,cnn_features,classical_features
    
    with gr.Tab("Inference"):
        with gr.Row():
            image_upload = gr.File(file_count='single',file_types=['image'],label='Upload Image to Infer',interactive=True)
        with gr.Column():
            gr.Markdown("# CNN Settings")
            with gr.Accordion(open=False):
                cnn_max_channel_visual = gr.Number(value=8,precision=0,label='Max CNN Channels to Preview',interactive=True)
                cnn_couple_channel = gr.Checkbox(value=False,label='Couple Channels into RGB')
        with gr.Column():
            gr.Markdown("# Classical Settings")
            with gr.Accordion(open=False):
                classic_show_original = gr.Checkbox(value=True,label='Show Original Image as Features')
        with gr.Column():
            gr.Markdown("# Predictions")
            verbose = gr.Markdown()
            image_preview = gr.Plot(value=None,label="Input Image")
            cnn_features = gr.Gallery(label='CNN Extracted Features',columns=1,object_fit='fill',interactive=False)
            classical_features = gr.Gallery(label='Classical Extracted Features',columns=1,object_fit='fill',interactive=False)
            cnn_prediction=gr.Textbox(interactive=False,value='No Predictions',label='CNN Predictions')
            classical_prediction=gr.Textbox(interactive=False,value='No Predictions',label='Classical Model Predictions')
            prediction_btn = gr.Button('Predict',variant='primary')
        
        prediction_btn.click(
            fn=predict_image,
            inputs=[image_upload,classic_show_original,cnn_max_channel_visual,cnn_couple_channel],
            outputs=[verbose,image_preview,cnn_prediction,classical_prediction,cnn_features,classical_features]
        )
        

if __name__ == "__main__":
    demo.launch(share=False)