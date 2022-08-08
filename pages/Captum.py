# Importing the necessary libraries.
import numpy as np
import streamlit as st
from PIL import Image
from skimage.segmentation import mark_boundaries
import os
import random
import copy
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# import torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms as T
from torchinfo import summary

# captum modules
import captum
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

@st.cache
def seed_everything (seed=42):
    """
    It sets the seed for the random number generator in Python, NumPy, and PyTorch
    
    :param seed: The seed value to use for the random number generator, defaults to 42 (optional)
    :return: The seed value.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    return seed

@st.cache
def load_model(model='mobilenetv3', weights_path: str = 'assets/weights/mobilenetv3_ver2_best_weights.pth'):  
    """
    It loads a pre-trained model and returns it
    
    :param model: The model to be used, defaults to mobilenetv3 (optional)
    :param weights_path: str = 'assets/weights/mobilenetv3_ver2_best_weights.pth', defaults to
    assets/weights/mobilenetv3_ver2_best_weights.pth
    :type weights_path: str (optional)
    :return: The model is being returned.
    """
    if model == 'resnet18':
        model_ft = models.resnet18()
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        model_ft.load_state_dict(torch.load(weights_path))
        return model_ft
    
    elif model == 'efficientnetb1':
        model_ft = models.efficientnet_b1()
        num_ftrs = model_ft.classifier[-1].in_features
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_ft.classifier[-1] = nn.Linear(num_ftrs, 2)
        model_ft.load_state_dict(torch.load(weights_path))
        return model_ft
    
    elif model == 'mobilenetv3':
        model_ft = models.mobilenet_v3_large()
        num_ftrs = model_ft.classifier[-1].in_features
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_ft.classifier[-1] = nn.Linear(num_ftrs, 2)
        model_ft.load_state_dict(torch.load(weights_path))
        return model_ft
    
@st.cache
def get_image(path):
    """
    It takes a path to an image, opens it, converts it to RGB, and resizes it to 256x256
    
    :param path: The path to the image you want to classify
    :return: The image is being returned.
    """
    return Image.open(path).convert('RGB').resize((256, 256))

def get_pil_transform(): 
    """
    > It takes an image, resizes it to 256x256, and returns the resized image
    :return: A function that takes in an image and returns a transformed image.
    """
    return T.Compose([
        T.Resize((256, 256)),
    ])

def get_preprocess_transform():
    """
    It takes an image as input, converts it to a tensor, and normalizes it
    :return: A function that takes in an image and returns a tensor.
    """
    norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return T.Compose([
        T.ToTensor(),
        T.Normalize(**norm_cfg)
    ])
    
if __name__ == '__main__':
    # Setting the page title, icon, layout, and initial sidebar state.
    st.set_page_config(
        page_title='XAI',
        page_icon='📷',
        layout='wide',
        initial_sidebar_state='expanded',
        menu_items={
            "About": "# This is a project to demonstrate the use of XAI",
            "Get Help": "https://github.com/TYH71/Pneumonia-Classification-w-XAI"
        }
    )
    
    # pre-set variable
    seed = seed_everything(seed=0)
    classes = ["NORMAL", "PNEUMONIA"]
    pill_transf = get_pil_transform()
    preprocess_transform = get_preprocess_transform()
    
    image_path = {os.path.basename(fp):fp for fp in glob.glob("./assets/image/*/*.jpeg")}
    
    st.title("Gradient Based Attribution")
    st.info("<hello world>")
    
    # Creating a sidebar.
    with st.sidebar:
        st.header('Parameters Settings')
        selected_case = st.selectbox("Select Class", image_path.keys())
    
    # Loading the model and displaying the model summary.
    with st.expander('Model Summary'):
        try:
            model = load_model()
            model_summary = summary(model, input_size=(1, 3, 224, 224), verbose=0, depth=2)
            st.code(model_summary)
        except Exception as e:
            print(e)
            st.error("Model not loaded!\n{}".format(e))
    
    if selected_case:
        selected_path = image_path[selected_case]
        assert os.path.exists(selected_path), "The selected image path does not exist!"
        
        col1, col2 = st.columns(2)
        col4, col5, col6 = st.columns(3)
        
        # original image and ground truth
        with col1:
            img = get_image(path=selected_path)
            st.subheader("Original Image")
            st.image(img, caption="Ground Truth: {}".format(selected_case), use_column_width=True)

        # run inference on image
        img_T = preprocess_transform(pill_transf(img))
        preds = model(img_T.reshape(1, 3, 256, 256)).detach()
        logits = F.softmax(preds)
        pred_idx = torch.argmax(logits)
        conf_pneumonia, conf_normal = logits[0]
        pred_class = classes[pred_idx]
        
        col4.metric(label='Predicted', value=pred_class)
        col5.metric(label='Probability of Pneumonia', value="{:.2%}".format(conf_normal))
        col6.metric(label='Probability of Normal', value="{:.2%}".format(conf_pneumonia))

        # feature attribution
        with col2:
            st.subheader('Attribution Map')        
        
