# import libraries
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from skimage.segmentation import mark_boundaries
import os
import random

# import torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision import transforms as T
from torchinfo import summary

# import lime modules
from lime import lime_image

# Setting seed
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Setting the page title, icon, layout, and initial sidebar state.
st.set_page_config(
    page_title='XAI',
    page_icon='📷',
    layout='wide',
    initial_sidebar_state='expanded',
)

@st.cache
def load_model(weights_path: str = 'assets/weights/best_weights.pth'):
    """
    It loads a pretrained ResNet18 model, replaces the last layer with a new layer that has 2 outputs,
    and loads the weights from the file specified by the weights_path parameter
    
    :param weights_path: The path to the weights file, defaults to assets/weights/best_weights.pth
    :type weights_path: str (optional)
    :return: A model with the weights loaded from the path.
    """
    model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model_ft.fc.in_features
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft.load_state_dict(torch.load(weights_path))
    return model_ft

@st.cache
def get_image(path):
    """
    It opens the image file, converts it to RGB, and returns the image
    
    :param path: The path to the image you want to classify
    :return: The image is being converted to RGB format.
    """
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

@st.cache
def get_pil_transform(): 
    """
    > It takes an image, resizes it to 256x256, and then crops it to 224x224
    :return: A function that takes in an PIL image and returns a transformed image.
    """
    return T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224)
    ])
    
@st.cache
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
    
@st.cache
def batch_predict(images):
    """
    > It takes a list of images, preprocesses them, and then runs them through the model to get a
    prediction
    
    :param images: a list of images
    :param model: the model that we want to use to make predictions
    :return: The probabilities of the image being a dog or a cat.
    """
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

@st.cache
def run_explanation(img, explainer=lime_image.LimeImageExplainer()):
    """
    `run_explanation` takes an image, and returns a `LimeImageExplanation` object, which contains the
    explanations for the image.
    
    :param img: the image to be explained
    :param explainer: the explainer object. We'll use the default LIMEImageExplainer
    """
    return explainer.explain_instance(
        np.array(pill_transf(img)),
        batch_predict, # inference function
        top_labels = 2,
        random_seed = 42,
        batch_size = 64,
        num_features = 100,
        num_samples = 750 # number of images that will be sent to classification function
    )

if __name__ == '__main__':
    classes = ["NORMAL", "PNEUMONIA"]
    pill_transf = get_pil_transform()
    preprocess_transform = get_preprocess_transform()
    
    st.title("Model Agnostic w/ LIME")
    
    st.info("LIME (Local Interpretable Model-Agnostic Explanations) is an algorithm that can explain individual predictions of any black-box classifier or regressor, by approximateing it locally with an interpretable method.")
    
    # Creating a sidebar.
    with st.sidebar:
        st.header('Parameters Settings')
        selected_class = st.selectbox("Select Class", ['PNEUMONIA', 'NORMAL'])
        max_features = st.slider("Max Features", 1, 10, 5)
    
    # Loading the model and displaying the model summary.
    with st.expander('Model Summary'):
        try:
            model = load_model()
            model_summary = summary(model, input_size=(1, 3, 224, 224), verbose=0, depth=2)
            st.code(model_summary)
        except Exception as e:
            print(e)
            st.error("Model not loaded!\n{}".format(e))


    image_path = dict(
        NORMAL = './assets/image/NORMAL2-IM-1440-0001.jpeg',
        PNEUMONIA = './assets/image/person1951_bacteria_4882.jpeg'
    )
    
    if selected_class:
        selected_path = image_path[selected_class]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            img = get_image(path=selected_path)
            st.image(img, caption="Ground Truth: {}".format(selected_class), use_column_width=True)
            
            # running explanation
            explanation = run_explanation(img)    
        
        # positive explanations
        with col2:
            pos_temp, pos_mask = explanation.get_image_and_mask(
                explanation.top_labels[0], 
                positive_only=True, 
                negative_only=False, 
                num_features=max_features, 
                hide_rest=False
            )
            pos_img_boundary = mark_boundaries(pos_temp/255.0, pos_mask, color=np.array([0, 255, 0])/255.)
            st.image(pos_img_boundary, caption="Positive Explanation; Predicted: {}".format(classes[explanation.top_labels[0]]))
            
        # negative explanations
        with col3:
            neg_temp, neg_mask = explanation.get_image_and_mask(
                explanation.top_labels[0], 
                positive_only=False, 
                negative_only=True, 
                num_features=max_features, 
                hide_rest=False
            )
            neg_img_boundary = mark_boundaries(neg_temp/255.0, neg_mask, color=np.array([255, 0, 0])/255.)
            st.image(neg_img_boundary, caption="Negative Explanation; Predicted: {}".format(classes[explanation.top_labels[0]]))
      