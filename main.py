# Importing the necessary libraries.
import numpy as np
import streamlit as st
from PIL import Image
from skimage.segmentation import mark_boundaries
import os
import random
import copy

# import torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms as T
from torchinfo import summary

# import lime modules
from lime import lime_image

@st.cache
def seed_everything(seed=0):
    """
    > It sets the seed for the random number generator in Python, NumPy, and PyTorch
    
    :param seed: the random seed to use for the random number generator, defaults to 42 (optional)
    :return: The seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed) # for pytorch
    torch.backends.cudnn.deterministic = True # for pytorch
    return seed

@st.cache
def load_model(model='mobilenetv3', weights_path: str = 'assets/weights/mobilenetv3_ver2_best_weights.pth'):  
    """
    It loads a pre-trained model and returns it
    
    :param model: The model to be used, defaults to mobilenetv3 (optional)
    :param weights_path: str = 'assets/weights/mobilenetv3_best_weights.pth', defaults to
    assets/weights/mobilenetv3_best_weights.pth
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
    It opens the image file, converts it to RGB, and returns the image
    
    :param path: The path to the image you want to classify
    :return: The image is being converted to RGB format.
    """
    return Image.open(path).convert('RGB').resize((256, 256))
    
    # bug: InMemoryFileManager: Missing file <cache>.jpeg 
    # with open(os.path.abspath(path), 'rb') as f:
    #     with Image.open(f) as img:
    #         return img.convert('RGB')

def get_pil_transform(): 
    """
    > It takes an image, resizes it to 256x256, and then crops it to 224x224
    :return: A function that takes in an PIL image and returns a transformed image.
    """
    return T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224)
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

# bug: removed cache because it was causing memory to overload
# run_explanation() runs batch_predict() multiple times, which causes memory to overload
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
    print("using device:", device)
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

# objects in cache are removed after 1 hour
# explanations are removed after each session, so cache doesn't require to persist that long
@st.cache(ttl=1*3600)
def run_explanation(img, explainer=lime_image.LimeImageExplainer(feature_selection='auto', random_state=seed)):
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
        random_seed = seed,
        batch_size = 16,
        distance_metric='l2',
        num_samples = 100 # number of images that will be sent to classification function
    )

# @st.cache(ttl=12*3600) # objects in cache are removed after 12 hours
def generate_img_boundary(explanation, positive, max_features, hide_rest):
    """
    It takes an explanation object, a boolean indicating whether we want to see positive or negative
    features, the maximum number of features to show, and a boolean indicating whether we want to hide
    the rest of the image. It then returns an image with the boundaries of the features highlighted
    
    :param explanation: the explanation object returned by the explainer
    :param positive: True if we want to see the positive features, False if we want to see the negative
    features
    :param max_features: The maximum number of features to show
    :param hide_rest: If True, the rest of the image that doesn't pertain to the explanation will be
    hidden
    """
    color = np.array([0, 255, 0] if positive else [255, 0, 0]) / 255.
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=positive, 
        negative_only=not positive, 
        num_features=max_features, 
        hide_rest=hide_rest
    )
    img_boundary = mark_boundaries(temp/255.0, mask, color=color)
    return img_boundary
    
# A common idiom in Python to use `if __name__ == '__main__':` to guard the code that parses command
# line arguments and invokes the main function.
if __name__ == '__main__':
    # Setting the page title, icon, layout, and initial sidebar state.
    st.set_page_config(
        page_title='XAI',
        page_icon='📷',
        layout='wide',
        initial_sidebar_state='expanded',
    )
    
    # pre-set variable
    seed = seed_everything(seed=0)
    classes = ["NORMAL", "PNEUMONIA"]
    pill_transf = get_pil_transform()
    preprocess_transform = get_preprocess_transform()
    image_path = {
        "PNEUMONIA #1": './assets/image/PNEUMONIA/person1946_bacteria_4874.jpeg',
        "PNEUMONIA #2": "./assets/image/PNEUMONIA/person1946_bacteria_4875.jpeg",
        "NORMAL #1": './assets/image/NORMAL/NORMAL2-IM-1440-0001.jpeg',
        "NORMAL #2": "./assets/image/PNEUMONIA/person1946_bacteria_4875.jpeg"
        # "NORMAL (False Positive)": "assets\\image\\NORMAL\\NORMAL2-IM-1436-0001.jpeg"
    }
    
    # Title
    st.title("Model Agnostic w/ LIME")
    st.info("LIME (Local Interpretable Model-Agnostic Explanations) is an algorithm that can explain individual predictions of any black-box classifier or regressor, by approximateing it locally with an interpretable method.")
    
    # Creating a sidebar.
    with st.sidebar:
        st.header('Parameters Settings')
        selected_case = st.selectbox("Select Class", image_path.keys())
        max_features = st.slider("Max Features", 1, 10, 5)
        hide_rest = st.select_slider("Hide Rest", [False, True])
    
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
        
        col1, col2, col3 = st.columns(3)
        # original image and ground truth
        with col1:
            img = get_image(path=selected_path)
            st.subheader("Original Image")
            st.image(img, caption="Ground Truth: {}".format(selected_case), use_column_width=True)

            # Checking if the selected case is in the session state. 
            # If it is not, it will run the explanation and
            # store it in the session state.
            if selected_case not in st.session_state:
                st.session_state[selected_case] = copy.deepcopy(run_explanation(img))
            assert st.session_state[selected_case] is not None, "Explanation not found!"
            explanation = st.session_state[selected_case]
            pred_class = classes[explanation.top_labels[0]]

        # positive explanations
        with col2:
            pos_img_boundary = generate_img_boundary(explanation, positive=True, max_features=max_features, hide_rest=hide_rest)
            st.subheader("Positive Explanations")
            st.image(
                pos_img_boundary, 
                caption="Predicted: {}".format(pred_class),
                use_column_width=True
            )
            
        # negative explanations
        with col3:
            neg_img_boundary = generate_img_boundary(explanation, positive=False, max_features=max_features, hide_rest=hide_rest)
            st.subheader("Negative Explanation")
            st.image(
                neg_img_boundary, 
                caption="Predicted: {}".format(pred_class), 
                use_column_width=True
            )