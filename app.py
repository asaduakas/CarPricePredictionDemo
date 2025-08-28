import streamlit as st
import torch
import numpy as np
from PIL import Image
import joblib
from torchvision import transforms
from fastai.vision.all import load_learner

cnn = load_learner('vision_model.pkl')
forest = joblib.load('forest_model.pkl')
pca = joblib.load('pca.pkl')

st.set_page_config(page_title="Car Price Estimator", page_icon=":car:")
st.title("Car Price Estimator (demo)")
st.write("Upload a car photo to get a predicted price range! ")
file = st.file_uploader("Upload a car image", type=["jpeg", "jpg", "png"])

st.sidebar.header("Car features")
car_age = st.sidebar.number_input("Car Age", min_value=0, max_value=30, value=5)
runned_miles = st.sidebar.number_input("Runned Kilometers", min_value=0, max_value=1_000_000, value=50_000)
seat_num = st.sidebar.number_input("Seats", min_value=1, max_value=10, value=5)

if file:
    img = Image.open(file)

    device = "cpu"
    cnn.model[0] = cnn.model[0].to(device)
    cnn.model.eval()
    
    tfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    x = tfms(img).unsqueeze(0).to(device)
   
    print("About to run torch.no_grad")
    with torch.no_grad():
       feat = cnn.model[0](x)  
       emb = torch.nn.functional.adaptive_avg_pool2d(feat, 1).squeeze().cpu().numpy()
    
    print("Finished running torch.no_grad!")
    emb_pca = pca.transform(emb.reshape(1, -1))
    
    X_input = np.hstack([[car_age, runned_miles, seat_num], emb_pca.flatten()])
    
    log_price = forest.predict(X_input.reshape(1,-1))[0]
    price = np.exp(log_price)
    
    st.success(f"Predicted price: ${price:,.2f}")