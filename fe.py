import streamlit as st
import torch
from torchvision import transforms, models
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image

class EmotionDetectionCNN(nn.Module):
    def __init__(self):
        super(EmotionDetectionCNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 7)

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model():
    model = EmotionDetectionCNN()
    model.load_state_dict(torch.load("emotion_model.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

st.title("Emotion Detection from Image")

uploaded_file = st.file_uploader("Upload a grayscale image (48x48)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        st.write(f"Predicted Emotion: **{class_names[predicted.item()]}**")
