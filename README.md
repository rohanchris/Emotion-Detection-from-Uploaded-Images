# Emotion-Detection-from-Uploaded-Images

1. Import Necessary Libraries
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
Libraries: The code imports essential libraries for data manipulation, deep learning, image processing, visualization, and creating the web app.

2. Streamlit Setup
st.title("Emotion Detection from Facial Expressions")
st.write("This app detects emotions from facial expressions using a Convolutional Neural Network.")
Streamlit Title: Sets the title and description for the web application.

3. Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Device Selection: Checks if a GPU is available and sets the computation device accordingly, enhancing performance for training and inference.

4. Define File Paths
TRAIN_DIR = 'C:/Users/dell/Face Exp/Images/train'  # Update path accordingly
TEST_DIR = 'C:/Users/dell/Face Exp/Images/test'    # Update path accordingly
File Paths: Specifies the directories for training and testing datasets. Make sure to update these paths according to your local setup.

5. Data Transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])
Transformations: Prepares images for the CNN by converting them to grayscale, resizing them to 48x48 pixels, and converting them into PyTorch tensors.

6. Load Datasets
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
Loading Data: Uses ImageFolder to load images and labels from the specified directories and creates data loaders for batching and shuffling during training.

7. Define the CNN Model
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, len(train_dataset.classes))

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
CNN Architecture: Defines the CNN architecture with two convolutional layers followed by fully connected layers. It uses ReLU activation and max pooling.

8. Load Model for Inference
@st.cache_resource
def load_model():
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load('emotion_cnn_model.pth', weights_only=True))
    model.eval()
    return model
Model Loading: A function to load the trained CNN model. The model is cached for efficiency during re-runs.

9. Define a Function for Making Predictions
def predict_emotion(image):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
        return train_dataset.classes[predicted.item()]
    except Exception as e:
        st.error(f"Error in predicting emotion: {e}")
        return None
Emotion Prediction: Converts the uploaded image to the appropriate format, processes it, and returns the predicted emotion label.

10. Image Upload Section in Streamlit
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    predicted_emotion = predict_emotion(image)
    st.write(f"Predicted Emotion: {predicted_emotion}")
File Uploader: Allows users to upload an image file for emotion detection. The predicted emotion is displayed after classification.

11. Visualize Some Test Results
def visualize_predictions(test_loader, num_images=5):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(15, 6))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i].cpu().numpy().transpose(1, 2, 0), cmap='gray')
        plt.title(f'True: {train_dataset.classes[labels[i]]}\nPred: {train_dataset.classes[preds[i]]}')
        plt.axis('off')

    st.pyplot(plt)
Visualization Function: Displays images from the test set alongside their true and predicted labels for a visual validation of the model’s performance.

12. Generate Classification Report and Confusion Matrix
def generate_report_and_confusion_matrix():
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=train_dataset.classes, zero_division=0)
    st.text("Classification Report:\n" + report)

    conf_matrix = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    st.pyplot(plt)
Report Generation: Computes and displays the classification report and confusion matrix based on the model’s predictions on the test set.

