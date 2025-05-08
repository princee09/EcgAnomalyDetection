import streamlit as st
import torch
import cv2
import numpy as np
import os
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
import timm
import neurokit2 as nk

def generate_gradcam(image_path, model_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    target_layers = [model.patch_embed.proj]
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    input_tensor = preprocess(image_pil).unsqueeze(0).to(device)
    
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_normalized = image_resized / 255.0
    visualization = show_cam_on_image(image_normalized, grayscale_cam, use_rgb=True)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_class = torch.argmax(outputs, dim=1).cpu().numpy()[0]
    
    class_names = ['Myocardial Infarction', 'History of MI', 'Abnormal Heartbeat', 'Normal']
    return visualization, class_names[predicted_class], probabilities

def extract_ecg_signal(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (800, 600))
    _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    signal = [point[0][1] for contour in contours for point in contour]
    signal = np.interp(signal, (min(signal), max(signal)), (-1, 1))
    return signal

def classify_arrhythmia(heart_rate):
    if 65 <= heart_rate <= 85:
        return "Normal Sinus Rhythm (NSR)"
    elif heart_rate < 65:
        if 40 <= heart_rate < 65:
            return "Bradyarrhythmia - Sinus Bradycardia"
        elif heart_rate < 40:
            return "Bradyarrhythmia - Possible Sick Sinus Syndrome / AV Block"
    elif heart_rate > 85:
        if 85 < heart_rate <= 120:
            return "Tachyarrhythmia - Sinus Tachycardia"
        elif 120 < heart_rate <= 150:
            return "Tachyarrhythmia - Atrial Tachycardia / Supraventricular Tachycardia (SVT)"
        elif heart_rate > 150:
            return "Tachyarrhythmia - Ventricular Tachycardia / Fibrillation"
    return "Unknown classification"

def main():
    st.title("ECG Anomalies Detection with Grad-CAM Visualization")
    uploaded_image = st.file_uploader("Upload an ECG Image", type=["jpg", "png", "jpeg"])
    model_path = "ecg-classification-project/data/preprocessed/models/vit_ecg_classifier.pth"
    if uploaded_image:
        image_path = f"temp_{uploaded_image.name}"
        
        with open(image_path, "wb") as f:
            f.write(uploaded_image.read())
        
        st.image(image_path, caption="Uploaded ECG Image", use_container_width=True)
        
        if st.button("Process Image"):
            ecg_signal = extract_ecg_signal(image_path)
            r_peaks = nk.ecg_findpeaks(ecg_signal, sampling_rate=500)
            heart_rate = np.mean(nk.ecg_rate(r_peaks, sampling_rate=500))
            
            heatmap, predicted_class, probabilities = generate_gradcam(image_path, model_path)
            st.image(heatmap, caption="Grad-CAM Heatmap", width=600)
            st.write(f"### Predicted Class: {predicted_class}")
            
            st.write("### Probability Scores")
            for class_name, prob in zip(['Myocardial Infarction', 'History of MI', 'Abnormal Heartbeat', 'Normal'], probabilities):
                st.write(f"{class_name}: {prob:.4f}")
            
            st.write(f"### Estimated Heart Rate: {heart_rate:.2f} bpm")
            st.write(f"### Arrhythmia Classification: {classify_arrhythmia(heart_rate)}")

if __name__ == "__main__":
    main()
