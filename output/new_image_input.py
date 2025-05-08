import streamlit as st
import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
import timm
import neurokit2 as nk

# --- Preprocessing Helpers ---
def resize_with_aspect_ratio(image, target_width, target_height):
    h, w = image.shape[:2]
    aspect_ratio = w / h
    if w > h:
        new_w = target_width
        new_h = int(target_width / aspect_ratio)
    else:
        new_h = target_height
        new_w = int(target_height * aspect_ratio)
    resized_image = cv2.resize(image, (new_w, new_h))

    top = (target_height - new_h) // 2
    bottom = target_height - new_h - top
    left = (target_width - new_w) // 2
    right = target_width - new_w - left

    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def preprocess_image(image_path, target_width=224, target_height=224):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Highlight dark ECG lines
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 3)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"No contours found in image: {image_path}")
        return None

    # Combine bounding boxes of non-text ECG contours
    ecg_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        # Skip very wide or very flat contours (likely text or borders)
        if h < 10 or aspect_ratio > 10:
            continue
        ecg_contours.append((x, y, w, h))

    if not ecg_contours:
        print("No suitable ECG contours found.")
        return None

    # Get the union of all ECG contour boxes
    x_min = min([x for x, y, w, h in ecg_contours])
    y_min = min([y for x, y, w, h in ecg_contours])
    x_max = max([x + w for x, y, w, h in ecg_contours])
    y_max = max([y + h for x, y, w, h in ecg_contours])

    # Optional padding
    padding = 10
    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_max = min(x_max + padding, image.shape[1])
    y_max = min(y_max + padding, image.shape[0])

    # Crop and resize
    cropped = image[y_min:y_max, x_min:x_max]
    resized = cv2.resize(cropped, (target_width, target_height))

    return resized

def preview_image(image_path, new_width=224, new_height=224):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    resized_image = preprocess_image(image_path, new_width, new_height)
    if resized_image is None:
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image_rgb)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(resized_image_rgb)
    ax[1].set_title('Preprocessed Image')
    ax[1].axis('off')

    st.pyplot(fig)

# --- GradCAM + Classification ---
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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image_rgb = preprocess_image(image_path)
    if image_rgb is None:
        raise ValueError(f"Preprocessing failed for image: {image_path}")

    image_pil = Image.fromarray(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(image_pil).unsqueeze(0).to(device)
    
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    
    image_normalized = image_rgb / 255.0
    visualization = show_cam_on_image(image_normalized, grayscale_cam, use_rgb=True)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_class = torch.argmax(outputs, dim=1).cpu().numpy()[0]
    
    class_names = ['Myocardial Infarction', 'History of MI', 'Abnormal Heartbeat', 'Normal']
    return visualization, class_names[predicted_class], probabilities

# --- ECG Signal Extraction ---
def extract_ecg_signal(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (800, 600))
    _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    signal = [point[0][1] for contour in contours for point in contour]
    signal = np.interp(signal, (min(signal), max(signal)), (-1, 1))
    return signal

# --- Arrhythmia Classifier ---
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

# --- Streamlit App ---
def main():
    # Add custom styling
    st.markdown("""
        <style>
            .main-header {
                color: white;
                font-size: 2.5em;
                font-weight: 600;
                margin-bottom: 1em;
                text-align: center;
                text-shadow: 0 0 10px rgba(255,255,255,0.3);
            }
            
            .sub-header {
                color: white;
                font-size: 1.8em;
                font-weight: 500;
                margin: 1.5em 0 1em 0;
                text-shadow: 0 0 5px rgba(255,255,255,0.2);
            }
            
            .upload-section {
                background: rgba(255, 255, 255, 0.9);
                border-radius: 15px;
                padding: 2rem;
                margin: 2rem 0;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            
            .results-section {
                background: rgba(255, 255, 255, 0.9);
                border-radius: 15px;
                padding: 2rem;
                margin: 2rem 0;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            
            .stButton>button {
                background: linear-gradient(45deg, #1a8754, #20c997);
                color: white;
                border: none;
                border-radius: 25px;
                padding: 0.8em 2em;
                font-weight: 500;
                transition: all 0.3s ease;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-top: 1.5em;
            }
            
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(32, 201, 151, 0.3);
                background: linear-gradient(45deg, #20c997, #1a8754);
            }
            
            .metric-card {
                background: rgba(240, 248, 255, 0.8);
                border-radius: 10px;
                padding: 1rem;
                margin: 0.5rem 0;
                border-left: 4px solid #1e88e5;
            }
            
            .file-uploader {
                margin: 2em 0;
            }
            
            .stImage {
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            
            .hospital-blue-text {
                color: #1e88e5;
                font-weight: 500;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">ECG Anomalies Detection with Grad-CAM Visualization</h1>', unsafe_allow_html=True)
    
    # Create a container for the upload section
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #1e88e5; text-align: center; margin-bottom: 1.5em;">Upload Your ECG Image</h2>', unsafe_allow_html=True)
        
        # Center the file uploader
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            uploaded_image = st.file_uploader("", type=["jpg", "png", "jpeg"], key="file_uploader", label_visibility="collapsed")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    model_path = "ecg-classification-project/data/preprocessed/models/vit_ecg_classifier.pth"
    
    if uploaded_image:
        image_path = f"temp_{uploaded_image.name}"
        
        with open(image_path, "wb") as f:
            f.write(uploaded_image.read())

        # Preview section
        st.markdown('<h2 class="sub-header">📷 Image Preview</h2>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="results-section">', unsafe_allow_html=True)
            preview_image(image_path)
            st.markdown('</div>', unsafe_allow_html=True)

        # Process button - centered
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            process_button = st.button("Process Image", use_container_width=True)

        if process_button:
            with st.spinner("Analyzing ECG image..."):
                # Results section
                st.markdown('<h2 class="sub-header">🔍 Analysis Results</h2>', unsafe_allow_html=True)
                
                with st.container():
                    st.markdown('<div class="results-section">', unsafe_allow_html=True)
                    
                    # Extract ECG signal and calculate heart rate
                    ecg_signal = extract_ecg_signal(image_path)
                    r_peaks = nk.ecg_findpeaks(ecg_signal, sampling_rate=500)
                    heart_rate = np.mean(nk.ecg_rate(r_peaks, sampling_rate=500))
                    
                    # Generate GradCAM visualization
                    heatmap, predicted_class, probabilities = generate_gradcam(image_path, model_path)
                    
                    # Display results in columns
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.image(heatmap, caption="Grad-CAM Heatmap", use_container_width=True)
                    
                    with col2:
                        st.markdown(f"<h3 style='color: #1e88e5;'>Predicted Class</h3>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h2 style='color: #1a8754; text-align: center;'>{predicted_class}</h2></div>", unsafe_allow_html=True)
                        
                        st.markdown(f"<h3 style='color: #1e88e5; margin-top: 1.5em;'>Heart Rate</h3>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h2 style='color: #1a8754; text-align: center;'>{heart_rate:.1f} bpm</h2></div>", unsafe_allow_html=True)
                        
                        st.markdown(f"<h3 style='color: #1e88e5; margin-top: 1.5em;'>Arrhythmia Classification</h3>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h3 style='color: #1a8754; text-align: center;'>{classify_arrhythmia(heart_rate)}</h3></div>", unsafe_allow_html=True)
                    
                    # Probability scores
                    st.markdown("<h3 style='color: #1e88e5; margin-top: 2em;'>Probability Scores</h3>", unsafe_allow_html=True)
                    
                    # Create a horizontal bar chart for probabilities
                    class_names = ['Myocardial Infarction', 'History of MI', 'Abnormal Heartbeat', 'Normal']
                    fig, ax = plt.subplots(figsize=(10, 4))
                    bars = ax.barh(class_names, probabilities, color=['#20c997', '#3498db', '#f39c12', '#2c3e50'])
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Probability')
                    ax.grid(axis='x', linestyle='--', alpha=0.6)
                    
                    # Add probability values on bars
                    for bar, prob in zip(bars, probabilities):
                        ax.text(max(0.05, min(prob - 0.1, 0.9)), bar.get_y() + bar.get_height()/2, 
                                f'{prob:.2f}', va='center', color='white', fontweight='bold')
                    
                    st.pyplot(fig)
                    
                    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
