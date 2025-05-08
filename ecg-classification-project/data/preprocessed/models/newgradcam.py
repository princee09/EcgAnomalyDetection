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
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# Define explicit file paths
file_path_dat = "C:\\Users\\pbimu\\Downloads\\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\\21000_hr\\21009_hr.dat"

# Extract record name (without extension)
record_name = file_path_dat[:-4]

# Read the ECG signal
record = wfdb.rdrecord(record_name)
ecg_signal = record.p_signal  # ECG signals (shape: [5000, 12])
fs = record.fs  # Sampling frequency (500 Hz)
lead_names = record.sig_name  # Lead names

# Select the best lead based on the strongest R-peaks
best_lead_index = 1  # Default to Lead II
best_peak_count = 0

for i in range(12):
    signal = ecg_signal[:, i]
    peaks, _ = find_peaks(signal, height=np.mean(signal) + np.std(signal), distance=fs*0.6)
    
    if len(peaks) > best_peak_count:
        best_peak_count = len(peaks)
        best_lead_index = i

best_lead_signal = ecg_signal[:, best_lead_index]
best_lead_name = lead_names[best_lead_index]

# Detect R-peaks on the best lead
peaks, _ = find_peaks(best_lead_signal, height=np.mean(best_lead_signal) + np.std(best_lead_signal), distance=fs*0.6)

# Estimate Heart Rate
rr_intervals = np.diff(peaks) / fs
heart_rate = 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0

print(f"✅ Selected Lead: {best_lead_name}")

# Plot all 12 leads
plt.figure(figsize=(12, 8))
for i in range(12):
    plt.subplot(6, 2, i + 1)
    plt.plot(ecg_signal[:, i], label=f'Lead {lead_names[i]}')
    plt.legend()
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude (mV)")

plt.tight_layout()
plt.show()

# Plot R-peak detection on the selected lead
plt.figure(figsize=(10, 4))
plt.plot(best_lead_signal, label=f"Best Lead: {best_lead_name}", color='blue')
plt.scatter(peaks, best_lead_signal[peaks], color='red', label="R-peaks", marker='o')
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude (mV)")
plt.title(f"R-peak Detection - {best_lead_name} - Estimated HR: {heart_rate:.2f} BPM")
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
for i in range(12):
    plt.subplot(6, 2, i + 1)
    plt.plot(ecg_signal[:, i], label=f'Lead {lead_names[i]}', color='blue')
    if i == best_lead_index:  
        plt.scatter(peaks, ecg_signal[peaks, i], color='red', marker='o', label="R-peaks")
    plt.legend()

plt.tight_layout()
plt.show()


def plot_ecg(dat_file=file_path_dat):
    output_image = os.path.splitext(os.path.basename(dat_file))[0] + ".png"
    
    # Read the header file to get metadata
    record = wfdb.rdrecord(dat_file.replace(".dat", ""))
    signals = record.p_signal  # ECG signal data
    sampling_rate = record.fs  # Sampling frequency
    lead_names = record.sig_name  # ECG lead names
    
    num_leads = len(lead_names)
    duration = signals.shape[0] / sampling_rate  # Calculate duration in seconds
    time_axis = np.linspace(0, duration, signals.shape[0])  # Time axis
    
    # Set up ECG grid style
    plt.figure(figsize=(12, 8))
    plt.suptitle("ECG REPORT", fontsize=14, fontweight='bold')
    
    for i in range(num_leads):
        plt.subplot(num_leads, 1, i + 1)
        plt.plot(time_axis, signals[:, i], color='black', linewidth=1)
        plt.ylabel(lead_names[i], fontsize=10, fontweight='bold', rotation=0, labelpad=20)
        plt.xticks([])  # Hide x-axis ticks except last subplot
        plt.yticks([])
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.xlabel("Time (s)")
    plt.savefig(output_image, dpi=300)
    plt.close()  # Close the plot to free memory
    
    return output_image


def generate_gradcam(image_path, model_path, output_dir=None):
    # Check if files exist
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    try:
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=4)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # Define Grad-CAM++ target layers - using patch embedding projection layer
    target_layers = [model.patch_embed.proj]  # Suitable for Grad-CAM

    # Preprocess function for inference
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    try:
        # Load and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image_pil = Image.fromarray(image_rgb)  # Convert to PIL Image
        input_tensor = preprocess(image_pil).unsqueeze(0).to(device)  # Apply transformations

        print(f"Input Tensor Shape: {input_tensor.shape}")  # Debugging output

        # Generate Grad-CAM++ heatmap
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor)[0]  # Ensure it's 2D

        # Overlay heatmap on the original image
        image_resized = cv2.resize(image_rgb, (224, 224))
        image_normalized = image_resized / 255.0
        visualization = show_cam_on_image(image_normalized, grayscale_cam, use_rgb=True)

        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class = torch.argmax(outputs, dim=1).cpu().numpy()[0]

        class_names = ['Myocardial Infarction', 'History of MI', 'Abnormal Heartbeat', 'Normal']

        # Save the heatmap if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_heatmap.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            print(f"Heatmap saved to: {output_path}")

        # Clean up CUDA memory if needed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return visualization, class_names[predicted_class], probabilities

    except Exception as e:
        print(f"Error processing image: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, None, None
def extract_ecg_signal(image_path):
    """Extracts the ECG waveform from an image using OpenCV preprocessing."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (800, 600))  # Resize for consistency
    
    # Apply thresholding to extract ECG waveform
    _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours of the ECG waveform
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    signal = []
    for contour in contours:
        for point in contour:
            signal.append(point[0][1])  # Extract y-coordinates
    
    signal = np.array(signal)
    signal = np.interp(signal, (signal.min(), signal.max()), (-1, 1))  # Normalize
    return signal

def classify_arrhythmia(heart_rate):
    """
    Classifies arrhythmia based on heart rate (HR) values.
    :param heart_rate: (float) The estimated heart rate in beats per minute (bpm).
    :return: (str) Arrhythmia classification.
    """
    if 60 <= heart_rate <= 90:
        arrhythmia_type = "Normal Sinus Rhythm (NSR)"
    elif heart_rate < 60:
        arrhythmia_type = "Bradyarrhythmia (Slow Heart Rate)"
        if 40 <= heart_rate < 60:
            arrhythmia_type += " - Sinus Bradycardia"
        elif heart_rate < 40:
            arrhythmia_type += " - Possible Sick Sinus Syndrome / AV Block"
    elif heart_rate > 90:
        arrhythmia_type = "Tachyarrhythmia (Fast Heart Rate)"
        if 90 < heart_rate <= 130:
            arrhythmia_type += " - Sinus Tachycardia"
        elif 130 < heart_rate <= 160:
            arrhythmia_type += " - Atrial Tachycardia / Supraventricular Tachycardia (SVT)"
        elif heart_rate > 160:
            arrhythmia_type += " - Ventricular Tachycardia / Fibrillation"
    else:
        arrhythmia_type = "Unknown classification"
    
    return arrhythmia_type

if __name__ == "__main__":
    model_path = 'ecg-classification-project/data/preprocessed/models/vit_ecg_classifier.pth'
    image_path = plot_ecg()
    print(f"ECG report saved as: {image_path}")
    output_dir = 'ecg-classification-project/data/preprocessed/visualizations'
    ecg_signal = extract_ecg_signal(image_path)
    heatmap, predicted_class, probabilities = generate_gradcam(image_path, model_path, output_dir)
    print(f"Estimated Heart Rate: {heart_rate:.2f} bpm")
    arrhythmia_result = classify_arrhythmia(heart_rate)
    print(f"Arrhythmia Classification: {arrhythmia_result}")

    if predicted_class:
        print(f"Predicted Class: {predicted_class}")
        for i, (class_name, prob) in enumerate(zip(['Myocardial Infarction', 'History of MI', 'Abnormal Heartbeat', 'Normal'], probabilities)):
            print(f"{class_name}: {prob:.4f}")