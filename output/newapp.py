import streamlit as st
import os
import wfdb
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import torch
import timm
import cv2
import matplotlib.gridspec as gridspec
import io
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.animation as animation
from fpdf import FPDF
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import scipy.io.wavfile as wav
import subprocess
# Initialize session state

ECG_DATA_DIR = "D:/college_projects/ECG_image_project/temp"

if "page" not in st.session_state:
    st.session_state.page = "upload"

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
        
        .card-container {
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
        
        .success-message {
            background-color: rgba(32, 201, 151, 0.2);
            border-left: 4px solid #20c997;
            padding: 1em;
            border-radius: 5px;
            margin: 1em 0;
        }
        
        .error-message {
            background-color: rgba(255, 76, 76, 0.2);
            border-left: 4px solid #ff4c4c;
            padding: 1em;
            border-radius: 5px;
            margin: 1em 0;
        }
        
        .plot-container {
            background: white;
            border-radius: 10px;
            padding: 1em;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin: 1.5em 0;
        }
    </style>
""", unsafe_allow_html=True)

def go_to_results():
    st.session_state.page = "results"

def upload_page():
    st.markdown('<h1 class="main-header">ECG Signal Analysis</h1>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #1e88e5; text-align: center; margin-bottom: 1.5em;">Upload ECG Signal File</h2>', unsafe_allow_html=True)
        
        # Ensure ECG data directory exists
        os.makedirs(ECG_DATA_DIR, exist_ok=True)
        
        # Center the file uploader
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            uploaded_file = st.file_uploader("", type=["dat"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            file_path = os.path.join(ECG_DATA_DIR, uploaded_file.name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Store file details in session state
            st.session_state.file_name = uploaded_file.name
            st.session_state.file_path = file_path
            
            st.markdown(f"""
                <div class="success-message">
                    <h3 style="color: #1a8754; margin: 0;">File uploaded successfully! 😊</h3>
                    <p>File name: <b>{uploaded_file.name}</b></p>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.button("Analyze ECG Signal ➡️", on_click=go_to_results, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def result_page():
    if "file_path" not in st.session_state:
        st.markdown("""
            <div class="error-message">
                <h3 style="color: #ff4c4c; margin: 0;">❌ No file uploaded!</h3>
                <p>Please go back and upload a file.</p>
            </div>
        """, unsafe_allow_html=True)
        return

    st.markdown('<h1 class="main-header">ECG Analysis Results</h1>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown(f"""
            <h2 style="color: #1e88e5; text-align: center; margin-bottom: 1em;">Analysis for {st.session_state.file_name}</h2>
        """, unsafe_allow_html=True)
        
        file_path_dat = st.session_state.file_path
        record_name = file_path_dat[:-4]

        try:
            record = wfdb.rdrecord(record_name)
        except Exception as e:
            st.markdown(f"""
                <div class="error-message">
                    <h3 style="color: #ff4c4c; margin: 0;">❌ Error reading ECG file</h3>
                    <p>{str(e)}</p>
                </div>
            """, unsafe_allow_html=True)
            return

        ecg_signal = record.p_signal
        fs = record.fs
        lead_names = record.sig_name

        best_lead_index = 1
        best_peak_count = 0

        for i in range(len(lead_names)):
            signal = ecg_signal[:, i]
            peaks, _ = find_peaks(signal, height=np.mean(signal) + np.std(signal), distance=fs * 0.6)
            
            if len(peaks) > best_peak_count:
                best_peak_count = len(peaks)
                best_lead_index = i

        best_lead_signal = ecg_signal[:, best_lead_index]
        best_lead_name = lead_names[best_lead_index]

        peaks, _ = find_peaks(best_lead_signal, height=np.mean(best_lead_signal) + np.std(best_lead_signal), distance=fs * 0.6)

        rr_intervals = np.diff(peaks) / fs
        heart_rate = 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
        st.session_state.heart_rate = heart_rate
        
        # Display metrics in cards
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #1e88e5; margin: 0;">Selected Lead</h3>
                    <h2 style="color: #1a8754; text-align: center; margin: 0.5em 0;">{best_lead_name}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #1e88e5; margin: 0;">Heart Rate</h3>
                    <h2 style="color: #1a8754; text-align: center; margin: 0.5em 0;">{heart_rate:.2f} BPM</h2>
                </div>
            """, unsafe_allow_html=True)
        
        # Arrhythmia classification
        arrhythmia_result = classify_arrhythmia(heart_rate)
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #1e88e5; margin: 0;">Arrhythmia Classification</h3>
                <h2 style="color: #1a8754; text-align: center; margin: 0.5em 0;">{arrhythmia_result}</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Option to upload another file
        if st.button("⬅️ Upload Another File", use_container_width=True):
            st.session_state.page = "upload"
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ECG Signal Plots
    st.markdown('<h2 class="sub-header">ECG Signal Plots</h2>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card-container plot-container">', unsafe_allow_html=True)
        
        fig, axes = plt.subplots(6, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i in range(12):
            axes[i].plot(ecg_signal[:, i], label=f'Lead {lead_names[i]}')
            axes[i].legend()
            axes[i].set_xlabel("Time (samples)")
            axes[i].set_ylabel("Amplitude (mV)")
            axes[i].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # R-peak Detection
    st.markdown('<h2 class="sub-header">R-peak Detection</h2>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card-container plot-container">', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(best_lead_signal, label=f"Best Lead: {best_lead_name}", color='blue')
        ax.scatter(peaks, best_lead_signal[peaks], color='red', label="R-peaks", marker='o')
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Amplitude (mV)")
        ax.set_title(f"R-peak Detection - {best_lead_name} - Estimated HR: {heart_rate:.2f} BPM")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ECG Signals from All Leads
    st.markdown('<h2 class="sub-header">ECG Signals from All Leads</h2>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card-container plot-container">', unsafe_allow_html=True)
        
        fig, axes = plt.subplots(6, 2, figsize=(12, 8), constrained_layout=True)

        for i in range(12):
            ax = axes[i // 2, i % 2]  # Arrange in 6 rows, 2 columns
            ax.plot(ecg_signal[:, i], label=f'Lead {lead_names[i]}', color='blue')

            if i == best_lead_index:  
                ax.scatter(peaks, ecg_signal[peaks, i], color='red', marker='o', label="R-peaks")

            ax.legend()
            ax.set_xlabel("Time (samples)")
            ax.set_ylabel("Amplitude (mV)")
            ax.grid(True, linestyle='--', alpha=0.7)

        st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Generate ECG Report Image
    st.markdown('<h2 class="sub-header">ECG Report</h2>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card-container plot-container">', unsafe_allow_html=True)
        
        # Generate the ECG report image
        ecg_image = plot_ecg(file_path_dat)
        
        if ecg_image:
            # Save the image for GradCAM processing
            temp_path = "temp_ecg_image.png"
            ecg_image.save(temp_path)
            
            # Display the ECG report image
            st.image(ecg_image, caption="ECG Report", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # GradCAM Analysis
    st.markdown('<h2 class="sub-header">GradCAM Analysis</h2>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        
        # Check if the ECG image was generated
        if 'ecg_image' in locals() and ecg_image:
            # Process with GradCAM
            model_path = "ecg-classification-project/data/preprocessed/models/vit_ecg_classifier.pth"
            
            with st.spinner("Preprocessing ECG image..."):
                # Preprocess the image
                preprocessed_path = preprocess_image(temp_path)
                
                if preprocessed_path and os.path.exists(preprocessed_path):
                    # Preview the original and preprocessed images
                    preview_result = preview_image(temp_path)
                    
                    if preview_result:
                        st.markdown("<h3 style='color: #1e88e5;'>Original vs Preprocessed Image</h3>", unsafe_allow_html=True)
                        st.image(preview_result["comparison_path"], caption="Left: Original, Right: Preprocessed", use_container_width=True)
                    
                    # Now generate GradCAM on the preprocessed image
                    with st.spinner("Generating GradCAM visualization..."):
                        # Set device
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                        # Load the trained model
                        try:
                            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=4)
                            model.load_state_dict(torch.load(model_path, map_location=device))
                            model = model.to(device)
                            model.eval()
                        except Exception as e:
                            st.error(f"❌ Failed to load model: {e}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            return

                        # Preprocess function for inference
                        preprocess = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                        ])

                        try:
                            # Load and preprocess the image
                            image = cv2.imread(preprocessed_path)  # Use preprocessed image instead of original
                            if image is None:
                                st.error(f"❌ Failed to read preprocessed image: {preprocessed_path}")
                                st.markdown('</div>', unsafe_allow_html=True)
                                return

                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
                            image_pil = Image.fromarray(image_rgb)  
                            input_tensor = preprocess(image_pil).unsqueeze(0).to(device)  

                            # Generate Grad-CAM heatmap
                            cam = GradCAMPlusPlus(model=model, target_layers=[model.patch_embed.proj])
                            grayscale_cam = cam(input_tensor=input_tensor)[0]  

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
                            predicted_label = class_names[predicted_class]

                            # Save the visualization
                            gradcam_output_path = "gradcam_visualization.png"
                            cv2.imwrite(gradcam_output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

                            # Display Results
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.image(gradcam_output_path, caption="Grad-CAM Heatmap", use_container_width=True)
                            
                            with col2:
                                st.markdown(f"<h3 style='color: #1e88e5;'>Predicted Class</h3>", unsafe_allow_html=True)
                                st.markdown(f"<div class='metric-card'><h2 style='color: #1a8754; text-align: center;'>{predicted_label}</h2></div>", unsafe_allow_html=True)
                                
                                # Probability scores
                                st.markdown("<h3 style='color: #1e88e5; margin-top: 2em;'>Probability Scores</h3>", unsafe_allow_html=True)
                                
                                # Create a horizontal bar chart for probabilities
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

                        except Exception as e:
                            st.error(f"❌ Error during GradCAM generation: {e}")
                else:
                    st.error("❌ Failed to preprocess ECG image.")
        else:
            st.error("❌ ECG report image could not be generated.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Generate ECG Animation and PDF Report
    st.markdown('<h2 class="sub-header">ECG Video & Report</h2>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        
        # Ensure directories exist
        os.makedirs("static/animations", exist_ok=True)
        
        # Generate animation
        with st.spinner("Generating ECG Video..."):
            video_file_path = generate_ecg_video_with_audio(file_path_dat)
        
        if video_file_path and os.path.exists(video_file_path):
            st.markdown('<h3 style="color: #1e88e5;">ECG Video</h3>', unsafe_allow_html=True)
            st.video(video_file_path)
            
            # Generate PDF report
            with st.spinner("Generating PDF report..."):
                # Create PDF
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_page()

                # Title
                pdf.set_font("Arial", "B", 16)
                pdf.cell(200, 10, txt="ECG Analysis Report", ln=True, align="C")

                # Heart rate and classification info
                pdf.ln(10)
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"Estimated Heart Rate: {heart_rate:.2f} BPM", ln=True)
                pdf.cell(200, 10, txt=f"Arrhythmia Classification: {arrhythmia_result}", ln=True)
                pdf.cell(200, 10, txt=f"Predicted Disease: {predicted_label}", ln=True)

                # ECG image
                pdf.ln(10)
                pdf.image(temp_path, x=10, w=180)

                # Save the PDF
                pdf_output_path = "ecg_report.pdf"
                pdf.output(pdf_output_path)
                
                # Add download button for PDF
                with open(pdf_output_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name="ecg_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                
                # Add email button
                if st.button("Send Report via Email", use_container_width=True):
                    with st.spinner("Sending email..."):
                        send_email(pdf_output_path, video_file_path)
        
        st.markdown('</div>', unsafe_allow_html=True)

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
    if len(signal) > 0:
        signal = np.interp(signal, (signal.min(), signal.max()), (-1, 1))  # Normalize
    return signal

def classify_arrhythmia(heart_rate):
    """
    Classifies arrhythmia based on heart rate (HR) values.
    :param heart_rate: (float) The estimated heart rate in beats per minute (bpm).
    :return: (str) Arrhythmia classification.
    """
    if 65 <= heart_rate <= 85:
        arrhythmia_type = "Normal Sinus Rhythm (NSR)"
    elif heart_rate < 65:
        arrhythmia_type = "Bradyarrhythmia (Slow Heart Rate)"
        if 45 <= heart_rate < 65:
            arrhythmia_type += " - Sinus Bradycardia"
        elif heart_rate < 45:
            arrhythmia_type += " - Possible Sick Sinus Syndrome / AV Block"
    elif heart_rate > 85:
        arrhythmia_type = "Tachyarrhythmia (Fast Heart Rate)"
        if 85 < heart_rate <= 130:
            arrhythmia_type += " - Sinus Tachycardia"
        elif 130 < heart_rate <= 160:
            arrhythmia_type += " - Atrial Tachycardia / Supraventricular Tachycardia (SVT)"
        elif heart_rate > 160:
            arrhythmia_type += " - Ventricular Tachycardia / Fibrillation"
    else:
        arrhythmia_type = "Unknown classification"
    
    return arrhythmia_type


def process_ecg_file(file_path):
    # Extract filename without extension
    base_name = os.path.splitext(os.path.basename(file_path))[0]  
    
    # Correct paths
    dat_file_path = os.path.join(ECG_DATA_DIR, base_name + ".dat")  
    hea_file_path = os.path.join(ECG_DATA_DIR, base_name + ".hea")  

    # Ensure the .dat file is in ECG_DATA_DIR
    if not os.path.exists(dat_file_path):
        os.makedirs(ECG_DATA_DIR, exist_ok=True)  
        with open(dat_file_path, "wb") as f:
            f.write(open(file_path, "rb").read())  

    # Check if the corresponding .hea file exists
    if os.path.exists(hea_file_path):
        return wfdb.rdrecord(os.path.splitext(dat_file_path)[0])  # ✅ Removes .dat before reading
    else:
        st.error(f"❌ Corresponding .hea file not found: {hea_file_path}")
        return None

def generate_ecg_video_with_audio(dat_file):
    """Generates a scrolling ECG animation with synchronized audio and saves as MP4."""

    base_name = os.path.splitext(os.path.basename(dat_file))[0]
    output_folder = "static/animations"
    os.makedirs(output_folder, exist_ok=True)

    output_video = os.path.join(output_folder, f"{base_name}_animation.mp4")
    output_wav = os.path.join(output_folder, f"{base_name}.wav")
    final_output = os.path.join(output_folder, f"{base_name}_final.mp4")

    # ------------------- Load .dat and Create ECG Animation -------------------
    data = np.fromfile(dat_file, dtype=np.int16).astype(np.float32)
    if np.isnan(data).any():
        data = np.nan_to_num(data, nan=np.nanmean(data))

    num_leads = 12
    data = data.reshape(-1, num_leads)
    lead_index = 1
    ecg_signal = data[:5000, lead_index]
    sampling_rate = 500  # Hz

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, sampling_rate)
    ax.set_ylim(np.min(ecg_signal), np.max(ecg_signal))
    ax.set_title("Continuous Scrolling ECG Animation")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude (µV)")

    line, = ax.plot([], [], lw=2)

    def update(frame):
        start = frame * 10
        end = start + sampling_rate
        if end > 5000:
            start, end = 4500, 5000
        line.set_data(np.arange(sampling_rate), ecg_signal[start:end])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=500, interval=20, blit=True)
    ani.save(output_video, writer="ffmpeg", fps=40)
    plt.close()

    # ------------------- Generate Audio from .dat using WFDB -------------------
    try:
        record_name = os.path.splitext(dat_file)[0]
        record = wfdb.rdrecord(record_name)
        signals = record.p_signal
        lead_ii = signals[:, 1]
        lead_ii = lead_ii / np.max(np.abs(lead_ii))

        audio_sample_rate = 44100
        audio_signal = np.interp(
            np.linspace(0, len(lead_ii), int(len(lead_ii) * audio_sample_rate / record.fs)),
            np.arange(len(lead_ii)),
            lead_ii
        )
        audio_signal = (audio_signal * 32767).astype(np.int16)
        wav.write(output_wav, audio_sample_rate, audio_signal)
    except Exception as e:
        print(f"Audio generation failed: {e}")
        return output_video  # Return video-only if audio fails

    # ------------------- Combine Video and Audio -------------------
    try:
        command = [
            "ffmpeg",
            "-y",  # Overwrite
            "-i", output_video,
            "-i", output_wav,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            final_output
        ]
        subprocess.run(command, check=True)
    except Exception as e:
        print(f"FFmpeg combination failed: {e}")
        return output_video  # Return video-only if merge fails

    return final_output


def send_email(pdf_path, video_path):
    # Email sending function
    sender_email = "ecgmodel@gmail.com"
    receiver_email = "pbinu1676@gmail.com"
    password = "ciwo xues rlfq tbbs"

    # Create the message container
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "ECG Analysis Report"

    # Attach the PDF
    with open(pdf_path, 'rb') as pdf_file:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(pdf_file.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(pdf_path)}')
        msg.attach(part)

    # Attach the video
    with open(video_path, 'rb') as video_file:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(video_file.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(video_path)}')
        msg.attach(part)

    # Set up the SMTP server
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        st.success("✅ Report sent successfully!")
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")
        

# Add these functions after the generate_gradcam function and before the main function

def preprocess_image(image_path):
    """
    Preprocess the ECG report image before feeding it to the model.
    This includes resizing and subtle enhancement without making it black.
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return None
            
        # Apply subtle Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Apply mild sharpening to enhance ECG lines
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(blurred, -1, kernel)
        
        # Resize to model input size
        processed_img = cv2.resize(sharpened, (224, 224))
        
        # Save the preprocessed image
        preprocessed_path = "preprocessed_" + os.path.basename(image_path)
        cv2.imwrite(preprocessed_path, processed_img)
        
        return preprocessed_path
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        return None

def preview_image(image_path):
    """
    Create a preview of the original and preprocessed images side by side.
    """
    try:
        # Read original image
        original = cv2.imread(image_path)
        if original is None:
            print(f"Failed to read original image: {image_path}")
            return None
            
        # Read preprocessed image
        preprocessed_path = "preprocessed_" + os.path.basename(image_path)
        preprocessed = cv2.imread(preprocessed_path)
        if preprocessed is None:
            print(f"Failed to read preprocessed image: {preprocessed_path}")
            return None
        
        # Resize both to the same height for side-by-side comparison
        height = 400
        original_resized = cv2.resize(original, (int(original.shape[1] * height / original.shape[0]), height))
        preprocessed_resized = cv2.resize(preprocessed, (int(preprocessed.shape[1] * height / preprocessed.shape[0]), height))
        
        # Create a side-by-side comparison
        comparison = np.hstack((original_resized, preprocessed_resized))
        
        # Save the comparison image
        comparison_path = "comparison_" + os.path.basename(image_path)
        cv2.imwrite(comparison_path, comparison)
        
        return {
            "original": original,
            "preprocessed": preprocessed,
            "comparison": comparison,
            "comparison_path": comparison_path
        }
    except Exception as e:
        print(f"Error in preview_image: {e}")
        return None

def plot_ecg(dat_file):
    """Generate a formatted ECG plot from a .dat file."""
    if not dat_file:
        return None

    record_name = os.path.splitext(dat_file)[0]  # Remove the extension (.dat)
    hea_file = record_name + ".hea"
    
    if not os.path.exists(hea_file):
        return None

    try:
        record = wfdb.rdrecord(record_name)
        signals = record.p_signal
        lead_names = record.sig_name

        # Define the order and structure of leads
        lead_order = [["I", "AVR", "V1", "V4"],
                      ["II", "AVL", "V2", "V5"],
                      ["III", "AVF", "V3", "V6"],
                      ["II"]]  # Rhythm strip at the bottom

        # Map lead names to their corresponding signal index
        lead_index = {name: i for i, name in enumerate(lead_names)}

        # ECG Plot Settings
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(4, 4, height_ratios=[1, 1, 1, 2])  # Rhythm strip has extra height

        # Grid background (ECG paper style)
        def add_grid(ax):
            ax.set_xticks(np.arange(0, 2, 0.2), minor=True)
            ax.set_yticks(np.arange(-2, 2, 0.5), minor=True)
            ax.grid(which="both", linestyle="--", linewidth=0.5, color="green")
            ax.grid(which="major", linestyle="-", linewidth=0.8, color="lightgray")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

        # Plot each lead in the defined structure
        for row in range(3):  # First three rows with four columns
            for col in range(4):
                lead_name = lead_order[row][col]
                if lead_name in lead_index:
                    ax = plt.subplot(gs[row, col])
                    ax.plot(signals[:, lead_index[lead_name]], color="red", linewidth=1)
                    add_grid(ax)
                    ax.set_title(lead_name, fontsize=10, loc='left')

        # Special case: Rhythm strip (Lead II spanning all columns)
        ax = plt.subplot(gs[3, :])  # Last row spans all four columns
        ax.plot(signals[:, lead_index["II"]], color="red", linewidth=1)
        add_grid(ax)
        ax.set_title("II (Rhythm Strip)", fontsize=10, loc='left')

        plt.tight_layout()

        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        return Image.open(buf)
    except Exception as e:
        print(f"Error in plot_ecg: {e}")
        return None

def generate_gradcam(image_path, model_path):
    """Generate Grad-CAM visualization for ECG image."""
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the trained model
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=4)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()

        # Preprocess function for inference
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Load and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image_pil = Image.fromarray(image_rgb)  
        input_tensor = preprocess(image_pil).unsqueeze(0).to(device)  

        # Generate Grad-CAM heatmap
        cam = GradCAMPlusPlus(model=model, target_layers=[model.patch_embed.proj])
        grayscale_cam = cam(input_tensor=input_tensor)[0]  

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
        predicted_label = class_names[predicted_class]

        # Save the visualization
        output_path = "gradcam_visualization.png"
        cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        
        return {
            "visualization": visualization,
            "predicted_class": predicted_label,
            "probabilities": probabilities,
            "class_names": class_names,
            "output_path": output_path
        }
    except Exception as e:
        print(f"Error in generate_gradcam: {e}")
        return None
    
def main():
    st.title("ECG Waveform Signal Extraction, Classification & Anomaly detection")

    # Ensure page state
    if "page" not in st.session_state:
        st.session_state.page = "upload"

    if st.session_state.page == "upload":
        upload_page()
    elif st.session_state.page == "results":
        result_page()

    # Retrieve uploaded file from session state
        if "file_path" in st.session_state:
            uploaded_file_path = st.session_state.file_path
            uploaded_file_name = st.session_state.file_name

            record = process_ecg_file(uploaded_file_path)
            if record:
                st.write("✅ ECG Data Processed Successfully!")
                
                # Generate ECG image
                ecg_image = plot_ecg(uploaded_file_path)
                if ecg_image is None:
                    st.error("❌ Failed to generate ECG image.")
                    return
                
                # **Define temp path for ECG image**
                # Instead of passing the image object directly, pass the file path of the saved image.
                temp_path = "temp_ecg_image.png"
                ecg_image.save(temp_path)  # Save the image properly

                # Check if the image is saved properly
                if not os.path.exists(temp_path):
                    st.error(f"❌ ECG Image file not found after saving: {temp_path}")
                    return

                
                # **Check if the model file exists**
                model_path = "ecg-classification-project/data/preprocessed/models/vit_ecg_classifier.pth"
                if not os.path.exists(model_path):
                    st.error("❌ Model file not found. Ensure the correct model path.")
                    return 

                # **Call Grad-CAM**
                gradcam_img = generate_gradcam(temp_path, model_path)


if __name__ == "__main__":
    main()