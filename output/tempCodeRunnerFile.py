def main():
    
    st.title("ECG Waveform Extraction & Classification")

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
            ecg_image = plot_ecg(uploaded_file_path)

            if ecg_image:
                temp_path = "temp_ecg_image.png"
                ecg_image.save(temp_path)
                st.image(ecg_image, caption="Generated ECG Image", use_column_width=True)

                # Extract ECG signal
                signal = extract_ecg_signal(temp_path)

                if len(signal) > 0:
                    # Plot the extracted ECG waveform
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(signal, color='blue')
                    ax.set_title("Extracted ECG Waveform")
                    ax.set_xlabel("Time (samples)")
                    ax.set_ylabel("Normalized Amplitude")
                    st.pyplot(fig)

                    # Call Grad-CAM generation with the saved image
                    gradcam_img = generate_gradcam(temp_path, "ecg-classification-project/data/preprocessed/models/vit_ecg_classifier.pth")
                    st.image(gradcam_img, caption="Grad-CAM Classification Heatmap", use_column_width=True)
                else:
                    st.error("No ECG waveform detected. Try another image.")
        else:
            st.error("Error processing ECG data.")
