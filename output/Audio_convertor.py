import wfdb
import numpy as np
import scipy.io.wavfile as wav
import os

def ecg_to_audio(dat_file, output_wav, sample_rate=44100):
    try:
        # Read ECG data
        record_name = os.path.splitext(dat_file)[0]  # Remove .dat extension
        record = wfdb.rdrecord(record_name)  # Load the record
        signals = record.p_signal  # Extract ECG signals (12 leads)
        lead_ii = signals[:, 1]  # Use Lead II as main audio representation

        # Normalize ECG to range [-1, 1] (like an audio signal)
        lead_ii = lead_ii / np.max(np.abs(lead_ii))
        
        # Resample to audio frequency
        audio_signal = np.interp(
            np.linspace(0, len(lead_ii), int(len(lead_ii) * sample_rate / record.fs)),
            np.arange(len(lead_ii)),
            lead_ii
        )

        # Convert to 16-bit PCM format for WAV file
        audio_signal = (audio_signal * 32767).astype(np.int16)

        # Save as WAV file
        wav.write(output_wav, sample_rate, audio_signal)
        print(f"Saved audio: {output_wav}")
    
    except Exception as e:
        print(f"Error processing {dat_file}: {e}")

# Example usage
ecg_to_audio(
    dat_file="C:\\Users\\pbimu\\Downloads\\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\\21000_hr\\21003_hr.dat", 
    output_wav="C:\\Users\\pbimu\\Downloads\\ECG_Output\\21003_hr.wav"
)
