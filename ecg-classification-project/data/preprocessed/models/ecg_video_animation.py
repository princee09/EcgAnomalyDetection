import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load .dat file
file_path = "ecg_signal_data/00008_hr.dat"
data = np.fromfile(file_path, dtype=np.int16)  # Read as 16-bit integers

# Convert to float to allow NaN handling
data = data.astype(np.float32)

# Handle NaN values (replace with the mean of valid values)
if np.isnan(data).any():
    data = np.nan_to_num(data, nan=np.nanmean(data))

# Reshape to (5000, 12) since PTB-XL has 12-lead ECG data
num_leads = 12
data = data.reshape(-1, num_leads)

# Select Lead II (index 1) for heart rate analysis
lead_index = 1  
ecg_signal = data[:5000, lead_index]  # First 10 sec (5000 samples)

# ECG Sampling Rate
sampling_rate = 500  # Hz

# Set up the animation
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(0, sampling_rate)  # Show 1-second window
ax.set_ylim(np.min(ecg_signal), np.max(ecg_signal))
ax.set_title("Continuous Scrolling ECG Animation")
ax.set_xlabel("Time (samples)")
ax.set_ylabel("Amplitude (µV)")

# Initialize line
line, = ax.plot([], [], lw=2)

def update(frame):
    start = frame * 10
    end = start + sampling_rate  # Show 1-second window
    if end > 5000:  # Max is 5000 samples (10 sec)
        start, end = 4500, 5000  # Last 1-second window
    line.set_data(np.arange(sampling_rate), ecg_signal[start:end])
    return line,

ani = animation.FuncAnimation(fig, update, frames=500, interval=20, blit=True)

# Save animation as MP4
output_video = "scrolling_ecg8.mp4"
ani.save(output_video, writer="ffmpeg", fps=25)
print(f"ECG animation saved as {output_video}")
