# ECG Analysis System

## Overview

The ECG Analysis System is a comprehensive web application built with Streamlit that provides advanced electrocardiogram (ECG) signal analysis and classification capabilities. The system offers multiple methods for ECG analysis through an intuitive user interface with a beautiful design.

## Features

- **Multi-modal ECG Analysis**: Analyze ECG data from both signal files and images
- **Signal-based Detection**: Process ECG signal files for heart condition classification
- **Image-based Detection**: Extract and analyze ECG patterns from uploaded ECG images
- **Advanced Visualization**: View processed ECG signals with detailed annotations
- **GradCAM Visualization**: Understand model decisions through gradient-based class activation mapping
- **Comprehensive Reporting**: Generate detailed PDF reports of analysis results

## Technologies Used

- **Frontend**: Streamlit with custom CSS for an interactive user experience
- **Signal Processing**: wfdb, scipy, neurokit2 for ECG signal analysis
- **Image Processing**: OpenCV, Pillow for ECG image preprocessing
- **Machine Learning**: PyTorch, torchvision, timm for deep learning models
- **Visualization**: Matplotlib, seaborn for data visualization
- **Reporting**: FPDF for PDF report generation

## Project Structure

- `Streamlit.py`: Main application entry point with navigation setup
- `output/`: Contains the application pages:
  - `home_page.py`: Landing page with project information
  - `newapp.py`: Signal-based ECG detection functionality
  - `new_image_input.py`: Image-based ECG detection functionality
- `requirements.txt`: Lists all Python dependencies
- `runtime.txt`: Specifies the Python version
- `setup.sh`: Installs system dependencies (ffmpeg)
- `render.yaml`: Configuration for deployment

## Installation and Setup

### Prerequisites

- Python 3.10 or higher
- ffmpeg (installed automatically via setup.sh)

### Installation

1. Clone the repository
2. Install the required dependencies:

```
pip install -r requirements.txt
```

3. Install system dependencies:

```
bash setup.sh
```

### Running the Application

To start the application locally:

```
streamlit run Streamlit.py
```

The application will be available at http://localhost:8501

## Dependencies

All required Python packages are listed in `requirements.txt`. Key dependencies include:

- streamlit==1.32.0
- matplotlib==3.8.2
- numpy==1.26.3
- wfdb==4.1.2
- opencv-python==4.8.1.78
- torch==2.1.2
- torchvision==0.16.2
- timm==0.9.12
- pytorch-grad-cam==0.4.1
- neurokit2==0.2.7
- scikit-learn==1.3.2

System dependencies like ffmpeg will be installed automatically during setup through the `setup.sh` script.