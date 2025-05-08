import streamlit as st
from PIL import Image

def main():
    # Remove st.set_page_config from here
    
    # Custom CSS with interactive elements
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100;300;400;500;700&display=swap');
            
            .main-title {
                color: white;
                font-size: 3.8em;
                font-weight: 700;
                margin-bottom: 1em;
                text-align: center;
                text-shadow: 0 0 10px rgba(255,255,255,0.5),
                           0 0 20px rgba(255,255,255,0.3),
                           0 0 30px rgba(255,255,255,0.2);
                letter-spacing: 2px;
            }
            
            .feature-card {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 2rem;
                margin: 1.2rem 0;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                transition: all 0.3s ease;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            
            .feature-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            }
            
            .feature-text {
                color: #2c3e50;
                font-size: 1.1em;
                line-height: 1.6;
                font-weight: 400;
            }
            
            .subtitle {
                color: 	#f5f5f5;
                font-size: 1.3em;
                font-weight: 500;
                margin-bottom: 3em;
                text-align: center;
                opacity: 0.5;
            }
            
            .feature-title {
                color: #2c3e50;
                font-size: 1.6em;
                font-weight: 600;
                margin-bottom: 0.8em;
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
            }
            
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(32, 201, 151, 0.3);
                background: linear-gradient(45deg, #20c997, #1a8754);
            }
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            }
            
            @keyframes ecgAnimation {
                0% { stroke-dashoffset: 2000; }
                100% { stroke-dashoffset: 0; }
            }
            
            .ecg-background {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                z-index: -1;
                opacity: 0.3;
            }
            
            .ecg-line {
                stroke: white;
                stroke-width: 4;
                fill: none;
                stroke-dasharray: 2000;
                stroke-dashoffset: 2000;
                animation: ecgAnimation 8s linear infinite;
                filter: drop-shadow(0 0 10px rgba(255, 255, 255, 0.9))
                       drop-shadow(0 0 15px rgba(255, 255, 255, 0.5));
            }

            .ecg-container {
                width: 100%;
                height: 100%;
                overflow: hidden;
            }
        </style>
    """, unsafe_allow_html=True)


    
    st.markdown('<h1 class="main-title">ECG Analysis & Anomaly Detection</h1>', unsafe_allow_html=True)
    
    st.markdown("""
        <p class="subtitle">
            Advanced AI-powered ECG analysis system for medical professionals
        </p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        signal_card = st.container()
        with signal_card:
            st.markdown("""
                <div class="feature-card">
                    <div style="font-size: 3em; text-align: center;">📊</div>
                    <h2 style="color: #3498db; text-align: center; font-size: 1.5em;">Signal Analysis</h2>
                    <p style="text-align: center; color: #7f8c8d;">
                        Upload ECG signals for detailed waveform analysis and instant heart rate calculation
                    </p>
                </div>
            """, unsafe_allow_html=True)
        if st.button("Go to Signal Analysis", key="signal_btn", use_container_width=True):
            st.switch_page("output/newapp.py")
    
    with col2:
        image_card = st.container()
        with image_card:
            st.markdown("""
                <div class="feature-card">
                    <div style="font-size: 3em; text-align: center;">🔍</div>
                    <h2 style="color: #3498db; text-align: center; font-size: 1.5em;">Image Processing</h2>
                    <p style="text-align: center; color: #7f8c8d;">
                        Process ECG images with advanced AI detection and Grad-CAM visualization
                    </p>
                </div>
            """, unsafe_allow_html=True)
        if st.button("Go to Image Analysis", key="image_btn", use_container_width=True):
            st.switch_page("output/new_image_input.py")
    
    # Additional features section
    st.markdown("""
        <div style='text-align: center; margin-top: 3em; padding: 2em;'>
            <h2 style='color: #3498db; margin-bottom: 1em;'>Key Features</h2>
        </div>
    """, unsafe_allow_html=True)
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.markdown("""
            <div class="feature-card">
                <div style="font-size: 2em; text-align: center;">📈</div>
                <h3 style="color: #3498db; text-align: center;">Real-time Analysis</h3>
                <p style="text-align: center; color: #7f8c8d;">Instant results with detailed metrics</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="feature-card">
                <div style="font-size: 2em; text-align: center;">🎥</div>
                <h3 style="color: #3498db; text-align: center;">Video Generation</h3>
                <p style="text-align: center; color: #7f8c8d;">Animated ECG visualizations</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
            <div class="feature-card">
                <div style="font-size: 2em; text-align: center;">📱</div>
                <h3 style="color: #3498db; text-align: center;">PDF Reports</h3>
                <p style="text-align: center; color: #7f8c8d;">Detailed analysis reports</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()