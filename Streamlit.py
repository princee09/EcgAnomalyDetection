import streamlit as st
import base64

# Set page config first
st.set_page_config(
    page_title="ECG Analysis System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────── 🎬 Add video background ───────

# Path to your local video
video_path = r"C:\Users\suman\OneDrive\Desktop\ECG_Project\WhatsApp Video 2025-04-26 at 10.08.23 PM.mp4"

try:
    # Read and encode video to base64
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()
        encoded_video = base64.b64encode(video_bytes).decode()

    # Inject video background into the app
    st.markdown(f"""
        <style>
            .video-background {{
                position: fixed;
                top: 0;
                left: 0;
                min-width: 100%;
                min-height: 100%;
                z-index: -1;
                opacity: 0.4;
            }}
            .stApp {{
                background-color: transparent;
            }}
            .stApp > header {{
                background-color: transparent;
            }}
        </style>

        <video autoplay loop muted playsinline class="video-background">
            <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
        </video>
    """, unsafe_allow_html=True)
except FileNotFoundError:
    st.error("⚠️ ECG video not found. Please check the file path.")

# ─────── 🎨 Keep your original gradient overlay ───────
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, rgba(255,255,255,0.1));
        }
    </style>
""", unsafe_allow_html=True)

# ─────── 🧭 Your original navigation setup ───────
about_page = st.Page(
    page="output/home_page.py",
    title="Home",
    icon=":material/home:",
    default=True,
)

project_1_page = st.Page(
    page="output/newapp.py",
    title="Detection using Signal",
    icon=":material/ecg:",
)

project_2_page = st.Page(
    page="output/new_image_input.py",
    title="Detection using Image",
    icon=":material/image:",
)

pg = st.navigation(pages=[about_page, project_1_page, project_2_page])

# Run the selected page
pg.run()
