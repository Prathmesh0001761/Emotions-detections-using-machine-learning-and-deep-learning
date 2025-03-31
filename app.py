import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import time
import base64

# Function to encode image in base64
def get_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Custom CSS styling with animations and better UI
def apply_custom_css():
    background_image = "background.jpg"  # Ensure this file is in the same directory
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
            
            .stApp {{
                background: url("data:image/jpg;base64,{get_base64(background_image)}") no-repeat center center fixed;
                background-size: cover;
                font-family: 'Poppins', sans-serif;
            }}
            
            .content-container {{
                background-color: rgba(16, 29, 44, 0.85);
                backdrop-filter: blur(8px);
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
                animation: fadeIn 1s ease-out;
            }}
            
            h1 {{
                color: #00d4ff;
                text-align: center;
                font-weight: 700;
                text-shadow: 0 2px 10px rgba(0, 212, 255, 0.3);
                margin-bottom: 5px;
            }}
            
            .subheader {{
                color: #aaaaaa;
                text-align: center;
                font-size: 1.1rem;
                margin-bottom: 30px;
            }}
            
            .stTabs {{
                background-color: transparent;
            }}
            
            [data-baseweb="tab-list"] {{
                gap: 10px;
            }}
            
            [data-baseweb="tab"] {{
                background-color: rgba(16, 29, 44, 0.7) !important;
                border-radius: 10px !important;
                padding: 10px 20px !important;
                transition: all 0.3s ease !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
            }}
            
            [data-baseweb="tab"]:hover {{
                background-color: rgba(0, 212, 255, 0.1) !important;
                transform: translateY(-2px);
            }}
            
            [aria-selected="true"] {{
                background-color: rgba(0, 212, 255, 0.2) !important;
                color: #00d4ff !important;
                font-weight: 600;
                border: 1px solid #00d4ff !important;
            }}
            
            .stFileUploader {{
                margin: 20px 0;
                border: 2px dashed rgba(0, 212, 255, 0.3) !important;
                border-radius: 10px !important;
                padding: 25px !important;
                transition: all 0.3s ease !important;
                background-color: rgba(0, 0, 0, 0.2) !important;
            }}
            
            .stFileUploader:hover {{
                border-color: #00d4ff !important;
                box-shadow: 0 0 20px rgba(0, 212, 255, 0.2) !important;
            }}
            
            .stButton>button {{
                background: linear-gradient(45deg, #00d4ff, #0083ff) !important;
                color: white !important;
                border: none !important;
                border-radius: 25px !important;
                padding: 12px 30px !important;
                font-weight: 600 !important;
                transition: all 0.3s !important;
                box-shadow: 0 4px 15px rgba(0, 132, 255, 0.3) !important;
                width: 100%;
            }}
            
            .stButton>button:hover {{
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 20px rgba(0, 132, 255, 0.4) !important;
            }}
            
            .audio-container {{
                margin: 20px 0;
                border-radius: 10px;
                overflow: hidden;
                background-color: rgba(0, 0, 0, 0.3);
            }}
            
            .result-box {{
                background: rgba(0, 0, 0, 0.4);
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
                border-left: 4px solid #00d4ff;
                animation: fadeIn 0.8s ease-out;
            }}
            
            .emotion-display {{
                font-size: 1.8rem;
                margin: 10px 0;
                text-align: center;
            }}
            
            .confidence {{
                color: #aaaaaa;
                text-align: center;
                font-size: 0.9rem;
            }}
            
            @keyframes fadeIn {{
                from {{ opacity: 0; }}
                to {{ opacity: 1; }}
            }}
            
            @keyframes pulse {{
                0% {{ transform: scale(1); }}
                50% {{ transform: scale(1.02); }}
                100% {{ transform: scale(1); }}
            }}
            
            .pulse {{
                animation: pulse 2s infinite;
            }}
            
            .file-info {{
                background-color: rgba(0, 0, 0, 0.3);
                border-radius: 8px;
                padding: 12px;
                margin: 10px 0;
                font-size: 0.9rem;
            }}
        </style>
    """, unsafe_allow_html=True)

# Enhanced waveform plot with dark theme
def wave_plot(data, sampling_rate):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    # Create the waveplot with custom colors
    librosa.display.waveshow(data, sr=sampling_rate, color='#00d4ff', x_axis='s')
    
    # Customize the plot
    ax.spines['bottom'].set_color('#00d4ff')
    ax.spines['left'].set_color('#00d4ff')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.xlabel("Time (s)", color='white', fontsize=12)
    plt.ylabel("Amplitude", color='white', fontsize=12)
    plt.title("Audio Waveform Analysis", fontweight="bold", color='white', fontsize=14, pad=20)
    plt.grid(color='#333333', alpha=0.3)
    
    st.pyplot(fig)

# Enhanced CNN model prediction with emoji visualization
def prediction(data, sampling_rate, file_name):
    emotion_dict = {
        0: "😐 Neutral",
        1: "😌 Calm",
        2: "😊 Happy",
        3: "😢 Sad",
        4: "😠 Angry",
        5: "😨 Fear",
        6: "🤢 Disgust",
        7: "😲 Surprise"
    }
    
    model = load_model("models/CnnModel.h5")
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    X_test = np.expand_dims([mfccs], axis=2)
    
    with st.spinner("🔍 Analyzing emotions with CNN..."):
        time.sleep(1.5)  # Simulate processing time for better UX
        predict = model.predict(X_test)
    
    detected_emotion = emotion_dict[np.argmax(predict)]
    confidence = np.max(predict) * 100
    
    # Display file info
    st.markdown(f"""
        <div class="file-info">
            <strong>File:</strong> {file_name}<br>
            <strong>Duration:</strong> {len(data)/sampling_rate:.2f} seconds
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced result display
    st.markdown(f"""
        <div class="result-box pulse">
            <h3 style="color: #00d4ff; text-align: center; margin-bottom: 15px;">Emotion Detection Result</h3>
            <div class="emotion-display">{detected_emotion}</div>
            <div class="confidence">Confidence: {confidence:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

# Enhanced MLP model prediction
def prediction_mlp(data, sampling_rate, file_name):
    emotion_dict = {
        0: "😐 Neutral",
        1: "😌 Calm",
        2: "😊 Happy",
        3: "😢 Sad",
        4: "😠 Angry",
        5: "😨 Fear",
        6: "🤢 Disgust",
        7: "😲 Surprise"
    }
    
    model = joblib.load("models/MLP_model.pkl")
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    
    with st.spinner("🔍 Analyzing emotions with MLP..."):
        time.sleep(1.5)  # Simulate processing time for better UX
        detected_emotion = emotion_dict[model.predict([mfccs])[0]]
    
    # Display file info
    st.markdown(f"""
        <div class="file-info">
            <strong>File:</strong> {file_name}<br>
            <strong>Duration:</strong> {len(data)/sampling_rate:.2f} seconds
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced result display
    st.markdown(f"""
        <div class="result-box pulse">
            <h3 style="color: #00d4ff; text-align: center; margin-bottom: 15px;">Emotion Detection Result</h3>
            <div class="emotion-display">{detected_emotion}</div>
        </div>
    """, unsafe_allow_html=True)

# Main app function with ultimate UI
def main():
    apply_custom_css()
    
    # Header with animation
    st.markdown("""
        <div class="content-container">
            <h1>🎤 SPEECH EMOTION CLASSIFIER</h1>
            <div class="subheader">Discover the emotions hidden in voice patterns</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs with icons
    tab1, tab2 = st.tabs(["🧠 CNN Model", "🤖 MLP Model"])
    
    with tab1:
        st.markdown("""
            <div class="content-container">
                <h2 style="color: #00d4ff; margin-bottom: 20px;">Deep Learning Emotion Detection</h2>
                <p style="margin-bottom: 25px;">Upload an audio file to analyze emotions using our advanced Convolutional Neural Network model.</p>
        """, unsafe_allow_html=True)
        
        audio_file = st.file_uploader("Drag and drop your audio file here", 
                                    type=['wav', 'mp3', 'ogg'], 
                                    key="cnn_uploader",
                                    help="Supported formats: WAV, MP3, OGG | Max size: 200MB")
        
        if audio_file is not None:
            st.markdown('<div class="audio-container">', unsafe_allow_html=True)
            st.audio(audio_file, format='audio/wav')
            st.markdown('</div>', unsafe_allow_html=True)
            
            with st.spinner("🔄 Processing audio file..."):
                data, sampling_rate = librosa.load(audio_file)
                time.sleep(1)  # Simulate loading for better UX
            
            wave_plot(data, sampling_rate)
            prediction(data, sampling_rate, audio_file.name)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
            <div class="content-container">
                <h2 style="color: #00d4ff; margin-bottom: 20px;">Machine Learning Emotion Detection</h2>
                <p style="margin-bottom: 25px;">Upload an audio file to analyze emotions using our efficient Multi-Layer Perceptron model.</p>
        """, unsafe_allow_html=True)
        
        audio_file = st.file_uploader("Drag and drop your audio file here", 
                                    type=['wav', 'mp3', 'ogg'], 
                                    key="mlp_uploader",
                                    help="Supported formats: WAV, MP3, OGG | Max size: 200MB")
        
        if audio_file is not None:
            st.markdown('<div class="audio-container">', unsafe_allow_html=True)
            st.audio(audio_file, format='audio/wav')
            st.markdown('</div>', unsafe_allow_html=True)
            
            with st.spinner("🔄 Processing audio file..."):
                data, sampling_rate = librosa.load(audio_file)
                time.sleep(1)  # Simulate loading for better UX
            
            wave_plot(data, sampling_rate)
            prediction_mlp(data, sampling_rate, audio_file.name)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    
        st.markdown("""
            <div class="content-container" style="animation: fadeIn 1s ease-out;">
                
                
                <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 25px;">
                    This <strong>Speech Emotion Recognition (SER)</strong> system uses cutting-edge machine learning 
                    techniques to detect emotions from speech signals. The application provides two distinct 
                    approaches to emotion detection, each with its own strengths:
                </p>
                
                <div style="background: rgba(0, 0, 0, 0.3); border-radius: 10px; padding: 20px; margin: 20px 0;">
                    <h3 style="color: #00d4ff; margin-top: 0;">🧠 CNN Model</h3>
                    <p>
                        Our <strong>Convolutional Neural Network</strong> model offers state-of-the-art deep learning-based 
                        emotion detection with superior accuracy. It processes the audio's MFCC features through 
                        multiple convolutional layers to identify complex emotional patterns in speech.
                    </p>
                </div>
                
                <div style="background: rgba(0, 0, 0, 0.3); border-radius: 10px; padding: 20px; margin: 20px 0;">
                    <h3 style="color: #00d4ff; margin-top: 0;">🤖 MLP Model</h3>
                    <p>
                        The <strong>Multi-Layer Perceptron</strong> model provides a more traditional machine learning 
                        approach to emotion detection. While slightly less accurate than the CNN model, it offers 
                        faster processing times and requires fewer computational resources.
                    </p>
                </div>
                
                <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid rgba(255, 255, 255, 0.1);">
                    <h3 style="color: #00d4ff; margin-bottom: 15px;">Project Details</h3>
                    <div style="background: rgba(0, 0, 0, 0.3); border-radius: 8px; padding: 15px;">
                        <p style="margin-bottom: 8px;"><strong>Created by:</strong> Prathmesh Kangane</p>
                        <p style="margin-bottom: 8px;"><strong>Version:</strong> 2.0 (Enhanced UI)</p>
                        <p><strong>Purpose:</strong> Educational demonstration of SER technology</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()