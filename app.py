from io import BytesIO
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from helper import processing_img
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

model = load_model("./models/model.keras")

emotion_labels = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happiness",
    4: "Sadness",
    5: "Surprise",
    6: "Neutral"
}

st.set_page_config(page_title="Emotion Recognition", page_icon="😊", layout="wide")
st.title("Face Expression Recognition App")

st.markdown("""
    <div style="font-size: 1.2rem; color: #555;">
        Upload an image or record a video, and I will predict the emotion (happy, sad, angry, etc.)!
    </div>
""", unsafe_allow_html=True)

way = st.radio("Choose how to provide input:", ("Upload an image", "Take an image", "Record a video"))

if way == "Upload an image":
    image_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if image_file:
        progress = st.progress(0)
        st.write("Processing Image...")
        
        try:
            img = Image.open(image_file)
            img = np.array(img)
            processed_img = processing_img(img)
        except Exception as e:
            st.error(f"Error processing image: {e}")

        progress.progress(50)
        
        pred = model.predict(processed_img)
        emotion_idx = np.argmax(pred, axis=1)
        emotion = emotion_labels[emotion_idx[0]]

        progress.progress(100)

        st.image(img, caption="Uploaded Image", use_container_width=False, width=270)
        st.success(f"Emotion: {emotion} detected!")

elif way == "Take an image":
    st.text("Click below to take a picture.")
    
    video_file = st.camera_input("Take a picture")

    if video_file is not None:
        try:
            image_bytes = video_file.getvalue()
            
            frame = Image.open(BytesIO(image_bytes))
            frame = np.array(frame)

            processed_frame = processing_img(frame)
            pred = model.predict(processed_frame)
            emotion_idx = np.argmax(pred, axis=1)
            emotion = emotion_labels[emotion_idx[0]]

            st.image(frame, caption=f"Predicted Emotion: {emotion}", width=270)
            st.success(f"Emotion: {emotion} detected!")
        except Exception as e:
            st.error(f"Error processing the image: {e}")




elif way == "Record a video":
    webrtc_streamer(
            key="key",
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            video_processor_factory=None,
            media_stream_constraints={"video": True, "audio": False},
            )

st.markdown("""
    <footer style="font-size: 14px; text-align: center; padding-top: 20px; color: white;text-shadow: 1px 1px 3px white;">
        Created by Mohamed Medhat Elnggar
    </footer>
""", unsafe_allow_html=True)
