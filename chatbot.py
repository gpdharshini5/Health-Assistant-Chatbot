import streamlit as st
import cv2
from deepface import DeepFace
import requests

# Define the Gemini API endpoint and your API key
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
API_KEY = "Key"

# Function to predict emotion from the image
def predict_emotion(image):
    result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
    return result[0]['dominant_emotion']

# Function to get a dynamic response from Gemini API based on emotion
def get_gemini_response(emotion):
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [{
            "parts": [{"text": f"I am feeling {emotion}. Can you help me feel better?"}]
        }]
    }
    response = requests.post(f"{GEMINI_API_URL}?key={API_KEY}", json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get('content')
    else:
        return "Sorry, I couldn't process your request at the moment."

# Streamlit app layout
st.title("Real-time Emotion-based Consoling Chatbot")
st.subheader("Let’s see how you’re feeling! Your emotions will be detected in real-time.")

# Streamlit webcam widget to show live feed
camera_placeholder = st.empty()
response_placeholder = st.empty()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        st.write("Error: Failed to capture image.")
        break

    emotion = predict_emotion(frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    camera_placeholder.image(frame_rgb, caption="Real-time Webcam Feed", use_column_width=True)
    response_placeholder.text(f"Detected Emotion: {emotion}")
    response = get_gemini_response(emotion)
    response_placeholder.text(f"Chatbot Response: {response}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
