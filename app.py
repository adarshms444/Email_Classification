import streamlit as st
import joblib
import re
from PIL import Image

# Load models and vectorizer
nb_model = joblib.load("nb_model.pkl")
lr_model = joblib.load("log_reg_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Set page config
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

# Simple CSS for styling
st.markdown(
    """
    <style>
        .main {
            background-color: #ffffff;
            font-family: 'Arial', sans-serif;
        }
        h1 {
            text-align: center;
            color: #000000;
            font-size: 30px;
            font-weight: bold;
        }
        h2 {
            color: #4A4A4A;
            font-size: 20px;
            text-align: center;
        }
        .stButton>button {
            width: 100%;
            font-size: 18px;
            background-color: #4CAF50;
            color: white;
            padding: 10px;
            border: none;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextArea>div>textarea {
            font-size: 16px;
            background-color: #ffffff;
            color: #000000;
            padding: 10px;
            width: 100%;
            border: 1px solid #cccccc;
            border-radius: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for model selection
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Function to set selected model
def set_model(model_name):
    st.session_state.selected_model = model_name

# Header section
st.markdown("<h1>📧 Email Spam Detector</h1>", unsafe_allow_html=True)
st.write("### Select a model and enter text to detect spam!")

# Model selection buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🤖 Naïve Bayes"):
        set_model("nb")
with col2:
    if st.button("⚡ Logistic Regression"):
        set_model("lr")

# User input
user_input = st.text_area("✍️ Enter your email text:", height=150)

# Predict function
def predict(text):
    if st.session_state.selected_model is None:
        st.warning("⚠️ Please select a model first.")
        return None

    text_tfidf = vectorizer.transform([text])

    if st.session_state.selected_model == "nb":
        prediction = nb_model.predict(text_tfidf)[0]
    elif st.session_state.selected_model == "lr":
        prediction = lr_model.predict(text_tfidf)[0]

    return "🚀 Spam" if prediction == 1 else "✅ Not Spam"

# Predict button
if st.button("🔍 Predict"):
    result = predict(user_input)
    if result:
        st.write(result)
