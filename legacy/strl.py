import requests
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import logging
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
global global_model
model_path = "global_model.h5"
SERVER_URL = "http://localhost:8000"

# Initialize session state for TEMP_MODEL_PATH
if 'TEMP_MODEL_PATH' not in st.session_state:
    st.session_state.TEMP_MODEL_PATH = None

def download_model():
    try:
        # Clean up existing temporary file
        if st.session_state.TEMP_MODEL_PATH and os.path.exists(st.session_state.TEMP_MODEL_PATH):
            os.remove(st.session_state.TEMP_MODEL_PATH)
            logger.info("Deleted existing temporary model file")
            st.session_state.TEMP_MODEL_PATH = None
        # Download model
        response = requests.get(f"{SERVER_URL}/model")
        if response.status_code != 200:
            logger.error(f"Failed to download model: {response.json()['error']}")
            return False, f"Failed to download model: {response.json()['error']}"
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(response.content)
            st.session_state.TEMP_MODEL_PATH = tmp.name
        logger.info(f"Model downloaded and saved to temporary file: {st.session_state.TEMP_MODEL_PATH}")
        return True, "Model downloaded successfully."
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return False, f"Error downloading model: {str(e)}"

def predict_labels(model_path: str, image_files: list):
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded for prediction from {model_path}")

        x_data = []
        for img_file in image_files:
            img = Image.open(img_file)
            img = img.resize((128, 128))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img) / 255.0
            x_data.append(img_array)
        x_data = np.array(x_data)
        logger.info(f"Image data prepared for prediction, shape: {x_data.shape}")

        # Predict labels
        predictions = model.predict(x_data)
        predicted_labels = np.argmax(predictions, axis=1)
        logger.info(f"Predicted labels: {predicted_labels.tolist()}")
        return predicted_labels
    except Exception as e:
        logger.error(f"Failed to predict labels: {str(e)}")
        raise Exception(f"Failed to predict labels: {str(e)}")

def fine_tune_model(model_path: str, image_files: list, epochs: int = 5):
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded for fine-tuning from {model_path}")

        # Predict labels for the images
        y_train = predict_labels(model_path, image_files)
        
        x_train = []
        for img_file in image_files:
            img = Image.open(img_file)
            img = img.resize((128, 128))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img) / 255.0
            x_train.append(img_array)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        logger.info(f"Image data prepared for fine-tuning, x_train shape: {x_train.shape}, y_train: {y_train.tolist()}")

        # Verify input shape
        expected_shape = (x_train.shape[0], 128, 128, 3)
        if x_train.shape[1:] != expected_shape[1:]:
            raise ValueError(f"Prepared data shape {x_train.shape[1:]} does not match model input shape {expected_shape[1:]}")

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=epochs, validation_split=0.2)
        logger.info(f"Model fine-tuned for {epochs} epochs")

        # Overwrite the temporary file
        if os.path.exists(model_path):
            os.remove(model_path)
            logger.info("Deleted existing model before saving")
        model.save(model_path)
        logger.info(f"Model saved and overwritten at {model_path}")
        return True, "Model fine-tuned successfully."
    except Exception as e:
        logger.error(f"Failed to fine-tune model: {str(e)}")
        return False, f"Failed to fine-tune model: {str(e)}"

def upload_model(model_path: str):
    try:
        with open(model_path, "rb") as f:
            response = requests.post(f"{SERVER_URL}/upload_model", files={"file": f})
        if response.status_code != 200:
            logger.error(f"Failed to upload model: {response.json()['error']}")
            return False, f"Failed to upload model: {response.json()['error']}"
        logger.info("Model uploaded and overwritten on server")
        return True, "Model uploaded successfully."
    except Exception as e:
        logger.error(f"Error uploading model: {str(e)}")
        return False, f"Error uploading model: {str(e)}"

st.title("Model Fine-Tuning Interface")

server_url = st.text_input("Server URL", value=SERVER_URL)
image_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

epochs = st.slider("Number of Epochs", min_value=1, max_value=20, value=5)

if st.button("Download Model"):
    with st.spinner("Downloading model..."):
        success, message = download_model()
        if success:
            st.success(message)
        else:
            st.error(message)

if st.button("Fine-Tune and Upload"):
    if not image_files:
        st.error("Please upload at least one image.")
    elif not st.session_state.TEMP_MODEL_PATH or not os.path.exists(st.session_state.TEMP_MODEL_PATH):
        st.error("Please download the model first.")
    else:
        with st.spinner("Processing..."):
            success, message = fine_tune_model(st.session_state.TEMP_MODEL_PATH, image_files, epochs)
            if success:
                st.success(message)
                success, message = upload_model(st.session_state.TEMP_MODEL_PATH)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.error(message)