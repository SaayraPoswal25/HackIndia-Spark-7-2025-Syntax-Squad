from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Matches Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model variable
global_model = None
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

@app.on_event("startup")
async def startup_event():
    global global_model
    model_path = "global_model.h5"  # Correct model file name
    
    # Check if model file exists
    if not os.path.exists(model_path):
        logger.error(f"Model file {model_path} not found!")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Files in directory: {os.listdir('.')}")
    else:
        logger.info(f"Loading model from {model_path}...")
        try:
            global_model = load_model(model_path)
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Brain Tumor Classification API is running"}

@app.post("/predict-cancer/")
async def predict_image(file: UploadFile = File(...)):
    global global_model
    
    if global_model is None:
        logger.error("Model not loaded")
        return JSONResponse(
            content={"error": "Model not loaded. Please check server logs for model loading issues."},
            status_code=500
        )
    
    try:
        # Validate file size (max 10MB)
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            logger.warning(f"File too large: {len(contents)} bytes")
            return JSONResponse(
                content={"error": "File too large. Maximum size is 10MB."},
                status_code=400
            )

        # Validate and preprocess image
        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception as e:
            logger.warning(f"Invalid image file: {str(e)}")
            return JSONResponse(
                content={"error": f"Invalid image file: {str(e)}. Please upload a valid PNG or JPEG image."},
                status_code=400
            )

        try:
            image = image.resize((128, 128))
            img_array = img_to_array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            return JSONResponse(
                content={"error": f"Image preprocessing failed: {str(e)}"},
                status_code=500
            )

        # Predict using global model
        logger.info(f"Predicting for file: {file.filename}")
        try:
            predictions = global_model.predict(img_array)
            predicted_class_index = int(np.argmax(predictions, axis=1)[0])
            confidence_score = float(np.max(predictions))
            tumor_type = class_labels[predicted_class_index]
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return JSONResponse(
                content={"error": f"Prediction failed: {str(e)}"},
                status_code=500
            )

        # Format result
        result = "No Cancer" if tumor_type == 'notumor' else "Cancer"
        logger.info(f"Predicted class: {tumor_type}, Confidence: {confidence_score}")

        return JSONResponse(content={
            "prediction": result,
            "confidence": confidence_score,
            "tumor_type": tumor_type
        })

    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"error": f"Unexpected error: {str(e)}"},
            status_code=500
        )

if _name_ == "_main_":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)