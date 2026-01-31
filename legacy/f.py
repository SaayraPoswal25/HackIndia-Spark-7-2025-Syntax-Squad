from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse
import uvicorn
import logging
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Store model in memory or temporary file
MODEL_FILE = None
TEMP_MODEL_PATH = None

# Initialize with a default model file (must exist initially)
def load_initial_model():
    global MODEL_FILE, TEMP_MODEL_PATH
    try:
        with open("global_model.h5", "rb") as f:
            MODEL_FILE = f.read()
        logger.info("Initial model loaded into memory")
        # Create a temporary file for server-side storage
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(MODEL_FILE)
            TEMP_MODEL_PATH = tmp.name
        logger.info(f"Model stored temporarily at {TEMP_MODEL_PATH}")
    except FileNotFoundError:
        logger.error("Initial model.h5 not found. Please place it in the server directory.")
        raise Exception("Initial model.h5 not found.")

load_initial_model()

@app.get("/model")
async def download_model():
    if MODEL_FILE is None:
        logger.error("No model available on server")
        return JSONResponse(content={"error": "No model available."}, status_code=404)
    logger.info("Sending model to client")
    return Response(content=MODEL_FILE, media_type="application/octet-stream", headers={"Content-Disposition": "attachment; filename=model.h5"})

@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    global MODEL_FILE, TEMP_MODEL_PATH
    try:
        content = await file.read()
        MODEL_FILE = content
        logger.info("Model updated in memory")
        # Overwrite temporary file
        if TEMP_MODEL_PATH and os.path.exists(TEMP_MODEL_PATH):
            os.remove(TEMP_MODEL_PATH)
            logger.info("Deleted existing temporary model file")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(MODEL_FILE)
            TEMP_MODEL_PATH = tmp.name
        logger.info(f"Model saved to new temporary file: {TEMP_MODEL_PATH}")
        return JSONResponse(content={"message": "Model uploaded and overwritten successfully."}, status_code=200)
    except Exception as e:
        logger.error(f"Failed to upload model: {str(e)}")
        return JSONResponse(content={"error": f"Failed to upload model: {str(e)}"}, status_code=500)

if __name__ == "_main_":
    uvicorn.run(app, host="0.0.0.0", port=8000)