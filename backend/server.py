import os
# Set TensorFlow environment variables BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Depends, HTTPException, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import shutil
import logging
from typing import List, Optional
import numpy as np
import secrets
# Delay TensorFlow import until needed to avoid mutex issues
# import tensorflow as tf
from contextlib import asynccontextmanager
from pydantic import BaseModel

from utils import aggregate_weights, load_private_key, decrypt_payload, generate_keys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FL-Server")

# Constants
MODEL_PATH = "models/global_model.h5"
KEYS_DIR = "keys"
MIN_CLIENTS_FOR_AGGREGATION = 2  # For demo purposes, we can set this low

# Global state
pending_updates = []  # Stores (client_id, weights)
global_model = None

# Simple authentication (for demo purposes)
# In production, use proper authentication with database
AUTH_USERS = {
    "admin": "admin123",
    "user1": "password1",
    "doctor": "doctor123"
}
active_sessions = {}  # token -> username
security = HTTPBearer()

class LoginRequest(BaseModel):
    username: str
    password: str

def load_model_if_needed():
    """Lazy load the model only when needed"""
    global global_model
    if global_model is None:
        try:
            # Import TensorFlow only when needed
            import tensorflow as tf
            if os.path.exists(MODEL_PATH):
                global_model = tf.keras.models.load_model(MODEL_PATH)
                logger.info("Global model loaded successfully.")
            else:
                logger.warning("Model file not found, will create on first use")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    return global_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global global_model
    
    # 1. Ensure keys exist
    if not os.path.exists(os.path.join(KEYS_DIR, "private_key.pem")):
        logger.info("Generating new RSA keys...")
        os.makedirs(KEYS_DIR, exist_ok=True)
        generate_keys(KEYS_DIR)
        
    # 2. Ensure model directory exists
    if not os.path.exists(MODEL_PATH):
        logger.info("No global model found. Will create on first request...")
        os.makedirs("models", exist_ok=True)
    
    # Don't load model at startup to avoid mutex errors
    logger.info("Server started. Model will be loaded on first request.")
        
    yield
    
    # Shutdown logic
    pass

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_initial_model():
    """Creates a basic VGG16-based model for the demo if none exists."""
    # Import TensorFlow only when needed
    import tensorflow as tf
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
    from tensorflow.keras.models import Sequential
    
    base_model = VGG16(include_top=False, input_shape=(128, 128, 3), weights='imagenet')
    base_model.trainable = False  # Freeze base
    
    model = Sequential([
        Input(shape=(128, 128, 3)),
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  # 4 classes: glioma, meningioma, notumor, pituitary
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.save(MODEL_PATH)
    logger.info(f"Initial model created and saved to {MODEL_PATH}")

@app.get("/")
def read_root():
    return {"status": "active", "message": "Secure Federated Learning Server Online"}

@app.get("/model")
def download_model(authorization: Optional[str] = Header(None)):
    """Clients download the global model from here. Authentication optional for FL clients."""
    if not os.path.exists(MODEL_PATH):
        # Try to create initial model if it doesn't exist
        try:
            create_initial_model()
        except Exception as e:
            logger.error(f"Failed to create initial model: {e}")
            return JSONResponse({"error": "Model not found and could not be created"}, status_code=404)
    
    # Optional: log if authenticated user downloads
    if authorization:
        token = authorization.replace("Bearer ", "").strip()
        if token in active_sessions:
            logger.info(f"Model downloaded by authenticated user: {active_sessions[token]}")
    
    return FileResponse(MODEL_PATH, media_type="application/octet-stream", filename="global_model.h5")

@app.get("/public_key")
def get_public_key():
    """Clients get the public key to encrypt their updates."""
    key_path = os.path.join(KEYS_DIR, "public_key.pem")
    if not os.path.exists(key_path):
        return JSONResponse({"error": "Keys not generated yet"}, status_code=500)
    return FileResponse(key_path, media_type="application/x-pem-file", filename="public_key.pem")

@app.post("/update")
async def receive_update(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Receives an encrypted model update from a client.
    """
    try:
        # Read encrypted payload
        encrypted_content = await file.read()
        
        # Load private key
        private_key = load_private_key(os.path.join(KEYS_DIR, "private_key.pem"))
        
        # Decrypt
        decrypted_data = decrypt_payload(encrypted_content, private_key)
        
        # Extract weights (and potentially client_id, metrics, etc.)
        # Assuming decrypted_data is {'weights': [...], 'client_id': '...', 'samples': 100}
        client_weights = decrypted_data.get('weights')
        client_id = decrypted_data.get('client_id', 'unknown')
        
        if not client_weights:
            return JSONResponse({"error": "No weights found in payload"}, status_code=400)
            
        logger.info(f"Received update from client {client_id}")
        
        # Add to pending updates
        pending_updates.append(client_weights)
        
        # Check if we should aggregate
        if len(pending_updates) >= MIN_CLIENTS_FOR_AGGREGATION:
            background_tasks.add_task(perform_aggregation)
            return JSONResponse({"status": "accepted", "message": "Update received. Aggregation triggered."})
            
        return JSONResponse({"status": "accepted", "message": "Update received. Waiting for more clients."})
        
    except Exception as e:
        logger.error(f"Error processing update: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/status")
def get_status():
    return {
        "round": 1, # Todo: persistent round tracking
        "pending_updates": len(pending_updates),
        "required_updates": MIN_CLIENTS_FOR_AGGREGATION,
        "clients_online": 0 # Real-time tracking requires websockets, skipping for now
    }

# Authentication endpoints for Streamlit interface
@app.post("/login")
def login(credentials: LoginRequest):
    """Login endpoint for Streamlit fine-tuning interface."""
    username = credentials.username
    password = credentials.password
    
    if username in AUTH_USERS and AUTH_USERS[username] == password:
        # Check if user is already logged in
        for token, user in active_sessions.items():
            if user == username:
                return {"token": token, "message": "Already logged in"}
        
        # Generate token
        token = secrets.token_urlsafe(32)
        active_sessions[token] = username
        logger.info(f"User {username} logged in successfully")
        return {"token": token, "message": "Login successful"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/logout")
def logout(authorization: Optional[str] = Header(None)):
    """Logout endpoint for Streamlit fine-tuning interface."""
    if authorization:
        token = authorization.replace("Bearer ", "").strip()
        if token in active_sessions:
            username = active_sessions.pop(token)
            logger.info(f"User {username} logged out")
            return {"message": "Logout successful"}
    raise HTTPException(status_code=401, detail="Not authenticated")

@app.get("/check_session")
def check_session():
    """Check if any user is currently logged in."""
    if active_sessions:
        # Return first active user (simple implementation)
        token, username = next(iter(active_sessions.items()))
        return {"logged_in": True, "user": username}
    return {"logged_in": False}

def verify_token(authorization: Optional[str] = Header(None)):
    """Verify authentication token."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.replace("Bearer ", "").strip()
    if token not in active_sessions:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return active_sessions[token]

@app.post("/upload_model")
async def upload_model(
    file: UploadFile = File(...),
    username: str = Depends(verify_token)
):
    """Upload a fine-tuned model (requires authentication)."""
    try:
        # Save uploaded model
        content = await file.read()
        
        # Backup existing model if it exists
        if os.path.exists(MODEL_PATH):
            backup_path = f"{MODEL_PATH}.backup"
            shutil.copy(MODEL_PATH, backup_path)
            logger.info(f"Backed up existing model to {backup_path}")
        
        # Write new model
        os.makedirs("models", exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            f.write(content)
        
        # Invalidate cached model
        global global_model
        global_model = None
        
        logger.info(f"Model uploaded by {username}")
        return {"message": "Model uploaded successfully", "uploaded_by": username}
    except Exception as e:
        logger.error(f"Error uploading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload model: {str(e)}")

def perform_aggregation():
    global pending_updates, global_model
    
    logger.info("Starting Federated Aggregation (FedAvg)...")
    
    if not pending_updates:
        return
    
    # Load model if not already loaded
    model = load_model_if_needed()
    if model is None:
        logger.error("Cannot perform aggregation: model not available")
        return
    
    # Aggregate
    new_weights = aggregate_weights(pending_updates)
    
    # Update global model
    model.set_weights(new_weights)
    
    # Save new global model
    model.save(MODEL_PATH)
    
    logger.info(f"Global model updated with {len(pending_updates)} client contributions.")
    
    # Clear pending updates
    pending_updates = []

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
