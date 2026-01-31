# Model Download & Fine-Tuning Guide

## 🎯 Overview

This guide shows you **all the ways** to download, fine-tune, and manage your Federated Learning model.

---

## 📥 **Method 1: Streamlit Web Interface (Recommended for Fine-Tuning)**

### Access
- **URL**: http://localhost:8501
- **Status**: ✅ Running

### Features
- ✅ User-friendly web interface
- ✅ Download model with one click
- ✅ Upload images for fine-tuning
- ✅ Fine-tune model with configurable epochs
- ✅ Upload fine-tuned model back to server
- ✅ Secure authentication

### Steps to Use:

1. **Open Streamlit Interface**
   ```
   http://localhost:8501
   ```

2. **Login**
   - Username: `admin` / Password: `admin123`
   - Or: `user1` / `password1`
   - Or: `doctor` / `doctor123`

3. **Download Model**
   - Click "Download Model" button
   - Model will be downloaded to temporary storage

4. **Fine-Tune Model**
   - Upload MRI images (PNG/JPG/JPEG)
   - Select number of epochs (1-20)
   - Click "Fine-Tune and Upload"
   - Model will be fine-tuned and automatically uploaded

---

## 📥 **Method 2: Direct API Download (For Programmatic Access)**

### Endpoint
```
GET http://localhost:8000/model
```

### Usage Examples:

#### Using cURL:
```bash
curl -O http://localhost:8000/model
# Saves as 'model' file, rename to global_model.h5
```

#### Using Python:
```python
import requests

response = requests.get("http://localhost:8000/model")
with open("global_model.h5", "wb") as f:
    f.write(response.content)
print("Model downloaded successfully!")
```

#### Using wget:
```bash
wget http://localhost:8000/model -O global_model.h5
```

#### Using Browser:
Simply open: http://localhost:8000/model
The browser will download the model file.

---

## 📥 **Method 3: Federated Learning Client Script**

### Location
```
client/client.py
```

### Usage
```bash
cd client
python3 client.py
```

### What it does:
1. Downloads global model from server
2. Downloads public key for encryption
3. Trains model locally (with your data)
4. Encrypts model updates
5. Uploads encrypted updates to server

### Configuration
Edit `client/client.py` to:
- Change `SERVER_URL` if needed
- Add your local training data in `LOCAL_DATA_DIR`
- Modify training parameters

---

## 🔧 **Fine-Tuning Options**

### Option A: Using Streamlit (Easiest)
- See Method 1 above
- No coding required
- Web-based interface

### Option B: Manual Fine-Tuning (Python)

```python
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 1. Download model (or use existing)
# model = tf.keras.models.load_model("global_model.h5")

# 2. Prepare your data
def load_images(image_paths):
    images = []
    for path in image_paths:
        img = Image.open(path)
        img = img.resize((128, 128))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img) / 255.0
        images.append(img_array)
    return np.array(images)

# 3. Load model
model = tf.keras.models.load_model("global_model.h5")

# 4. Prepare training data
x_train = load_images(["image1.jpg", "image2.jpg", ...])
y_train = np.array([0, 1, 2, 3, ...])  # Labels: 0=glioma, 1=meningioma, 2=notumor, 3=pituitary

# 5. Fine-tune
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_split=0.2)

# 6. Save fine-tuned model
model.save("fine_tuned_model.h5")
```

---

## 📤 **Uploading Fine-Tuned Model**

### Method 1: Streamlit Interface
- After fine-tuning, click "Fine-Tune and Upload"
- Model is automatically uploaded

### Method 2: API Upload (Requires Authentication)

```python
import requests

# 1. Login first
login_response = requests.post(
    "http://localhost:8000/login",
    json={"username": "admin", "password": "admin123"}
)
token = login_response.json()["token"]

# 2. Upload model
with open("fine_tuned_model.h5", "rb") as f:
    files = {"file": f}
    headers = {"Authorization": token}
    response = requests.post(
        "http://localhost:8000/upload_model",
        files=files,
        headers=headers
    )
print(response.json())
```

### Using cURL:
```bash
# 1. Login
TOKEN=$(curl -X POST http://localhost:8000/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' \
  | jq -r '.token')

# 2. Upload model
curl -X POST http://localhost:8000/upload_model \
  -H "Authorization: $TOKEN" \
  -F "file=@fine_tuned_model.h5"
```

---

## 🔐 **Authentication**

### Default Users:
- **admin** / **admin123**
- **user1** / **password1**
- **doctor** / **doctor123**

### Login Endpoint:
```
POST http://localhost:8000/login
Body: {"username": "admin", "password": "admin123"}
Response: {"token": "...", "message": "Login successful"}
```

### Logout Endpoint:
```
POST http://localhost:8000/logout
Headers: {"Authorization": "YOUR_TOKEN"}
```

---

## 📍 **All Available Endpoints**

| Endpoint | Method | Auth Required | Description |
|----------|--------|---------------|-------------|
| `/` | GET | No | Server status |
| `/model` | GET | Optional | Download global model |
| `/public_key` | GET | No | Get encryption public key |
| `/update` | POST | No | Upload encrypted FL update |
| `/status` | GET | No | Get FL status |
| `/login` | POST | No | Login for fine-tuning |
| `/logout` | POST | Yes | Logout |
| `/check_session` | GET | No | Check active sessions |
| `/upload_model` | POST | Yes | Upload fine-tuned model |

---

## 🚀 **Quick Start Commands**

### Start All Services:
```bash
# Backend (already running on port 8000)
cd backend && uvicorn server:app --host 0.0.0.0 --port 8000

# Frontend (already running on port 5173)
cd frontend && npm run dev

# Streamlit (already running on port 8501)
cd sever && streamlit run server.py --server.port 8501
```

### Download Model:
```bash
# Quick download
curl http://localhost:8000/model -o global_model.h5
```

### Run Federated Learning Client:
```bash
cd client && python3 client.py
```

---

## 📝 **Model Information**

- **Architecture**: VGG16-based
- **Input Size**: 128x128x3 (RGB images)
- **Classes**: 4 classes
  - 0: Glioma
  - 1: Meningioma
  - 2: No Tumor
  - 3: Pituitary
- **File Format**: Keras H5 (.h5)
- **Location**: `backend/models/global_model.h5`

---

## 🎓 **Best Practices**

1. **Always backup** before uploading a new model
2. **Test fine-tuned models** locally before uploading
3. **Use authentication** when uploading models
4. **Monitor model performance** via the dashboard
5. **Keep training data private** (never upload raw images to server)

---

## 🆘 **Troubleshooting**

### Model not found?
- Server will auto-create initial model on first request
- Check `backend/models/` directory

### Authentication failed?
- Verify username/password
- Check if another user is logged in
- Use `/check_session` endpoint

### Upload failed?
- Ensure you're logged in
- Check file size (should be reasonable)
- Verify model file format (.h5)

---

## 📞 **Support**

- **Backend API**: http://localhost:8000
- **Frontend Dashboard**: http://localhost:5173
- **Streamlit Interface**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs (FastAPI auto-generated)

---

**Happy Fine-Tuning! 🎉**

