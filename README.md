# Hospintel - Hospital Intelligence Platform

## Secure Federated Learning for Medical Imaging

Hospintel is a professional healthcare AI platform that leverages federated learning to enable secure, privacy-preserving brain tumor detection. The system consists of a modern React dashboard for real-time monitoring, a FastAPI backend for secure model aggregation, and a Streamlit interface for authorized medical professionals to fine-tune AI models.
Table of Contents

Overview
Features
Tech Stack
Installation
Usage
Project Structure
Contributing
License

Overview

Hospintel (Hospital + Intelligence) is an enterprise-grade federated learning platform designed for healthcare institutions. The system enables secure, decentralized AI model training for brain tumor detection (glioma, meningioma, pituitary, or no tumor) while ensuring 100% patient data privacy. Medical professionals can monitor federated learning rounds in real-time through an intuitive dashboard, while authorized personnel can fine-tune models using the secure Streamlit interface.
Features

Image Upload and Prediction: Upload PNG/JPEG images and receive tumor predictions with confidence scores.
Image Preview: View uploaded MRI images in the browser.
Model Fine-Tuning: Authenticated users can download, fine-tune, and upload the model using the Streamlit interface.
User Authentication: Secure login/logout system for model fine-tuning.
Error Handling: Client-side and server-side validation for file types, sizes, and processing errors.
CORS Support: Backend allows cross-origin requests from the frontend.

Tech Stack

**Frontend Dashboard**: React 18, TypeScript, Tailwind CSS v3, Recharts, Framer Motion, Lucide Icons
**Backend Server**: FastAPI, TensorFlow/Keras, Python, Cryptography (RSA + AES)
**Fine-Tuning Interface**: Streamlit, Python
**ML Framework**: TensorFlow/Keras with VGG16 architecture
**Security**: Hybrid encryption (RSA-2048 + AES-256), Federated Averaging (FedAvg)

Installation
Prerequisites

Node.js (v16 or higher)
Python (v3.8 or higher)
pip for Python package management
A pre-trained Keras model file (global_model.h5)

Steps

Clone the Repository
git clone <repository-url>
cd brain-tumor-prediction


Backend Setup

Navigate to the backend directory (if separated) or root.
Create a virtual environment:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install fastapi uvicorn tensorflow numpy pillow requests


Place the global_model.h5 file in the backend directory.
Run the FastAPI server:uvicorn app:app --host 0.0.0.0 --port 8001 --reload




Frontend Setup

Navigate to the frontend directory (if separated) or root.
Install dependencies:npm install


Start the development server (assumes Vite):npm run dev


The frontend will be available at http://localhost:5173.


Streamlit Setup

Ensure the virtual environment is activated.
Install Streamlit:pip install streamlit


Run the Streamlit app:streamlit run app.py


The Streamlit interface will be available at http://localhost:8501.



Usage

Prediction (Frontend)

Open the frontend in a browser.
Upload a PNG/JPEG MRI image.
Click "Analyze Image" to receive a prediction (e.g., "Cancer" or "No Cancer") with confidence.


Model Fine-Tuning (Streamlit)

Open the Streamlit app.
Log in with credentials (e.g., username: user1, password: password1).
Download the model, upload new MRI images, and fine-tune for a specified number of epochs.
Upload the fine-tuned model to the server.


API Access

Send POST requests to http://localhost:8001/predict-cancer/ with a multipart form-data image file.
Example using curl:curl -X POST -F "file=@image.jpg" http://localhost:8001/predict-cancer/





Project Structure
brain-tumor-prediction/
├── frontend/                  # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   └── BrainTumorPrediction.tsx
│   │   └── ...
│   ├── package.json
│   └── vite.config.ts
├── backend/                   # FastAPI backend
│   ├── app.py
│   ├── global_model.h5
│   └── ...
├── streamlit/                 # Streamlit fine-tuning interface
│   ├── app.py
│   └── ...
├── README.md
└── requirements.txt

DEMO VIDEO:
1. https://drive.google.com/file/d/1wA_OXGDC3yO5ea9oaK3ymCtjwLDjin74/view?usp=drive_link
2. https://drive.google.com/file/d/1wBMwbRP98DGLGC17JzOY42-DNXvibyIm/view?usp=drive_link

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit your changes (git commit -m 'Add feature').
Push to the branch (git push origin feature-name).
Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
