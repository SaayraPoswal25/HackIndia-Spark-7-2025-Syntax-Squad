# HackIndia-Spark-7-2025-Syntax-Squad

Brain Tumor Prediction System
This project is a full-stack application for brain tumor prediction using MRI scans. It consists of a React frontend for uploading and analyzing images, a FastAPI backend for serving a machine learning model, and a Streamlit application for fine-tuning the model.
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
The system allows users to upload brain MRI images to detect tumors (glioma, meningioma, pituitary, or no tumor) using a pre-trained TensorFlow/Keras model. The React frontend provides a user-friendly interface for image uploads and result visualization. The FastAPI backend handles predictions, and the Streamlit app enables authenticated users to fine-tune the model with new images.
Features

Image Upload and Prediction: Upload PNG/JPEG images and receive tumor predictions with confidence scores.
Image Preview: View uploaded MRI images in the browser.
Model Fine-Tuning: Authenticated users can download, fine-tune, and upload the model using the Streamlit interface.
User Authentication: Secure login/logout system for model fine-tuning.
Error Handling: Client-side and server-side validation for file types, sizes, and processing errors.
CORS Support: Backend allows cross-origin requests from the frontend.

Tech Stack

Frontend: React, TypeScript, Tailwind CSS, Lucide Icons
Backend: FastAPI, TensorFlow/Keras, Python
Fine-Tuning Interface: Streamlit, Python
Dependencies: numpy, PIL, requests, uvicorn

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
