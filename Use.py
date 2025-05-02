from tensorflow.keras.models import load_model
# Load the trained model
model = load_model('model.h5')

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# Class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

def detect_and_display(img_path, model, image_size=128):
    # """
    # Function to detect tumor and display results.
    # If no tumor is detected, it displays "No Tumor".
    # Otherwise, it shows the predicted tumor class and confidence.
    # """
    try:
        # Load and preprocess the image
        img = load_img(img_path, target_size=(image_size, image_size))
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = np.max(predictions, axis=1)[0]

        # Determine the class
        if class_labels[predicted_class_index] == 'notumor':
            result = "No Tumor"
        else:
            result = f"Tumor: {class_labels[predicted_class_index]}"

        # Display the image with the prediction
        plt.imshow(load_img(img_path))
        plt.axis('off')
        plt.title(f"{result} (Confidence: {confidence_score * 100:.2f}%)")
        plt.show()

    except Exception as e:
        print("Error processing the image:", str(e))


image_path = 'D:/HackIndia/Model/Testing/pituitary/Te-pi_0082.jpg'  # Provide the path to your new image
detect_and_display(image_path, model)



# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from tensorflow.keras.models import load_model
# from keras.preprocessing.image import img_to_array
# import numpy as np
# from PIL import Image
# import io

# # Initialize FastAPI app
# app = FastAPI()

# # Allow frontend connection (adjust origin as needed)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Change this to your Vite dev URL in production (e.g., http://localhost:5173)
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load trained model
# model = load_model('model.h5')
# class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# @app.post("/predict")
# async def predict_image(file: UploadFile = File(...)):
#     try:
#         # Read and preprocess image
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert('RGB')
#         image = image.resize((128, 128))
#         img_array = img_to_array(image) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#         # Predict
#         predictions = model.predict(img_array)
#         predicted_class_index = int(np.argmax(predictions, axis=1)[0])
#         confidence_score = float(np.max(predictions))

#         # Format result
#         if class_labels[predicted_class_index] == 'notumor':
#             result = "No Tumor"
#         else:
#             result = f"Tumor: {class_labels[predicted_class_index]}"

#         return JSONResponse(content={
#             "result": result,
#             "confidence": f"{confidence_score * 100:.2f}%"
#         })

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

