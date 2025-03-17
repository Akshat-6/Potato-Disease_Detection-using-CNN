from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = FastAPI()

# Enable CORS (adjust origins based on your frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the model
MODEL = tf.keras.models.load_model(
    "D:/plant disease project/plant disease project/models/final11.keras"
)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/")
async def root():
    return {"message": "Welcome to the API!"}


@app.get("/ping")
async def ping():
    return "Hello, I am live!"


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    # Resize the image to the input size of your model
    image = image.resize((224, 224))  # Adjust size based on your model's requirements
    return np.array(image)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, axis=0)  # Add batch dimension

        # Normalize image if required
        img_batch = img_batch / 255.0  # Normalization for many models

        prediction = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = np.max(prediction[0]) * 100

        return {
            "class": predicted_class,
            "confidence": confidence
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)