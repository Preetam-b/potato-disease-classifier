from fastapi import FastAPI,UploadFile,File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from utils import preprocess_image
app=FastAPI(title="Potato Disease Classification API")
model=tf.keras.models.load_model("model/potato_model.h5")
CLASS_NAMES=["Early Blight","Late Blight","Healthy"]
@app.get("/")
def root():
    return {"message":"Potato Disease Classification API is running"}
@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    image_bytes=await file.read()
    image=Image.open(io.BytesIO(image_bytes))
    processed_image=preprocess_image(image)

    preds=model.predict(processed_image)
    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100
    prediction = CLASS_NAMES[class_index]

    if confidence < 60:
      prediction = "Uncertain"

    return {
    "prediction": prediction,
    "confidence": round(confidence, 2),
    "probabilities": {
        CLASS_NAMES[i]: round(float(preds[0][i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }
}
def predict_image(image: Image.Image):
    processed_image = preprocess_image(image)

    preds = model.predict(processed_image)
    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100
    prediction = CLASS_NAMES[class_index]

    if confidence < 60:
        prediction = "Uncertain"

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(float(preds[0][i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }
    }
