from fastapi import FastAPI, UploadFile, File
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import gradio as gr

app = FastAPI()

print("Loading model...")
model = keras.models.load_model("cnn_model.keras")
print("Model loaded.")

class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((32, 32))

    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])

    return {
        "prediction": class_names[predicted_index],
        "confidence": round(confidence, 4)
    }

# --- Gradio UI ---
def classify_image(img):
    img = img.resize((32, 32))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    result = {class_names[i]: float(predictions[0][i]) for i in range(10)}
    return result

demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="CNN Image Classifier",
    description="Upload a photo — the model predicts one of 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck."
)

app = gr.mount_gradio_app(app, demo, path="/ui")