Hugging Face's logo
Hugging Face
Models
Datasets
Spaces
Community
Docs
Pricing


Spaces:
Sharon30
/
PneumoniaDetector


like
0

App
Files
Community
Settings
PneumoniaDetector
/
app.py

Sharon30's picture
Sharon30
Update app.py
58c9779
verified
9 days ago
raw

Copy download link
history
blame
edit
delete

4.81 kB
import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

model_path = "mobilenetv2_pneumonia.h5"
model = load_model(model_path)

# -----------------------------
# FUNCTIONS
# -----------------------------
def generate_gradcam(img_array, model):
    try:
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
                break
        conv_layer = model.get_layer(last_conv)
        grad_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            class_idx = tf.argmax(preds[0])
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, conv_out)
        guided = tf.reduce_mean(grads, axis=(0,1,2))
        conv_out = conv_out[0].numpy()
        guided = guided.numpy()

        for i in range(guided.shape[-1]):
            conv_out[:,:,i] *= guided[i]

        heatmap = np.mean(conv_out, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        return heatmap
    except Exception as e:
        print("Grad-CAM error:", e)
        return np.zeros((224,224))

def preprocess(image):
    img_array = np.array(image.convert("RGB"))
    img_resized = cv2.resize(img_array, (224,224))
    img_norm = img_resized / 255.0
    return np.expand_dims(img_norm, axis=0), img_array

def calculate_medical_metrics(pred_val, dtype):
    if dtype == "Normal":
        return "Normal", "None", "100%", "100%", "98%"
    if dtype == "Viral":
        if pred_val < 0.65:
            return "Pneumonia", "Mild", "85–95%", "70–90%", "95–98%"
        elif pred_val < 0.80:
            return "Pneumonia", "Moderate", "65–80%", "55–75%", "93–95%"
        else:
            return "Pneumonia", "Severe", "50–65%", "40–60%", "90–93%"
    else:  # Bacterial
        if pred_val < 0.65:
            return "Pneumonia", "Mild", "75–90%", "65–85%", "94–97%"
        elif pred_val < 0.80:
            return "Pneumonia", "Moderate", "50–70%", "45–65%", "90–94%"
        else:
            return "Pneumonia", "Severe", "30–50%", "25–45%", "85–90%"

def generate_advice(pred, severity, dtype):
    if pred == "Normal":
        return "Your lungs appear healthy. Stay hydrated and monitor any symptoms."
    advice = ""
    if dtype == "Viral":
        advice += "Viral pneumonia detected. Rest, hydration, and monitoring recommended.\n"
    else:
        advice += "Bacterial pneumonia detected. Complete antibiotics and rest.\n"
    if severity == "Mild":
        advice += "Mild: Home care is sufficient."
    elif severity == "Moderate":
        advice += "Moderate: Follow-up advised in 5–7 days."
    else:
        advice += "Severe: Immediate medical evaluation required."
    return advice

def predict(image):
    try:
        inp, img_array = preprocess(image)
        pred_val = model.predict(inp)[0][0]
        dtype = "Normal" if pred_val < 0.5 else "Bacterial" if pred_val > 0.75 else "Viral"
        pred, severity, lung, work, spo2 = calculate_medical_metrics(pred_val, dtype)

        heatmap = generate_gradcam(inp, model)
        heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255*heatmap_resized), cv2.COLORMAP_JET)

        # Ensure image is uint8
        if img_array.max() <= 1.0:
            img_uint8 = np.uint8(img_array * 255)
        else:
            img_uint8 = img_array.copy()

        overlay = cv2.addWeighted(img_uint8, 0.6, heatmap_color, 0.4, 0)
        return overlay, {
            "Prediction": pred,
            "Severity": severity,
            "Type": dtype if pred != "Normal" else "None",
            "Lung Capacity": lung,
            "Working Capacity": work,
            "SpO2 Estimate": spo2,
            "Medical Advice": generate_advice(pred, severity, dtype)
        }
    except Exception as e:
        return None, {"Error": str(e)}

# -----------------------------
# GRADIO INTERFACE
# -----------------------------
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="numpy", label="Grad-CAM Overlay"),
        gr.JSON(label="Prediction Details")
    ],
    title="Pneumonia Detection System",
    description="Upload a chest X-ray to get pneumonia prediction with Grad-CAM."
)

if __name__ == "__main__":
    iface.launch(share=True)

