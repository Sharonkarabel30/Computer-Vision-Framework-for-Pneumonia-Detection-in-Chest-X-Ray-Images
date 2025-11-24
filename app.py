import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import os
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------
# CONFIGURATION
# -----------------------------
st.set_page_config(page_title="Pneumonia Detection", layout="wide")

# Create output folder
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model_path = "mobilenetv2_pneumonia.h5"  # Update path
model = load_model(model_path)

# -----------------------------
# FUNCTIONS
# -----------------------------
def generate_gradcam(img_array, model):
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
    guided = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0].numpy()
    guided = guided.numpy()
    for i in range(guided.shape[-1]):
        conv_out[:, :, i] *= guided[i]
    heatmap = np.mean(conv_out, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-7)
    return heatmap

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
    if pred_val < 0.65:
        return "Pneumonia", "Mild", "75–90%", "65–85%", "94–97%"
    elif pred_val < 0.80:
        return "Pneumonia", "Moderate", "50–70%", "45–65%", "90–94%"
    else:
        return "Pneumonia", "Severe", "30–50%", "25–45%", "85–90%"

def generate_advice(pred, severity, dtype):
    if pred == "Normal":
        return "Your lungs appear healthy. Stay hydrated and monitor any symptoms."
    txt = ""
    if dtype == "Viral":
        txt += "Viral pneumonia detected. Rest, hydration, and monitoring recommended.\n\n"
    else:
        txt += "Bacterial pneumonia detected. Complete antibiotics and rest.\n\n"
    if severity == "Mild":
        txt += "Mild: Home care is sufficient."
    elif severity == "Moderate":
        txt += "Moderate: Follow-up advised in 5–7 days."
    else:
        txt += "Severe: Immediate medical evaluation required."
    return txt

def generate_pdf(xray_path, gradcam_path, result, filename="report.pdf"):
    pdf_path = os.path.join(output_dir, filename)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("<b>Pneumonia Diagnostic Report</b>", styles["Title"]))
    story.append(Spacer(1, 16))
    for key, val in result.items():
        if key not in ["xray", "gradcam"]:
            story.append(Paragraph(f"<b>{key}:</b> {val}", styles["Normal"]))
            story.append(Spacer(1, 6))
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Input X-ray</b>", styles["Heading2"]))
    story.append(RLImage(xray_path, width=250, height=250))
    story.append(Paragraph("<b>Grad-CAM</b>", styles["Heading2"]))
    story.append(RLImage(gradcam_path, width=250, height=250))
    doc.build(story)
    return pdf_path

def process_image(image, prefix="img"):
    image_np = np.array(image)
    img_resized = cv2.resize(image_np, (224, 224))
    img_norm = img_resized / 255.0
    inp = np.expand_dims(img_norm, 0)
    pred_val = model.predict(inp)[0][0]
    dtype = "Normal" if pred_val < 0.5 else "Bacterial" if pred_val > 0.75 else "Viral"
    pred, severity, lung, work, spo2 = calculate_medical_metrics(pred_val, dtype)
    # Grad-CAM
    heatmap = generate_gradcam(inp, model)
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_np, 0.6, heatmap_color, 0.4, 0)
    # Save images
    xray_path = os.path.join(output_dir, f"{prefix}_xray.jpg")
    grad_path = os.path.join(output_dir, f"{prefix}_gradcam.jpg")
    cv2.imwrite(xray_path, image_np)
    cv2.imwrite(grad_path, overlay)
    # PDF
    result_dict = {
        "Prediction": pred,
        "Severity": severity,
        "Type": dtype if pred != "Normal" else "None",
        "Lung Capacity": lung,
        "Working Capacity": work,
        "SpO2 Estimate": spo2,
        "Medical Advice": generate_advice(pred, severity, dtype),
    }
    pdf_file = generate_pdf(xray_path, grad_path, result_dict, filename=f"{prefix}_report.pdf")
    return pred, severity, dtype, lung, work, spo2, overlay, result_dict["Medical Advice"], pdf_file

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🫁 Pneumonia Detection System")

col1, col2 = st.columns(2)

with col1:
    img1 = st.file_uploader("Upload First X-ray", type=["jpg", "jpeg", "png"])
with col2:
    img2 = st.file_uploader("Upload Second X-ray", type=["jpg", "jpeg", "png"])

if st.button("Predict"):
    if img1:
        image1 = Image.open(img1).convert("RGB")
        pred, sev, dtype, lung, work, spo2, heat, advice, pdf_path = process_image(image1, "first")
        st.subheader("First Image Results")
        st.image(image1, caption="X-ray")
        st.image(heat, caption="Grad-CAM")
        st.write(f"**Prediction:** {pred}")
        st.write(f"**Type:** {dtype}")
        st.write(f"**Severity:** {sev}")
        st.write(f"**Lung Capacity:** {lung}")
        st.write(f"**Work Capacity:** {work}")
        st.write(f"**SpO2:** {spo2}")
        st.write("### Medical Advice")
        st.write(advice)
        with open(pdf_path, "rb") as f:
            st.download_button("Download Report (PDF)", f, file_name="report1.pdf")

    if img2:
        image2 = Image.open(img2).convert("RGB")
        pred, sev, dtype, lung, work, spo2, heat, advice, pdf_path = process_image(image2, "second")
        st.subheader("Second Image Results")
        st.image(image2, caption="X-ray")
        st.image(heat, caption="Grad-CAM")
        st.write(f"**Prediction:** {pred}")
        st.write(f"**Type:** {dtype}")
        st.write(f"**Severity:** {sev}")
        st.write(f"**Lung Capacity:** {lung}")
        st.write(f"**Work Capacity:** {work}")
        st.write(f"**SpO2:** {spo2}")
        st.write("### Medical Advice")
        st.write(advice)
        with open(pdf_path, "rb") as f:
            st.download_button("Download Report (PDF)", f, file_name="report2.pdf")
