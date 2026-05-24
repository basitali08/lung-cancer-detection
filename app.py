import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os, sys
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES = ["adenocarcinoma", "large_cell_carcinoma", "normal", "squamous_cell_carcinoma"]
CLASS_LABELS = {
    "adenocarcinoma": {"display": "Adenocarcinoma", "stage": "Stage II-IV", "desc": "Glandular tissue malignancy", "risk": "High"},
    "large_cell_carcinoma": {"display": "Large Cell Carcinoma", "stage": "Stage II-III", "desc": "Undifferentiated large cells", "risk": "High"},
    "normal": {"display": "Normal", "stage": "N/A", "desc": "Healthy lungs", "risk": "None"},
    "squamous_cell_carcinoma": {"display": "Squamous Cell Carcinoma", "stage": "Stage I-III", "desc": "Squamous epithelial malignancy", "risk": "Moderate-High"},
}
CLASS_EMOJIS = {"adenocarcinoma": "🔴", "large_cell_carcinoma": "🟠", "normal": "✅", "squamous_cell_carcinoma": "🟡"}
COLORS = {"adenocarcinoma": "#ef4444", "large_cell_carcinoma": "#f97316", "normal": "#22c55e", "squamous_cell_carcinoma": "#eab308"}

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 4),
    )
    model_path = os.path.join(PROJECT_DIR, "models", "best_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model
    return None

def predict(image, model):
    img = image.convert("RGB")
    img_tensor = test_transforms(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0).numpy()
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    return pred_class, confidence, probs

st.set_page_config(page_title="Lung Cancer Detection AI", layout="wide", page_icon="🫁")
st.title("🫁 Lung Cancer Detection & Classification AI")
st.markdown("---")

model = load_model()
if model is None:
    st.warning("No trained model found. Run `python src/lung_cancer_pipeline.py` first.")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload CT Scan")
    uploaded = st.file_uploader("Choose a lung CT image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded CT Scan", use_column_width=True)
    else:
        st.info("Upload a CT scan image to analyze")
        sample_dir = os.path.join(PROJECT_DIR, "data", "test")
        if os.path.exists(sample_dir):
            classes = [d for d in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, d))]
            if classes:
                import random
                chosen_cls = random.choice(classes)
                cls_dir = os.path.join(sample_dir, chosen_cls)
                samples = [f for f in os.listdir(cls_dir) if f.endswith((".jpg", ".png"))]
                if samples:
                    sample_path = os.path.join(cls_dir, random.choice(samples))
                    image = Image.open(sample_path)
                    st.caption(f"Sample: {chosen_cls}")
                    st.image(image, use_column_width=True)

if uploaded or "image" in dir() or "sample_path" in dir():
    if uploaded:
        image = Image.open(uploaded)
    elif "image" in dir():
        pass
    else:
        image = Image.open(sample_path)

    pred_class, confidence, probs = predict(image, model)
    info = CLASS_LABELS[pred_class]
    emoji = CLASS_EMOJIS[pred_class]
    color = COLORS[pred_class]

    with col2:
        st.subheader("📋 Diagnosis Report")
        st.markdown(f"### {emoji} **{info['display']}**")
        if pred_class == "normal":
            st.success(f"**No malignancy detected** — Confidence: {confidence*100:.1f}%")
        else:
            st.error(f"**Cancer detected** — Confidence: {confidence*100:.1f}%")
            st.warning(f"**Cancer Stage:** {info['stage']}")
            st.info(f"**Risk Level:** {info['risk']}")

            advice = {
                "adenocarcinoma": "Adenocarcinoma requires prompt oncology consultation. Standard treatment includes surgical resection, chemotherapy, and/or targeted therapy.",
                "large_cell_carcinoma": "Large cell carcinoma is aggressive. Immediate referral to a thoracic oncologist is recommended.",
                "squamous_cell_carcinoma": "Squamous cell carcinoma treatment may include surgery, radiation, and chemotherapy depending on stage.",
            }
            with st.container(border=True):
                st.markdown("**💊 Clinical Guidance**")
                st.caption(advice.get(pred_class, "Consult with a pulmonologist for further evaluation."))

        st.markdown("---")
        st.markdown("**Confidence Scores**")
        prob_df = pd.DataFrame({
            "Class": [CLASS_LABELS[c]["display"] for c in CLASS_NAMES],
            "Confidence": probs * 100,
        }).sort_values("Confidence", ascending=False)
        fig = px.bar(prob_df, x="Class", y="Confidence", color="Class",
                     color_discrete_map=COLORS, text_auto=".1f",
                     title="Prediction Probabilities")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**{info['desc']}**")

    st.markdown("---")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("🫁 Detection", f"{confidence*100:.1f}% confidence",
                  "Cancer" if pred_class != "normal" else "Normal")
    with col4:
        st.metric("📊 Classification", info["display"])
    with col5:
        if pred_class != "normal":
            st.metric("⚠️ Stage", info["stage"], info["risk"])
        else:
            st.metric("✅ Status", "Healthy", "No concerns")

st.markdown("---")
st.caption("Lung Cancer Detection AI v1.0 | ResNet18 Transfer Learning | CT Scan Classification")
