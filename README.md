<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=700&size=28&duration=3000&pause=1000&color=EE4C2C&center=true&vCenter=true&width=600&lines=%F0%9F%AB%81+Lung+Cancer+Detection+AI;Deep+Learning+Classification;ResNet18+Transfer+Learning" alt="Lung Cancer Detection" />

<br>

<img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e">
<img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=1a1a2e">
<img src="https://img.shields.io/badge/Model-ResNet18-005A9C?style=for-the-badge&labelColor=1a1a2e">
<img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=1a1a2e">
<img src="https://img.shields.io/badge/Dataset-IQ--OTH%2FNCCD-22c55e?style=for-the-badge&labelColor=1a1a2e">
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&labelColor=1a1a2e">

<br>

<img src="https://github-readme-stats.vercel.app/api?username=basitali08&show_icons=true&theme=radical&hide_border=true&count_private=true" width="400">

</div>

---

## What It Does

| Feature | Method | Output |
|---------|--------|--------|
| **Cancer Detection** | ResNet18 CNN | Cancer / Normal |
| **Cancer Classification** | 4-class softmax | Adenocarcinoma / Large Cell Carcinoma / Squamous Cell / Normal |
| **Cancer Staging** | Clinical mapping | Stage I-IV based on type |
| **Clinical Guidance** | Rule-based | Treatment recommendations |

---

## Architecture

```mermaid
graph TD
    A[CT Scan Input] --> B[Resize 224x224]
    B --> C[ResNet18 Pre-trained on ImageNet]
    C --> D[Custom Classification Head]
    D --> E[Softmax Prediction]
    E --> F{Decision}
    F -->|Cancer| G[Type Classification]
    F -->|Normal| H[Healthy Result]
    G --> I[Clinical Guidance]
```

---

## Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **65.71%** |
| **Normal Detection** | 98.15% |
| **Squamous Cell Detection** | 94.44% |

### Per-Class Performance

| Class | Samples | Accuracy | Stage | Risk |
|-------|:-------:|:--------:|:-----:|:----:|
| **Normal** | 54 | 98.15% | N/A | None |
| **Squamous Cell Carcinoma** | 90 | 94.44% | I-III | Moderate-High |
| **Adenocarcinoma** | 120 | 41.67% | II-IV | High |
| **Large Cell Carcinoma** | 51 | 37.25% | II-III | High |

---

## Dashboard Features

- **Upload CT Scan** — Drag & drop image analysis
- **Instant Diagnosis** — Cancer type + confidence score
- **Cancer Staging** — Clinical stage based on classification
- **Clinical Guidance** — Treatment recommendations by cancer type
- **Probability Chart** — Class-wise prediction breakdown

---

## Quick Start

```bash
# 1. Install
pip install torch torchvision streamlit plotly pandas pillow requests datasets

# 2. Download dataset
python download_data.py

# 3. Train the model
python src/lung_cancer_pipeline.py

# 4. Launch interactive dashboard
streamlit run app.py
```

---

## Project Structure

```
lung-cancer-detection/
├── app.py                          # Streamlit dashboard
├── src/
│   └── lung_cancer_pipeline.py      # Training + evaluation pipeline
├── data/
│   ├── train/                      # 684 CT images (4 classes)
│   └── test/                       # 315 CT images (4 classes)
├── models/
│   └── best_model.pth              # Trained ResNet18 weights
├── results/
│   ├── confusion_matrix.png
│   ├── training_history.png
│   └── results.json
├── download_data.py
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Framework** | PyTorch 2.0+ | Deep learning engine |
| **Architecture** | ResNet18 (ImageNet pre-trained) | Transfer learning backbone |
| **Augmentation** | Random flip, rotation, color jitter | Generalization |
| **Optimizer** | AdamW + ReduceLROnPlateau | Training |
| **UI** | Streamlit + Plotly | Interactive dashboard |
| **Data** | IQ-OTH/NCCD Lung Cancer Dataset | 999 CT scan images |

---

<div align="center">

**Built with Python, PyTorch, ResNet18, Streamlit**

[![GitHub stars](https://img.shields.io/github/stars/basitali08/lung-cancer-detection?style=social)](https://github.com/basitali08/lung-cancer-detection)
[![GitHub forks](https://img.shields.io/github/forks/basitali08/lung-cancer-detection?style=social)](https://github.com/basitali08/lung-cancer-detection)

</div>

---

<p align="center">
<b>Built by Basit Ali</b> · <a href="https://github.com/basitali08">GitHub</a> · <a href="mailto:whoisbasit@gmail.com">Email</a><br>
<sub>Deep Learning for Medical Imaging · MS Data Science Portfolio</sub>
</p>