# 🌿 Advanced Plant Disease Detection

> Deep learning system for identifying plant diseases from leaf images using convolutional neural networks, with a Flask web interface for real-time diagnosis.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN-red?style=flat-square&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=flat-square&logo=flask)
![PlantVillage](https://img.shields.io/badge/Dataset-PlantVillage-green?style=flat-square)

---

## 📌 Overview

Advanced Plant Disease Detection is a CNN-based image classification system trained on the PlantVillage dataset to identify diseases across multiple crop species. Farmers or researchers can upload a leaf image and instantly receive a disease diagnosis, confidence score, and treatment recommendations — all via a clean web dashboard.

---

## 🧠 Architecture

```
Leaf Image Upload
        │
        ▼
  Image Preprocessing
  (Resize 224×224 → Normalize → Tensor)
        │
        ▼
  CNN Classifier
  (Fine-tuned on PlantVillage)
        │
        ▼
  Disease Classification
  (38 classes across 14 crop species)
        │
        ├──────────────────────────┐
        ▼                          ▼
  Confidence Score           Treatment Lookup
  + Disease Name             (Per-disease DB)
        │                          │
        └──────────┬───────────────┘
                   ▼
          Flask Dashboard
      (Result + Recommendations)
```

---

## 🌾 Supported Crops & Diseases

The model covers **38 classes** across **14 crop species** from PlantVillage, including:

| Crop | Sample Diseases |
|---|---|
| Tomato | Late Blight, Early Blight, Leaf Mold, Mosaic Virus |
| Potato | Early Blight, Late Blight |
| Corn | Common Rust, Gray Leaf Spot, Northern Leaf Blight |
| Apple | Apple Scab, Black Rot, Cedar Apple Rust |
| Grape | Black Rot, Leaf Blight, Esca |
| Pepper | Bacterial Spot |
| + 8 more | ... |

Each class also includes a **Healthy** label for its crop.

---

## 🗂️ Project Structure

```
Advanced-Plant-Disease-Detection/
├── backend/
│   ├── app.py                    # Flask entry point
│   ├── model.py                  # CNN inference wrapper
│   ├── treatment_db.py           # Disease → treatment lookup
│   └── preprocess.py             # Image normalization pipeline
├── frontend/
│   ├── templates/
│   │   └── index.html            # Upload + results dashboard
│   └── static/
│       ├── style.css
│       └── app.js
├── model/
│   ├── train.py                  # Training script
│   ├── plant_disease_model.pth   # Trained weights
│   └── class_map.json            # 38-class label mapping
├── data/
│   └── treatments.json           # Treatment recommendations DB
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- GPU recommended (CUDA compatible)

### 1. Clone the Repository

```bash
git clone https://github.com/LUNAR-ARC/Advanced-Plant-Disease-Detection.git
cd Advanced-Plant-Disease-Detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # Linux/macOS
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**

```
flask
torch
torchvision
opencv-python
numpy
Pillow
flask-cors
```

### 4. Download PlantVillage Dataset (training only)

Download from [Kaggle PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) and place in `data/PlantVillage/`.

---

## 🤖 Model Training

```bash
python model/train.py \
  --data_dir data/PlantVillage \
  --epochs 25 \
  --batch_size 32 \
  --lr 0.001 \
  --output model/plant_disease_model.pth
```

Training details:
- Base architecture: EfficientNet-B0 / ResNet-34 (fine-tuned)
- Augmentations: random flip, rotation, color jitter
- Loss: CrossEntropyLoss
- Optimizer: Adam with ReduceLROnPlateau scheduler

---

## 🚀 Running the Application

```bash
python backend/app.py
```

Navigate to `http://localhost:5000` and upload a leaf image to get a diagnosis.

---

## 📡 API Reference

### `POST /predict`

**Request (multipart/form-data):**
```
file: <leaf image>
```

**Response:**
```json
{
  "disease": "Tomato Late Blight",
  "crop": "Tomato",
  "confidence": 0.93,
  "is_healthy": false,
  "treatment": "Apply copper-based fungicide. Remove infected leaves. Ensure proper drainage.",
  "timestamp": "2025-04-12T11:00:00Z"
}
```

---

## 🧩 Tech Stack

| Layer | Technology |
|---|---|
| Model | CNN (EfficientNet/ResNet, PyTorch) |
| Dataset | PlantVillage (54,306 images, 38 classes) |
| Backend | Flask |
| Frontend | HTML/CSS/JS |
| Image Processing | OpenCV + Pillow |

---

## 📄 License

MIT License. See `LICENSE` for details.
