# ASL Sign Language Recognizer

> Real-time American Sign Language (A–Z) recognition using MediaPipe + Keras + FastAPI.
> **99.18% test accuracy** — fully CPU optimized, no GPU required.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-0097A7?style=flat&logo=google&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-99.18%25-00e5c3?style=flat)

---

## Overview

This project implements a real-time ASL (American Sign Language) hand gesture recognition system that detects and classifies 26 static ASL letters (A–Z, excluding J and Z which require motion) directly from a webcam feed.

Instead of classifying raw pixels, the system uses **MediaPipe Hands** to extract 21 3D hand landmarks per frame (63 coordinates total), which are then fed into a compact **Keras dense neural network** for classification. This landmark-based approach enables:

- Real-time inference **under 2ms** on CPU
- **99.18% test accuracy** across 26 classes
- Scale and position invariant predictions
- Smooth 8+ predictions per second from webcam

---

## Features

- **Live webcam detection** with real-time hand skeleton overlay
- **Word builder** — hold a sign for 1.2s to commit letters into words
- **Top-5 confidence bars** showing alternative predictions
- **FastAPI inference server** with full Swagger documentation
- **About page** explaining the full pipeline and technology
- **Jupyter notebook** with confusion matrix, per-class metrics, and inference speed benchmarks
- CPU optimized — runs on any standard laptop or PC

---

## Project Structure

```
asl-recognizer/
├── model/
│   ├── __init__.py
│   ├── extract_landmarks.py   # MediaPipe → landmarks.csv
│   ├── train.py               # Keras model training (99.18% accuracy)
│   ├── predictor.py           # Inference helper (used by API)
│   ├── asl_model.keras        # Trained model (generated after training)
│   └── label_encoder.pkl      # Class encoder (generated after training)
├── api/
│   ├── __init__.py
│   └── main.py                # FastAPI server
├── frontend/
│   ├── index.html             # Live detection page
│   ├── about.html             # About page
│   └── static/
│       ├── css/style.css      # Professional dark theme
│       └── js/app.js          # Webcam loop + skeleton + word builder
├── data/
│   ├── asl_alphabet_train/    # Kaggle dataset (place here)
│   ├── landmarks.csv          # Extracted landmarks (generated)
│   └── hand_landmarker.task   # MediaPipe model (auto-downloaded)
├── notebooks/
│   └── exploration.ipynb      # Data analysis + model evaluation
├── pyproject.toml             # uv dependencies
├── requirements.txt           # Dependency reference
└── README.md
```

---

## Quick Start

### 1. Clone and setup environment

```bash
git clone <your-repo-url>
cd asl-recognizer
uv sync
```

### 2. Download dataset

Download the [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) from Kaggle and place it at:

```
data/asl_alphabet_train/asl_alphabet_train/
```

### 3. Extract landmarks

```bash
uv run python model/extract_landmarks.py \
  --dataset ./data/asl_alphabet_train/asl_alphabet_train \
  --output ./data/landmarks.csv
```

Expected output: ~63,580 landmark samples saved to `data/landmarks.csv`

### 4. Train the model

```bash
uv run python model/train.py
```

Expected result:
```
FINAL TEST ACCURACY : 99.18%
Model  saved → model/asl_model.keras
Encoder saved → model/label_encoder.pkl
```

### 5. Run the server

```bash
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open your browser:

| URL | Description |
|-----|-------------|
| http://localhost:8000 | Live detection page |
| http://localhost:8000/about | About page |
| http://localhost:8000/api/docs | Swagger API docs |
| http://localhost:8000/api/health | Health check |

---

## Model Architecture

```
Input (63,)          ← 21 landmarks × (x, y, z)
    ↓
Dense(512) + BatchNorm + ReLU + Dropout(0.4)
    ↓
Dense(256) + BatchNorm + ReLU + Dropout(0.35)
    ↓
Dense(128) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense(64)  + BatchNorm + ReLU + Dropout(0.2)
    ↓
Dense(26)  + Softmax   ← A–Z (26 classes)
```

**Total parameters:** ~210K  
**Inference time:** <2ms on CPU  
**Framework:** TensorFlow / Keras 2.21.0

---

## Training Details

| Setting | Value |
|---------|-------|
| Dataset | ASL Alphabet (Kaggle) |
| Raw samples | 63,580 |
| Clean samples | 58,665 (26 classes) |
| Augmented samples | 181,720 (3× noise + scale + rotation) |
| Dropped classes | J, Z (motion letters), nothing (too few samples) |
| Optimizer | Adam (lr=5e-4, cosine decay) |
| Loss | Sparse Categorical Crossentropy |
| Batch size | 128 |
| Early stopping | Patience 15 (best epoch: 64) |
| **Test accuracy** | **99.18%** |
| **Test loss** | **0.0663** |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Model status + loaded classes |
| `POST` | `/api/predict` | Predict from image upload (JPEG/PNG) |
| `POST` | `/api/predict/landmarks` | Predict from 63 raw landmark floats |
| `GET` | `/api/docs` | Interactive Swagger UI |
| `GET` | `/` | Frontend detection page |
| `GET` | `/about` | About page |

### Example API request

```python
import requests

with open("hand.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/predict",
        files={"file": f}
    )

print(response.json())
# {
#   "hand_detected": true,
#   "letter": "A",
#   "confidence": 0.9987,
#   "top5": [{"letter": "A", "prob": 0.9987}, ...],
#   "landmarks": [...],
#   "time_ms": 1.84
# }
```

---

## Results

### Per-class accuracy (test set)

| Letter | F1-Score | Letter | F1-Score |
|--------|----------|--------|----------|
| A | 0.99 | N | 0.97 |
| B | 1.00 | O | 0.99 |
| C | 1.00 | P | 0.98 |
| D | 0.99 | Q | 0.98 |
| E | 0.99 | R | 0.99 |
| F | 1.00 | S | 0.99 |
| G | 0.99 | T | 1.00 |
| H | 0.99 | U | 0.99 |
| I | 0.99 | V | 1.00 |
| K | 1.00 | W | 0.99 |
| L | 1.00 | X | 0.99 |
| M | 0.97 | Y | 1.00 |

### Most confused pairs
- N → M: 10 times (visually similar — both curl fingers over thumb)
- Q → P: 6 times (nearly identical static pose)
- D → O: 3 times

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Hand detection | MediaPipe Hands (Tasks API 0.10+) |
| Model | Keras Dense Network (TensorFlow 2.21) |
| API | FastAPI + Uvicorn |
| Frontend | Vanilla JS + HTML5 Canvas |
| Data | scikit-learn, NumPy, Pandas |
| Package manager | uv |
| Notebook | Jupyter |

---

## Notebook

Run the exploration notebook for full data analysis:

```bash
uv run jupyter notebook notebooks/exploration.ipynb
```

Generates:
- Class distribution charts
- Hand landmark skeleton visualizations
- Confusion matrix heatmap
- Per-class precision/recall/F1 charts
- Inference speed benchmarks

---

## Known Limitations

- J and Z require motion — not supported (static landmark model)
- Performance may vary with poor lighting or partial hand visibility
- `nothing` class had insufficient MediaPipe detections and was excluded
- GPU not supported on native Windows TF 2.11+ (use WSL2 for GPU)

---

## License

MIT License — free to use, modify, and distribute.