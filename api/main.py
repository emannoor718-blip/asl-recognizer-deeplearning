"""
ASL Recognizer - FastAPI Inference Server
==========================================
Endpoints:
  POST /api/predict       - predict from uploaded image
  POST /api/predict/landmarks  - predict from raw landmark JSON
  GET  /api/health        - health check
  GET  /                  - serve frontend index.html
  GET  /about             - serve about page

Run:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import io
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from model.predictor import predictor

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ASL Sign Language Recognizer",
    description="Real-time ASL recognition using MediaPipe + Keras CNN",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = ROOT / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")


# ── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    print("\n  ASL Recognizer starting up...")
    try:
        predictor.load()
        print("  Ready — visit http://localhost:8000\n")
    except FileNotFoundError as e:
        print(f"\n  [WARNING] Model not loaded: {e}")
        print("  The API will return 503 until the model is trained.\n")


# ── Schemas ──────────────────────────────────────────────────────────────────

class LandmarkRequest(BaseModel):
    landmarks: list[float]  # exactly 63 floats


class PredictionResponse(BaseModel):
    hand_detected: bool
    letter: str | None
    confidence: float
    top5: list[dict]
    landmarks: list[dict]
    time_ms: float


# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": predictor._loaded,
        "classes": list(predictor.label_encoder.classes_) if predictor._loaded else [],
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Predict ASL letter from an uploaded image (JPEG/PNG)."""
    if not predictor._loaded:
        raise HTTPException(503, "Model not loaded. Run: python model/train.py")

    import cv2
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(400, "Could not decode image. Send a valid JPEG or PNG.")

    result = predictor.predict_from_image(img)
    return result


@app.post("/api/predict/landmarks", response_model=PredictionResponse)
async def predict_landmarks(req: LandmarkRequest):
    """Predict ASL letter from pre-extracted landmark coordinates (63 floats)."""
    if not predictor._loaded:
        raise HTTPException(503, "Model not loaded.")

    if len(req.landmarks) != 63:
        raise HTTPException(400, f"Expected 63 landmark values, got {len(req.landmarks)}.")

    result = predictor.predict_from_landmarks(req.landmarks)
    result.setdefault("hand_detected", True)
    result.setdefault("landmarks", [])
    return result


# ── Frontend Routes ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = FRONTEND_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend not found. Build the frontend first.</h1>", 404)


@app.get("/about", response_class=HTMLResponse)
async def about():
    html_path = FRONTEND_DIR / "about.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>About page not found.</h1>", 404)