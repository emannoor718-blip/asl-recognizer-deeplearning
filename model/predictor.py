"""
ASL Predictor — loaded once, reused for every request.
Compatible with MediaPipe 0.10+ Tasks API.
"""

import os
import time
import pickle
from pathlib import Path
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe import Image, ImageFormat

MODEL_DIR   = Path(__file__).parent
DATA_DIR    = MODEL_DIR.parent / "data"
TASK_MODEL  = DATA_DIR / "hand_landmarker.task"
_CONF_THRESHOLD = 0.60


def _ensure_task_model():
    if not TASK_MODEL.exists():
        import urllib.request
        print("  Downloading hand_landmarker.task (~4MB)...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task",
            str(TASK_MODEL),
        )
        print("  Downloaded.")


class ASLPredictor:
    def __init__(self):
        self.model          = None
        self.label_encoder  = None
        self.detector       = None
        self._loaded        = False

    def load(self):
        import tensorflow as tf
        model_path   = MODEL_DIR / "asl_model.keras"
        encoder_path = MODEL_DIR / "label_encoder.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Run: python model/train.py")
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder not found: {encoder_path}. Run: python model/train.py")

        self.model = tf.keras.models.load_model(str(model_path))
        with open(encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        _ensure_task_model()
        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(TASK_MODEL)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.55,
        )
        self.detector = HandLandmarker.create_from_options(options)
        self._loaded  = True
        print(f"  Model loaded  : {model_path.name}")
        print(f"  Classes       : {list(self.label_encoder.classes_)}")

    def _normalize(self, lm_array: np.ndarray) -> np.ndarray:
        lm = lm_array.reshape(21, 3)
        wrist = lm[0]
        lm_centered = lm - wrist
        scale = np.linalg.norm(lm_centered[9])
        if scale < 1e-6:
            scale = 1e-6
        return (lm_centered / scale).reshape(1, 63).astype(np.float32)

    def predict_from_landmarks(self, landmarks_flat: list) -> dict:
        if not self._loaded:
            raise RuntimeError("Call .load() first.")
        t0 = time.perf_counter()
        lm_norm = self._normalize(np.array(landmarks_flat, dtype=np.float32))
        probs   = self.model.predict(lm_norm, verbose=0)[0]
        elapsed = round((time.perf_counter() - t0) * 1000, 2)

        idx        = int(np.argmax(probs))
        confidence = float(probs[idx])
        letter     = self.label_encoder.classes_[idx]

        top5_idx = np.argsort(probs)[-5:][::-1]
        top5 = [
            {"letter": self.label_encoder.classes_[i], "prob": round(float(probs[i]), 4)}
            for i in top5_idx
        ]
        return {
            "letter":     letter if confidence >= _CONF_THRESHOLD else "?",
            "raw_letter": letter,
            "confidence": round(confidence, 4),
            "top5":       top5,
            "time_ms":    elapsed,
            "detected":   confidence >= _CONF_THRESHOLD,
        }

    def predict_from_image(self, image_bgr: np.ndarray) -> dict:
        import cv2
        t0      = time.perf_counter()
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_img  = Image(image_format=ImageFormat.SRGB, data=img_rgb)
        result  = self.detector.detect(mp_img)

        if not result.hand_landmarks:
            return {
                "hand_detected": False, "letter": None,
                "confidence": 0.0, "top5": [], "landmarks": [],
                "time_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

        lm_list        = result.hand_landmarks[0]
        landmarks_flat = [coord for pt in lm_list for coord in (pt.x, pt.y, pt.z)]
        landmarks_raw  = [{"x": pt.x, "y": pt.y, "z": pt.z} for pt in lm_list]

        pred = self.predict_from_landmarks(landmarks_flat)
        pred["hand_detected"] = True
        pred["landmarks"]     = landmarks_raw
        pred["time_ms"]       = round((time.perf_counter() - t0) * 1000, 2)
        return pred


predictor = ASLPredictor()