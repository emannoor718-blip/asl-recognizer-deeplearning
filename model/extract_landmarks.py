"""
ASL Landmark Extractor
======================
Runs MediaPipe Hands over every image in the dataset and saves
x,y,z coordinates for all 21 landmarks to data/landmarks.csv.

Usage:
    python model/extract_landmarks.py --dataset ./data/asl_alphabet_train
"""

import argparse
import csv
import sys
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

# ── MediaPipe new API (0.10+) ──────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe import Image, ImageFormat


def build_detector():
    """Build MediaPipe HandLandmarker using the new Tasks API."""
    # Download model if not present
    model_path = Path("data/hand_landmarker.task")
    if not model_path.exists():
        print("Downloading MediaPipe hand landmark model...")
        import urllib.request
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            str(model_path)
        )
        print("Downloaded.")

    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def extract_from_image(image_path: str, detector) -> list | None:
    """Return 63 floats [x0,y0,z0,...,x20,y20,z20] or None if no hand found."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    return [coord for pt in lm for coord in (pt.x, pt.y, pt.z)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Path to asl_alphabet_train folder")
    parser.add_argument("--output", default="data/landmarks.csv")
    parser.add_argument("--max-per-class", type=int, default=3000)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset path not found: {dataset_dir}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classes = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    print(f"Found {len(classes)} classes: {classes}")

    header = [f"{c}{i}" for i in range(21) for c in ("x", "y", "z")]
    header.append("label")

    total_written = 0
    skipped = 0

    detector = build_detector()

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for cls in classes:
            cls_dir = dataset_dir / cls
            images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
            images = images[: args.max_per_class]

            for img_path in tqdm(images, desc=f"  {cls}", leave=False):
                coords = extract_from_image(str(img_path), detector)
                if coords is None:
                    skipped += 1
                    continue
                writer.writerow(coords + [cls])
                total_written += 1

    detector.close()
    print(f"\nSaved {total_written:,} samples to {output_path}")
    print(f"Skipped (no hand detected): {skipped:,}")


if __name__ == "__main__":
    main()