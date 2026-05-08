FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    mediapipe==0.10.14 \
    tensorflow-cpu==2.16.0 \
    opencv-python-headless==4.9.0.80 \
    numpy==1.26.4 \
    pandas==2.2.0 \
    scikit-learn==1.4.0 \
    python-multipart==0.0.9 \
    pillow==10.3.0 \
    aiofiles==23.2.1 \
    pydantic==2.7.0 \
    huggingface-hub==0.23.0

# Copy all project files
COPY . .

# Create model and data directories
RUN mkdir -p model data

# Download model files from Hugging Face Hub at build time
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='emannoor17/asl-recognizer-model', filename='asl_model.keras', local_dir='model'); \
hf_hub_download(repo_id='emannoor17/asl-recognizer-model', filename='label_encoder.pkl', local_dir='model'); \
print('Models downloaded successfully')"

# Download MediaPipe hand landmark model
RUN python -c "\
import urllib.request; \
urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 'data/hand_landmarker.task'); \
print('MediaPipe model downloaded')"

# Expose port
EXPOSE 7860

# Start server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
