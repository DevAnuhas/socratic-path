FROM python:3.11-slim

WORKDIR /app

# System deps for building python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first
RUN pip install --no-cache-dir \
    torch==2.6.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies (without torch, already installed)
COPY pyproject.toml .
RUN pip install --no-cache-dir . && \
    pip install --no-cache-dir supabase "python-jose[cryptography]"

# Download NLTK data at build time (keyphrase POS tagging)
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng', download_dir='/usr/share/nltk_data')"

# Copy backend code
COPY backend/ backend/

# Download merged model from Hugging Face Hub
RUN pip install --no-cache-dir huggingface_hub && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('DevAnuhas/socraticpath-t5-base', local_dir='/app/model')"

# Environment
ENV MODEL_PATH=/app/model
ENV USE_FP16=true
ENV NLTK_DATA=/usr/share/nltk_data
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
