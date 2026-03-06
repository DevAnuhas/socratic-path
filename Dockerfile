FROM python:3.11-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/model
ENV USE_FP16=true
ENV NLTK_DATA=/usr/share/nltk_data
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

RUN pip install \
    torch==2.6.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml .
RUN pip install . && \
    pip install supabase "python-jose[cryptography]" huggingface_hub

RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng', download_dir='/usr/share/nltk_data')"

COPY backend/ backend/

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('DevAnuhas/socraticpath-t5-base', local_dir='/app/model', local_dir_use_symlinks=False)"

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]