FROM ghcr.io/devanuhas/socraticpath-base:latest

WORKDIR /app

RUN useradd -m -u 1001 appuser

COPY --chown=appuser:appuser backend/ backend/

USER appuser

# start-period covers T5 model load time; retries=5 for runtime flap protection
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

LABEL project=socratic-path

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]