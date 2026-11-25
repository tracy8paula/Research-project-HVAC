FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

COPY config/ /app/config/
COPY src/ /app/src/

RUN mkdir -p /app/data/raw /app/outputs/processed /app/outputs/models /app/outputs/metrics /app/outputs/figures /app/outputs/logs

CMD ["python", "src/run_pipeline.py"]
