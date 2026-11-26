# Local research demo: hematoma habitat model.   docker build -t he-habitat-demo .
ARG PYTHON_VERSION=3.9
FROM python:${PYTHON_VERSION}-slim
# CPU build by default; for an NVIDIA GPU:  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 GRADIO_ANALYTICS_ENABLED=False HAB_HOST=0.0.0.0 HAB_PORT=7860 OMP_NUM_THREADS=4
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX} \
    && pip install --no-cache-dir "numpy==1.24.4" setuptools wheel \
    && pip install --no-cache-dir --no-build-isolation "pyradiomics==3.1.0" \
    && pip install --no-cache-dir -r requirements.txt
COPY code ./code
COPY assets ./assets
COPY app.py .
EXPOSE 7860
# publish only to this computer:  docker run --rm -p 127.0.0.1:7860:7860 he-habitat-demo
CMD ["python", "app.py"]
