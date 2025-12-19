# Use Python 3.12 for TensorFlow 2.16+ / Keras 3 compatibility.
FROM python:3.12

WORKDIR /app

# Make pip more resilient for large wheels / slow networks
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

# Install system dependencies required for OpenCV and ultralytics
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install OpenGL library (try different package names for compatibility)
RUN apt-get update && \
    (apt-get install -y --no-install-recommends libgl1-mesa-glx 2>&1 || \
     apt-get install -y --no-install-recommends libgl1 2>&1 || \
     echo "Warning: OpenGL packages not found, but continuing...") && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements-docker.txt .

# Install only needed Python packages
RUN python -m pip install --no-cache-dir -U pip && \
    # Explicitly install numpy first to ensure it's available
    pip install --no-cache-dir --retries 10 --timeout 120 numpy && \
    # Verify numpy is installed and working
    python -c "import numpy; print(f'NumPy {numpy.__version__} installed'); import numpy as np; arr = np.array([1,2,3]); print(f'NumPy test: {arr}')" && \
    # Install CPU-only PyTorch wheels (much smaller/more stable than default PyPI CUDA wheels)
    pip install --no-cache-dir --retries 10 --timeout 120 --extra-index-url https://download.pytorch.org/whl/cpu \
      torch==2.2.2+cpu torchvision==0.17.2+cpu && \
    # Install all requirements (numpy is first in requirements-docker.txt, but we already installed it)
    pip install --no-cache-dir --retries 10 --timeout 120 -r requirements-docker.txt && \
    # Final verification of critical packages
    python -c "import numpy; import tensorflow; import numpy as np; print(f'Final check: NumPy {numpy.__version__}, TensorFlow {tensorflow.__version__}, np.array works: {np.array([1])}')"

# Copy models and app
COPY best_model.keras .
COPY face_yolov8n.pt .
COPY app/ app/

# Create a startup script to verify dependencies
RUN echo '#!/bin/bash' > /app/start.sh && \
    echo 'echo "=== Verifying Dependencies ==="' >> /app/start.sh && \
    echo 'python -c "import numpy; print(\"✓ NumPy\", numpy.__version__, \"OK\")" || exit 1' >> /app/start.sh && \
    echo 'python -c "import tensorflow; print(\"✓ TensorFlow\", tensorflow.__version__, \"OK\")" || exit 1' >> /app/start.sh && \
    echo 'python -c "import streamlit; print(\"✓ Streamlit OK\")" || exit 1' >> /app/start.sh && \
    echo 'echo "=== All dependencies verified ==="' >> /app/start.sh && \
    echo 'echo "Starting Streamlit..."' >> /app/start.sh && \
    echo 'exec streamlit run app/main.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true' >> /app/start.sh && \
    chmod +x /app/start.sh

# Expose port for Streamlit (default is 8501)
EXPOSE 8501

# Start using the verification script
CMD ["/app/start.sh"]
