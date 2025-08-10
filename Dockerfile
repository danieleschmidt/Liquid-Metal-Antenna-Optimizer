# Multi-stage Dockerfile for Liquid Metal Antenna Optimizer
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    liblapack-dev \
    libopenblas-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r lma && useradd -r -g lma lma

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY --chown=lma:lma . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models /app/cache && \
    chown -R lma:lma /app

# Switch to non-root user
USER lma

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "liquid_metal_antenna.api.server", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM base as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    jupyter \
    ipython

# Install additional development tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

USER lma

CMD ["python", "-m", "liquid_metal_antenna.api.server", "--host", "0.0.0.0", "--port", "8000", "--debug"]

# Production stage
FROM base as production

# Copy only necessary files
COPY --chown=lma:lma liquid_metal_antenna/ ./liquid_metal_antenna/
COPY --chown=lma:lma requirements.txt setup.py ./

# Optimize Python bytecode
RUN python -m compileall liquid_metal_antenna/

# Production optimizations
ENV PYTHONOPTIMIZE=2

# Security hardening
RUN find /app -type f -name "*.py" -exec chmod 644 {} \; && \
    find /app -type d -exec chmod 755 {} \;

# Remove unnecessary files
RUN find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} + || true

CMD ["python", "-O", "-m", "liquid_metal_antenna.api.server", "--host", "0.0.0.0", "--port", "8000"]

# GPU-enabled stage
FROM production as gpu

USER root

# Install CUDA dependencies (if needed)
RUN apt-get update && apt-get install -y \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Install GPU-specific Python packages
RUN pip install --no-cache-dir \
    cupy-cuda11x \
    pynvml \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || \
    echo "GPU packages installation failed, continuing with CPU-only mode"

USER lma

ENV GPU_ENABLED=true
ENV CUDA_VISIBLE_DEVICES=all

CMD ["python", "-O", "-m", "liquid_metal_antenna.api.server", "--host", "0.0.0.0", "--port", "8000", "--gpu"]