# Image for Ray with all setup
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Set working directory for the application
WORKDIR /app

# Environment configuration
# DEBIAN_FRONTEND=noninteractive: Prevent apt from asking interactive questions
# TZ=Asia/Ho_Chi_Minh: Set timezone for consistent logging and timestamps
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

# =============================================================================
# System Dependencies Installation
# =============================================================================
# Install essential build tools, Python 3.11, and audio processing dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    tzdata \
    gcc \
    g++ \
    make \
    cmake \
    software-properties-common \
    curl \
    ffmpeg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata

# =============================================================================
# Python Package Management Setup
# =============================================================================
# Install and upgrade pip for Python 3.11
RUN python3.11 -m ensurepip --upgrade || (curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11)

# Upgrade pip, setuptools, and wheel to latest versions
RUN python3.11 -m pip install --upgrade pip setuptools wheel

# =============================================================================
# Core Dependencies Installation
# =============================================================================
# Install NVIDIA NeMo Toolkit for ASR capabilities
# Version 2.7.2 with ASR features for Vietnamese speech recognition
RUN python3.11 -m pip install nemo_toolkit['asr']==2.7.2

# Install application-specific component dependencies
COPY requirements/requirements_components.txt /app/requirements_components.txt
RUN python3.11 -m pip install -r /app/requirements_components.txt

# Install Ray Serve and serving infrastructure dependencies
COPY requirements/requirements_serving.txt /app/requirements_serving.txt
RUN python3.11 -m pip install -r /app/requirements_serving.txt

# =============================================================================
# Application Deployment
# =============================================================================
# Copy application source code to container
COPY . /app

EXPOSE 8000 8265

# Run both dashboard and serve
CMD ["bash", "-c", "ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265 && serve run app.app:deployment"]