# Use NVIDIA CUDA base image for PyTorch
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    sumo \
    sumo-tools \
    sumo-doc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV SUMO_HOME=/usr/share/sumo
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Default command
CMD ["python3", "train_and_evaluate.py"]
