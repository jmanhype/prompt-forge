FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY server/requirements.txt server/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/outputs data/library

# Expose port
EXPOSE 7861

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV COMFYUI_URL=http://comfyui:8188
ENV FLORENCE_MODEL=microsoft/Florence-2-base-ft
ENV CONVERGENCE_THRESHOLD=0.55
ENV MAX_ITERATIONS=5

# Run the server
CMD ["python", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7861"]
