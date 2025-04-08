# Use the official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port (Cloud Run listens on $PORT)
ENV PORT 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
