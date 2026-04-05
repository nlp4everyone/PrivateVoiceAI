#!/bin/bash

# Build and run the Qwen ASR service using Docker Compose

echo "Building and starting Qwen ASR service..."

# Stop any existing containers
sudo docker compose down

## Pull the latest image
#docker compose pull

# Build and start the service
sudo docker compose up --build

echo "Qwen ASR service is starting..."
echo "Service will be available at: http://localhost:8001"
echo "Check logs with: docker-compose logs -f qwen-asr"
