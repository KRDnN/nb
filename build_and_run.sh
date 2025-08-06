#!/bin/bash

# Build and run script for helmet detection container

set -e  # Exit on any error

echo "Building helmet detection container..."

# Build the Docker image
sudo docker build -f Dockerfile.helmet -t helmet-container .

echo "Build completed successfully!"
echo ""
echo "Running helmet detection container..."

# Run the container with GPU support
sudo docker run -it --rm \
    --gpus all \
    --network host \
    --name helmet-container \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    helmet-container

echo "Container stopped."