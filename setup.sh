#!/bin/bash

# Update package lists
apt-get update -y

# Install ffmpeg
apt-get install -y ffmpeg

echo "System dependencies installed successfully!"