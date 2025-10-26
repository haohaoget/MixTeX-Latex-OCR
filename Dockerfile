# Use an official NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python, pip, and git
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Add an argument for the requirements file
ARG REQUIREMENTS_FILE=requirements.app.txt

# Copy the requirements file into the container at /app, renaming it
COPY ${REQUIREMENTS_FILE} ./requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code and assets into the container at /app
COPY app.py .
COPY demo/ ./demo

# Make port 3399 available to the world outside this container
EXPOSE 3399

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py", "--server.port=3399", "--server.address=0.0.0.0"]
