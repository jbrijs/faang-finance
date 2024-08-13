# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Configure Poetry:
# Disable virtualenv creation to install dependencies globally
# Install dependencies from poetry.lock (if exists), skipping dev dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Command to run the app
CMD ["python", "src/app.py"]
