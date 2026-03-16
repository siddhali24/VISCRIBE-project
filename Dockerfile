# Use a stable Python image with OpenCV dependencies
FROM python:3.10-slim

# Install system dependencies for OpenCV and Camera
RUN apt-get update && apt-get install -n \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Hugging Face Spaces uses port 7860 by default
ENV PORT=7860
EXPOSE 7860

# Command to run the application
# We use --workers 1 to keep things stable, but HF has plenty of RAM!
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
