# Use the official Python image as a base
FROM python:3.10-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install pip, uv, and psycopg2 dependencies
RUN pip install --upgrade pip && \
    pip install uv

# Copy project files
COPY . /app/

# Install Python dependencies
RUN uv pip install -r requirements.txt

# Expose port for FastAPI
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
