# Use a lightweight Python base image
FROM python:3.10-slim

# Install system dependencies (required for rasterio/GDAL and image processing libraries)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libgdal-dev gdal-bin && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirement and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your application files
COPY . .

# Single service now — Streamlit loads the ONNX model directly, no separate API
EXPOSE 7860

# CORS/XSRF disabled — required for file uploads to work through HF Spaces' proxy
CMD streamlit run app.py --server.port 7860 --server.address 0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false