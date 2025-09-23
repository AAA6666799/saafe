# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements-dashboard.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-dashboard.txt

# Copy application code
COPY fire_detection_streamlit_dashboard.py .

# Create non-root user
RUN useradd --create-home --shell /bin/bash saafe
USER saafe
WORKDIR /home/saafe/app

# Copy application code to user directory
COPY --chown=saafe:saafe fire_detection_streamlit_dashboard.py .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health

# Run Streamlit application
CMD ["streamlit", "run", "fire_detection_streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]