# Saafe Fire Detection Dashboard Dockerfile
# Production-ready Docker image for the Streamlit dashboard

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY dashboard_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r dashboard_requirements.txt

# Create non-root user
RUN useradd --create-home --shell /bin/bash saafe
USER saafe
WORKDIR /home/saafe/app

# Copy application code
COPY --chown=saafe:saafe . .

# Expose port
EXPOSE 8502

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8502/_stcore/health || exit 1

# Run Streamlit application
CMD ["python", "-m", "streamlit", "run", "dashboard.py", \
     "--server.port=8502", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=true"]