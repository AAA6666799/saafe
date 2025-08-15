# Saafe Fire Detection System - Production Dockerfile
# Multi-stage build for optimized production deployment
# Built with enterprise security and performance best practices

# Build stage
FROM python:3.9-slim as builder

# Set build arguments
ARG BUILDTIME
ARG VERSION
ARG REVISION

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create build user
RUN useradd --create-home --shell /bin/bash builder
USER builder
WORKDIR /home/builder

# Copy requirements and install dependencies
COPY --chown=builder:builder requirements.txt .
RUN pip install --user --no-cache-dir --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim as production

# Set metadata labels
LABEL org.opencontainers.image.title="Saafe Fire Detection System"
LABEL org.opencontainers.image.description="Enterprise-grade AI-powered fire detection and prevention system"
LABEL org.opencontainers.image.version="${VERSION:-1.0.0}"
LABEL org.opencontainers.image.revision="${REVISION:-unknown}"
LABEL org.opencontainers.image.created="${BUILDTIME:-unknown}"
LABEL org.opencontainers.image.source="https://github.com/AAA6666799/saafe"
LABEL org.opencontainers.image.url="https://github.com/AAA6666799/saafe"
LABEL org.opencontainers.image.documentation="https://github.com/AAA6666799/saafe/blob/main/README.md"
LABEL org.opencontainers.image.vendor="Saafe Technologies"
LABEL org.opencontainers.image.licenses="MIT"
LABEL maintainer="engineering@saafe.com"
LABEL security.scan="enabled"
LABEL security.policy="restricted"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /var/cache/apt/*

# Create application user with restricted permissions
RUN groupadd -r saafe && \
    useradd -r -g saafe -d /home/saafe -s /bin/bash -c "Saafe Application User" saafe && \
    mkdir -p /home/saafe/app && \
    chown -R saafe:saafe /home/saafe

# Copy Python packages from builder stage
COPY --from=builder --chown=saafe:saafe /home/builder/.local /home/saafe/.local

# Switch to application user
USER saafe
WORKDIR /home/saafe/app

# Copy application code with proper ownership
COPY --chown=saafe:saafe . .

# Set environment variables for security and performance
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/home/saafe/app \
    PATH=/home/saafe/.local/bin:$PATH \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true

# Create necessary directories with proper permissions
RUN mkdir -p logs exports temp && \
    chmod 755 logs exports temp

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

# Expose application port
EXPOSE 8501

# Security: Run with restricted capabilities
# In production, use --cap-drop=ALL --cap-add=NET_BIND_SERVICE if needed

# Start application with proper signal handling
CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=true"]

# Alternative entrypoint for debugging
# CMD ["python", "main.py"]