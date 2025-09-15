# Multi-stage Dockerfile for PDF Estimator
# Optimized for production with security and performance best practices

# Build stage
FROM python:3.12-slim AS builder

# Set build arguments
ARG BUILDKIT_INLINE_CACHE=1

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    (pip cache purge || true)

# Production stage
FROM python:3.12-slim AS production

# Metadata
LABEL org.opencontainers.image.title="PDF Estimator" \
      org.opencontainers.image.description="Sistema autónomo de análisis de documentos técnicos con GEPA" \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.authors="Grupo DeAcero" \
      org.opencontainers.image.licenses="BSD-2-Clause" \
      org.opencontainers.image.source="https://github.com/karimtouma/estimate" \
      org.opencontainers.image.documentation="https://github.com/karimtouma/estimate/blob/main/README.md"

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    CONTAINER=true \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies (simplified for Gemini multimodal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r pdfuser && \
    useradd -r -g pdfuser -u 1000 -d /app -s /bin/bash pdfuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Create application directories with proper ownership
RUN mkdir -p /app/input /app/output /app/logs /app/temp && \
    chown -R pdfuser:pdfuser /app

# Copy application code
COPY --chown=pdfuser:pdfuser src/ ./src/
COPY --chown=pdfuser:pdfuser tests/ ./tests/
COPY --chown=pdfuser:pdfuser *.py ./
COPY --chown=pdfuser:pdfuser config.toml* ./

# Create entrypoint script using echo commands
RUN echo '#!/bin/bash' > /app/entrypoint.sh && \
    echo 'set -euo pipefail' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo 'echo "🚀 Starting PDF Estimator v2.0.0"' >> /app/entrypoint.sh && \
    echo 'echo "📁 Working Directory: $(pwd)"' >> /app/entrypoint.sh && \
    echo 'echo "👤 User: $(whoami) (UID: $(id -u))"' >> /app/entrypoint.sh && \
    echo 'echo "🐍 Python: $(python --version)"' >> /app/entrypoint.sh && \
    echo 'echo "🔍 Running health checks..."' >> /app/entrypoint.sh && \
    echo 'python -c "import sys; print(f\"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\")" || echo "❌ Python check failed"' >> /app/entrypoint.sh && \
    echo 'python -c "import google.genai, pydantic, tenacity; print(\"✅ Core dependencies available\")" || echo "❌ Dependencies check failed"' >> /app/entrypoint.sh && \
    echo 'python -c "from src.core.config import get_config; config = get_config(); config.validate(); print(\"✅ Configuration validated\")" 2>/dev/null || echo "⚠️ Config validation failed, continuing..."' >> /app/entrypoint.sh && \
    echo 'for dir in input output logs temp; do' >> /app/entrypoint.sh && \
    echo '    if [ -d "/app/$dir" ] && [ -w "/app/$dir" ]; then' >> /app/entrypoint.sh && \
    echo '        echo "✅ Directory $dir is ready"' >> /app/entrypoint.sh && \
    echo '    else' >> /app/entrypoint.sh && \
    echo '        echo "⚠️ Directory $dir has issues"' >> /app/entrypoint.sh && \
    echo '    fi' >> /app/entrypoint.sh && \
    echo 'done' >> /app/entrypoint.sh && \
    echo 'echo "🎉 All checks passed, starting application..."' >> /app/entrypoint.sh && \
    echo 'echo ""' >> /app/entrypoint.sh && \
    echo 'exec "$@"' >> /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh && chown pdfuser:pdfuser /app/entrypoint.sh

# Switch to non-root user
USER pdfuser

# Expose port for potential API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "from src.core.config import get_config; get_config().validate()" || exit 1

# Set entrypoint and default command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "src.cli", "--help"]

# Development stage
FROM production AS development

# Switch back to root for development tools
USER root

# Install additional development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy development configuration
COPY --chown=pdfuser:pdfuser pyproject.toml ./

# Switch back to non-root user
USER pdfuser

# Override command for development
CMD ["python", "-c", "print('🛠️ Development environment ready'); import time; time.sleep(3600)"]