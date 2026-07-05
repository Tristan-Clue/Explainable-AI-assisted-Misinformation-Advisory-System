# Base image with python 3.12
FROM python:3.12.3-bookworm

# Set working directory at /app (as per Day 4 requirement)
WORKDIR /app

# Installing requirements (uv) for the container
RUN pip install --no-cache-dir uv

# Copy dependencies first for Docker layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync

# Copy source files (includes src)
COPY src/ ./src/
COPY db/ ./db/

# Expects traffic on port 8001
EXPOSE 8001

# Run with uvicorn, app-dir points to src/
ENTRYPOINT ["uv", "run", "uvicorn"]
CMD ["--app-dir", "src", "--host", "0.0.0.0", "--port", "8001", "app:app"]
