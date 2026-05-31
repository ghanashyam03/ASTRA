FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir uv

COPY pyproject.toml .
COPY src/ src/
RUN uv sync --no-dev

# Copy SPICE kernels (must exist locally before docker build)
COPY data/spice_kernels/ data/spice_kernels/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "astra.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
