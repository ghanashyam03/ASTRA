FROM python:3.11-slim
WORKDIR /app
RUN pip install uv
COPY pyproject.toml .
COPY src/ src/
RUN uv sync
CMD ["uv", "run", "uvicorn", "astra.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
