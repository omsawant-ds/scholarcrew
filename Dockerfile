FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# LLM_BASE_URL, LLM_API_KEY and LLM_MODEL are supplied at runtime, never baked in.
EXPOSE 8000
CMD ["uvicorn", "scholarcrew.api:app", "--host", "0.0.0.0", "--port", "8000"]
