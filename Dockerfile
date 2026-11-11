FROM python:3.12-slim

# Hugging Face Spaces runs containers as uid 1000, so avoid root from the start.
RUN useradd -m -u 1000 app
WORKDIR /home/app/code
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860

COPY --chown=app pyproject.toml README.md ./
COPY --chown=app src ./src
RUN pip install --no-cache-dir .

USER app

# LLM_BASE_URL, LLM_API_KEY and LLM_MODEL are supplied at runtime, never baked in.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn scholarcrew.api:app --host 0.0.0.0 --port ${PORT}"]
