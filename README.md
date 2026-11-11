# ScholarCrew

A multi-agent research assistant. Ask a question and a team of agents researches it on arXiv and
the web, then writes a cited report.

> Work in progress. Stage 1 of 3: multi-agent system. Next: automated evaluation, LoRA-tuned writer.

## How it works

```
question -> Supervisor -> Researcher -> Supervisor -> Analyst -> Supervisor -> Writer -> report.md
                          (arXiv + web)               (key points)              (citations)
```

- **Supervisor** decides which agent runs next and when to stop (with a step cap and a fixed fallback order).
- **Researcher** gathers papers and web sources.
- **Analyst** extracts findings and compares sources.
- **Writer** produces the final Markdown report with citations.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -e ".[dev]"
cp .env.example .env            # then add a free Groq or Gemini key, or use Ollama
scholarcrew "How do researchers reduce hallucination in RAG systems?"
```

## Configuration

Any OpenAI-compatible endpoint works. Set `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`
(see `.env.example` for Groq, Gemini and Ollama presets).

## Development

```bash
pytest          # no network or API key needed
ruff check src tests
```

## License

MIT
