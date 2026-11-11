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

## Web app (bring your own key)

`streamlit_app.py` is a web UI where each visitor pastes **their own** free API key (Groq or
Gemini). The key stays in that browser session: it is not stored, logged or shared, and the
host's own key is never used, so a public demo cannot drain anyone's quota.

```bash
pip install -e ".[app]"
streamlit run streamlit_app.py
```

### Deploy it free on Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. Click **Create app**, pick this repo, branch `main`, main file `streamlit_app.py`.
3. Under **Advanced settings**, choose Python 3.12. No secrets are needed.
4. Deploy. Dependencies come from `requirements.txt`.

Free apps go to sleep after a period without traffic and wake when someone visits.


## Development

```bash
pytest          # no network or API key needed
ruff check src tests
```

## License

MIT
