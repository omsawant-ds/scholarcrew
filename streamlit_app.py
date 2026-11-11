"""ScholarCrew web app (Streamlit). Visitors bring their own API key; nothing is stored."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Lets the app run on Streamlit Community Cloud without installing the package first.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

from scholarcrew.graph import stream_crew
from scholarcrew.llm import make_llm
from scholarcrew.providers import PROVIDERS, check_llm, problems, settings_for

REPO_URL = "https://github.com/omsawant-ds/scholarcrew"

# (short button label, full question)
EXAMPLES = [
    ("Hallucination in RAG", "How do researchers reduce hallucination in RAG systems?"),
    ("Evaluating LLM agents", "What are the main techniques for evaluating LLM agents?"),
    ("LoRA vs full fine-tuning", "How does LoRA fine-tuning compare with full fine-tuning?"),
    ("Multi-agent failures", "How do multi-agent LLM systems coordinate, and where do they fail?"),
]

# (key, icon, name, what it does)
STEPS = [
    ("researcher", "🔎", "Researcher", "Searches arXiv and the web"),
    ("analyst", "🧠", "Analyst", "Extracts and compares findings"),
    ("writer", "✍️", "Writer", "Drafts the cited report"),
]

st.set_page_config(page_title="ScholarCrew", page_icon="🔎", layout="centered")

# Neutral translucent colours so the cards look right in both light and dark themes.
st.markdown(
    """
    <style>
      .block-container { max-width: 880px; padding-top: 2.5rem; padding-bottom: 3rem; }
      #MainMenu, footer { visibility: hidden; }
      .hero h1 { margin: 0; font-size: 2.3rem; letter-spacing: -0.02em; }
      .hero p { margin: .35rem 0 0; opacity: .7; font-size: 1.02rem; }
      .pipe { display: grid; grid-template-columns: repeat(3, 1fr); gap: .75rem; margin: 1.25rem 0; }
      .step { border: 1px solid rgba(128,128,128,.28); border-radius: 12px; padding: .8rem .95rem;
              background: rgba(128,128,128,.06); transition: all .2s; }
      .step .n { font-size: .72rem; letter-spacing: .06em; text-transform: uppercase; opacity: .55; }
      .step .t { font-weight: 650; margin-top: .1rem; }
      .step .d { font-size: .85rem; opacity: .7; margin-top: .1rem; }
      .step.idle { opacity: .6; }
      .step.active { border-color: #4f46e5; box-shadow: 0 0 0 3px rgba(79,70,229,.18); }
      .step.done { border-color: rgba(34,197,94,.6); background: rgba(34,197,94,.08); }
      .step .s { float: right; font-size: .8rem; }
      div[data-testid="stMetric"] { border: 1px solid rgba(128,128,128,.25); border-radius: 12px;
              padding: .7rem .9rem; background: rgba(128,128,128,.05); }
      .stButton > button { border-radius: 10px; }
      .foot { text-align: center; opacity: .55; font-size: .85rem; margin-top: 2.5rem; }
      @media (max-width: 640px) {
        .pipe { grid-template-columns: 1fr; gap: .5rem; }
        .step .d { font-size: .82rem; }
        .hero h1 { font-size: 1.9rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def pipeline_html(status: dict[str, str]) -> str:
    """Static markup only: no user or model text is ever interpolated into HTML."""
    mark = {"idle": "", "active": "⏳ working", "done": "✓ done"}
    cards = "".join(
        f'<div class="step {status[key]}"><span class="s">{mark[status[key]]}</span>'
        f'<div class="n">Step {i}</div><div class="t">{icon} {name}</div>'
        f'<div class="d">{desc}</div></div>'
        for i, (key, icon, name, desc) in enumerate(STEPS, 1)
    )
    return f'<div class="pipe">{cards}</div>'


def queue_run() -> None:
    st.session_state.run_requested = True


def use_example(text: str) -> None:
    st.session_state.question = text
    st.session_state.run_requested = True


def render_sidebar():
    with st.sidebar:
        st.header("Your LLM")
        name = st.selectbox("Provider", list(PROVIDERS))
        provider = PROVIDERS[name]
        api_key = st.text_input("API key", type="password", placeholder="Paste your key")
        if provider.key_url:
            st.markdown(f"No key? [Get a free one →]({provider.key_url})")
        with st.expander("Advanced", expanded=not provider.base_url):
            base_url = st.text_input("Base URL", value=provider.base_url, key=f"url-{name}")
            model = st.text_input("Model", value=provider.default_model, key=f"model-{name}")
        st.divider()
        st.caption(
            "🔒 Your key lives only in this browser session. It is never saved or logged, and "
            "it is sent only to the provider you picked."
        )
        st.caption(f"[Source code on GitHub]({REPO_URL})")
    return settings_for(base_url, api_key, model)


def run_crew(question: str, settings, board) -> dict | None:
    """Run the agents, updating `board` live. Returns the result dict, or None on failure."""
    status = {key: "idle" for key, *_ in STEPS}
    board.markdown(pipeline_html(status), unsafe_allow_html=True)
    log: list[str] = []
    sources: list[dict] = []
    report = notes = ""
    started = time.perf_counter()

    with st.status("Checking your key and model...", expanded=False) as box:
        error = check_llm(settings)
        if error:
            box.update(label="Could not reach the model", state="error")
            st.error(error)
            return None

        box.update(label="Agents are working...", expanded=True)
        try:
            for node, update in stream_crew(question, make_llm(settings), settings):
                for line in update.get("log", []):
                    log.append(f"{node}: {line}")
                    st.write(f"**{node}** — {line}")
                if node == "supervisor" and update.get("next") in status:
                    status[update["next"]] = "active"
                elif node in status:
                    status[node] = "done"
                sources = update.get("sources", sources)
                notes = update.get("notes", notes)
                report = update.get("report", report)
                board.markdown(pipeline_html(status), unsafe_allow_html=True)
        except Exception as exc:  # noqa: BLE001 - show the problem instead of a stack trace
            box.update(label="Something went wrong", state="error")
            st.error(f"{type(exc).__name__}: {str(exc).replace(settings.api_key, '***')[:300]}")
            return None
        box.update(label="Done", state="complete", expanded=False)

    return {
        "question": question,
        "report": report,
        "notes": notes,
        "sources": sources,
        "log": log,
        "seconds": time.perf_counter() - started,
    }


def render_result(result: dict) -> None:
    sources = result["sources"]
    arxiv = sum(1 for s in sources if s["kind"] == "arxiv")
    cols = st.columns(4)
    cols[0].metric("Sources", len(sources))
    cols[1].metric("arXiv papers", arxiv)
    cols[2].metric("Web pages", len(sources) - arxiv)
    cols[3].metric("Time", f"{result['seconds']:.0f}s")

    report_tab, sources_tab, log_tab = st.tabs(["📄 Report", "📚 Sources", "🧾 Agent log"])
    with report_tab:
        if result["report"]:
            st.markdown(result["report"])
            st.download_button("⬇ Download report (.md)", result["report"], "report.md")
        else:
            st.warning("No report was produced. Try rephrasing the question.")
    with sources_tab:
        for s in sources:
            title = s["title"].replace("[", "(").replace("]", ")")
            with st.container(border=True):
                st.markdown(f"**[{title}]({s['url']})**  \n`{s['kind']}`")
                st.caption(s["text"][:280] + ("…" if len(s["text"]) > 280 else ""))
        if not sources:
            st.info("No sources were found.")
    with log_tab:
        st.code("\n".join(result["log"]) or "No log.", language="text")


# ---------------------------------------------------------------- page

settings = render_sidebar()
issue = problems(settings)
result = st.session_state.get("result")

st.markdown(
    '<div class="hero"><h1>🔎 ScholarCrew</h1>'
    "<p>A team of AI agents researches your question on arXiv and the web, "
    "then writes a report with citations.</p></div>",
    unsafe_allow_html=True,
)

board_slot = st.empty()
board_slot.markdown(
    pipeline_html({key: "done" if result else "idle" for key, *_ in STEPS}),
    unsafe_allow_html=True,
)

question_col, button_col = st.columns([5, 1], vertical_alignment="bottom")
question_col.text_input(
    "Research question",
    key="question",
    max_chars=300,
    placeholder="Ask a research question…",
    on_change=queue_run,
    label_visibility="collapsed",
)
button_col.button("Research", type="primary", use_container_width=True, on_click=queue_run)

st.caption("Try an example")
example_cols = st.columns(2)
for i, (label, text) in enumerate(EXAMPLES):
    example_cols[i % 2].button(
        label, key=f"ex{i}", on_click=use_example, args=(text,), use_container_width=True
    )

if issue:
    st.info(f"**Get started:** {issue}  \nOpen the sidebar (top left) if it is collapsed.")

question = st.session_state.get("question", "").strip()
if st.session_state.pop("run_requested", False) and not issue and question:
    result = run_crew(question, settings, board_slot)
    st.session_state.result = result
    if result is None:
        st.stop()
    st.rerun()  # redraw with the finished pipeline strip and results in place

if result:
    st.divider()
    st.subheader(result["question"])
    render_result(result)

st.markdown(
    f'<div class="foot">Open source · <a href="{REPO_URL}">GitHub</a></div>',
    unsafe_allow_html=True,
)
