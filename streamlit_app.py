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
    ("LoRA vs fine-tuning", "How does LoRA fine-tuning compare with full fine-tuning?"),
    ("Multi-agent failures", "How do multi-agent LLM systems coordinate, and where do they fail?"),
]

# (key, name, what it does)
STEPS = [
    ("researcher", "Researcher", "Searches arXiv and the web"),
    ("analyst", "Analyst", "Extracts and compares findings"),
    ("writer", "Writer", "Drafts the cited report"),
]
STATE_LABEL = {"idle": "Idle", "active": "Running", "done": "Complete"}

st.set_page_config(
    page_title="ScholarCrew",
    page_icon=":material/travel_explore:",
    layout="centered",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
      :root {
        --bg: #080a10; --surface: rgba(20,24,34,.62); --surface-solid: #12161f;
        --border: rgba(255,255,255,.08); --border-strong: rgba(255,255,255,.16);
        --text: #eceef4; --muted: #8e97ac; --accent: #6366f1; --accent-2: #8b5cf6;
        --cyan: #22d3ee; --ok: #34d399;
      }
      html, body, .stApp, button, input, textarea, [data-baseweb] {
        font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif !important;
      }
      .stApp {
        background:
          radial-gradient(900px 520px at 12% -8%, rgba(99,102,241,.26), transparent 60%),
          radial-gradient(760px 440px at 92% 2%, rgba(34,211,238,.15), transparent 55%),
          radial-gradient(700px 500px at 50% 110%, rgba(139,92,246,.12), transparent 60%),
          var(--bg);
        background-attachment: fixed;
      }
      .block-container { max-width: 880px; padding-top: 3.2rem; padding-bottom: 3rem; }

      /* Hide Streamlit chrome that is not useful to visitors */
      #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
      [data-testid="stHeader"] { background: transparent; }

      /* ---------- Hero ---------- */
      .hero { animation: rise .6s ease both; }
      .badge { display: inline-flex; align-items: center; gap: .5rem; padding: .3rem .8rem;
               border: 1px solid var(--border-strong); border-radius: 999px; background: var(--surface);
               color: #c7d2fe; font-size: .74rem; font-weight: 600; letter-spacing: .1em;
               text-transform: uppercase; backdrop-filter: blur(8px); }
      .badge i { width: 7px; height: 7px; border-radius: 50%; background: var(--cyan);
                 box-shadow: 0 0 10px var(--cyan); animation: pulse 1.8s infinite; }
      .hero h1 { margin: 1rem 0 .6rem; font-size: 3.4rem; line-height: 1.05; font-weight: 800;
                 letter-spacing: -.045em;
                 background: linear-gradient(95deg, #ffffff 0%, #c7d2fe 38%, #67e8f9 100%);
                 -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
      .hero p { margin: 0; color: var(--muted); font-size: 1.08rem; line-height: 1.65; max-width: 38rem; }
      .chips { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.1rem; }
      .chips span { padding: .22rem .7rem; border: 1px solid var(--border); border-radius: 999px;
                    color: var(--muted); font-size: .78rem; background: rgba(255,255,255,.03); }

      /* ---------- Agent pipeline ---------- */
      .pipe { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.1rem; margin: 2rem 0 .6rem; }
      .step { position: relative; background: var(--surface); border: 1px solid var(--border);
              border-radius: 14px; padding: 1rem 1.05rem; backdrop-filter: blur(10px);
              transition: border-color .25s, box-shadow .25s, transform .25s; }
      .step:not(:last-child)::after { content: ""; position: absolute; top: 50%; right: -1.1rem;
              width: 1.1rem; height: 2px; background: var(--border-strong); }
      .step .top { display: flex; justify-content: space-between; align-items: center; }
      .step .n { color: var(--muted); font-size: .74rem; font-weight: 700; letter-spacing: .1em; }
      .step .st { display: flex; align-items: center; gap: .4rem; color: var(--muted); font-size: .74rem; }
      .step .dot { width: 7px; height: 7px; border-radius: 50%; background: #3a4154; display: inline-block; }
      .step .t { font-weight: 650; margin-top: .65rem; font-size: 1rem; }
      .step .d { color: var(--muted); font-size: .84rem; margin-top: .2rem; }
      .step.active { border-color: rgba(99,102,241,.8); transform: translateY(-2px);
              box-shadow: 0 0 0 1px rgba(99,102,241,.5), 0 10px 40px -8px rgba(99,102,241,.55); }
      .step.active .dot { background: var(--accent); box-shadow: 0 0 10px var(--accent); animation: pulse 1.1s infinite; }
      .step.active .st { color: var(--text); }
      .step.done { border-color: rgba(52,211,153,.35); }
      .step.done::after { background: rgba(52,211,153,.5) !important; }
      .step.done .dot { background: var(--ok); box-shadow: 0 0 8px rgba(52,211,153,.7); }
      .step.done .st { color: var(--ok); }
      .bar { height: 3px; border-radius: 3px; background: rgba(255,255,255,.06); overflow: hidden; margin-bottom: 1.8rem; }
      .bar span { display: block; height: 100%; border-radius: 3px; transition: width .5s ease;
                  background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--cyan)); }
      @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }
      @keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }

      /* ---------- Inputs and buttons ---------- */
      .lbl { color: var(--muted); font-size: .74rem; font-weight: 700; letter-spacing: .1em;
             text-transform: uppercase; margin: 0 0 .5rem; }
      [data-baseweb="input"], [data-baseweb="select"] > div {
        background: var(--surface-solid) !important; border: 1px solid var(--border-strong) !important;
        border-radius: 12px !important; transition: border-color .2s, box-shadow .2s; }
      [data-baseweb="input"]:focus-within { border-color: var(--accent) !important;
        box-shadow: 0 0 0 4px rgba(99,102,241,.18); }
      .stTextInput input { height: 2.9rem; }
      .stButton > button { border-radius: 12px; font-weight: 550; transition: all .18s; }
      .stButton > button[kind="primary"] { height: 2.9rem; border: 0; color: #fff;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        box-shadow: 0 8px 24px -8px rgba(99,102,241,.8); }
      .stButton > button[kind="primary"]:hover { transform: translateY(-1px); filter: brightness(1.1);
        box-shadow: 0 12px 28px -8px rgba(99,102,241,.95); }
      .stButton > button[kind="secondary"] { background: rgba(255,255,255,.03); border: 1px solid var(--border);
        color: var(--muted); font-size: .84rem; border-radius: 999px; }
      .stButton > button[kind="secondary"]:hover { border-color: var(--accent); color: var(--text);
        background: rgba(99,102,241,.1); }

      /* ---------- Getting-started panel ---------- */
      .guide { background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
               padding: 1.2rem 1.4rem; margin-top: 1.4rem; backdrop-filter: blur(10px); }
      .guide .gt { font-weight: 650; margin-bottom: .6rem; }
      .guide ol { margin: 0; padding-left: 1.2rem; color: var(--muted); line-height: 1.95; }
      .guide a { color: #a5b4fc; }

      /* ---------- Results ---------- */
      div[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border);
              border-radius: 14px; padding: .9rem 1.1rem; backdrop-filter: blur(10px); }
      div[data-testid="stMetricLabel"] p { color: var(--muted); font-size: .8rem; }
      div[data-testid="stMetricValue"] { width: fit-content; font-weight: 700;
              background: linear-gradient(95deg, #fff, #a5b4fc); -webkit-background-clip: text;
              background-clip: text; -webkit-text-fill-color: transparent; }
      [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid var(--border); }
      [data-baseweb="tab"] { padding: .55rem 0; }
      [data-baseweb="tab-highlight"] { background: linear-gradient(90deg, var(--accent), var(--cyan)); }
      div[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--border); border-radius: 14px;
              background: var(--surface); }
      .stMarkdown p, .stMarkdown li { line-height: 1.75; }
      .stMarkdown table { border-collapse: collapse; }
      .stMarkdown th, .stMarkdown td { border: 1px solid var(--border-strong) !important; padding: .45rem .8rem; }

      section[data-testid="stSidebar"] { border-right: 1px solid var(--border); }
      .foot { text-align: center; color: var(--muted); font-size: .82rem; margin-top: 3rem; }
      .foot a { color: var(--muted); }

      @media (max-width: 640px) {
        .pipe { grid-template-columns: 1fr; gap: .6rem; }
        .step:not(:last-child)::after { display: none; }
        .hero h1 { font-size: 2.5rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def pipeline_html(status: dict[str, str]) -> str:
    """Static markup only: no user or model text is ever interpolated into HTML."""
    cards = "".join(
        f'<div class="step {status[key]}"><div class="top"><span class="n">{i:02d}</span>'
        f'<span class="st"><i class="dot"></i>{STATE_LABEL[status[key]]}</span></div>'
        f'<div class="t">{name}</div><div class="d">{desc}</div></div>'
        for i, (key, name, desc) in enumerate(STEPS, 1)
    )
    progress = sum({"done": 1.0, "active": 0.5}.get(s, 0.0) for s in status.values()) / len(STEPS)
    return (
        f'<div class="pipe">{cards}</div>'
        f'<div class="bar"><span style="width:{progress * 100:.0f}%"></span></div>'
    )


def queue_run() -> None:
    st.session_state.run_requested = True


def use_example(text: str) -> None:
    st.session_state.question = text
    st.session_state.run_requested = True


def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="lbl">LLM settings</div>', unsafe_allow_html=True)
        name = st.selectbox("Provider", list(PROVIDERS))
        provider = PROVIDERS[name]
        api_key = st.text_input("API key", type="password", placeholder="Paste your key")
        if provider.key_url:
            st.markdown(f"[Get a free {name.split(' (')[0]} key]({provider.key_url})")
        with st.expander("Advanced", expanded=not provider.base_url):
            base_url = st.text_input("Base URL", value=provider.base_url, key=f"url-{name}")
            model = st.text_input("Model", value=provider.default_model, key=f"model-{name}")
        st.divider()
        st.caption(
            "Your key lives only in this browser session. It is never saved or logged, and "
            "it is sent only to the provider you picked."
        )
        st.caption(f"[Source code]({REPO_URL})")
    return settings_for(base_url, api_key, model), name, provider


def run_crew(question: str, settings, board) -> dict | None:
    """Run the agents, updating `board` live. Returns the result dict, or None on failure."""
    status = {key: "idle" for key, *_ in STEPS}
    board.markdown(pipeline_html(status), unsafe_allow_html=True)
    log: list[str] = []
    sources: list[dict] = []
    report = notes = ""
    started = time.perf_counter()

    with st.status("Checking your key and model", expanded=False) as box:
        error = check_llm(settings)
        if error:
            box.update(label="Could not reach the model", state="error")
            st.error(error)
            return None

        box.update(label="Agents are working", expanded=True)
        try:
            for node, update in stream_crew(question, make_llm(settings), settings):
                for line in update.get("log", []):
                    log.append(f"{node}: {line}")
                    st.write(f"**{node}**: {line}")
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

    report_tab, sources_tab, log_tab = st.tabs(["Report", "Sources", "Agent log"])
    with report_tab:
        if result["report"]:
            with st.container(border=True):
                st.markdown(result["report"])
            st.download_button("Download report", result["report"], "report.md")
        else:
            st.warning("No report was produced. Try rephrasing the question.")
    with sources_tab:
        for s in sources:
            title = s["title"].replace("[", "(").replace("]", ")")
            with st.container(border=True):
                st.markdown(f"**[{title}]({s['url']})**  \n`{s['kind']}`")
                st.caption(s["text"][:280] + ("..." if len(s["text"]) > 280 else ""))
        if not sources:
            st.info("No sources were found.")
    with log_tab:
        st.code("\n".join(result["log"]) or "No log.", language="text")


# ---------------------------------------------------------------- page

settings, provider_name, provider = render_sidebar()
issue = problems(settings)
result = st.session_state.get("result")

st.markdown(
    '<div class="hero"><span class="badge"><i></i>Multi-agent research assistant</span>'
    "<h1>ScholarCrew</h1>"
    "<p>A supervisor coordinates researcher, analyst and writer agents to answer your question "
    "from arXiv and the web, with citations.</p>"
    '<div class="chips"><span>arXiv</span><span>Web search</span><span>LangGraph</span>'
    "<span>Cited reports</span><span>Bring your own key</span></div></div>",
    unsafe_allow_html=True,
)

board_slot = st.empty()
board_slot.markdown(
    pipeline_html({key: "done" if result else "idle" for key, *_ in STEPS}),
    unsafe_allow_html=True,
)

st.markdown('<div class="lbl">Research question</div>', unsafe_allow_html=True)
question_col, button_col = st.columns([5, 1], vertical_alignment="bottom")
question_col.text_input(
    "Research question",
    key="question",
    max_chars=300,
    placeholder="Ask a research question",
    on_change=queue_run,
    label_visibility="collapsed",
)
button_col.button("Research", type="primary", use_container_width=True, on_click=queue_run)

example_cols = st.columns(len(EXAMPLES))
for col, (label, text), i in zip(example_cols, EXAMPLES, range(len(EXAMPLES))):
    col.button(label, key=f"ex{i}", on_click=use_example, args=(text,), use_container_width=True)

if not settings.api_key:
    key_link = (
        f'<a href="{provider.key_url}" target="_blank">{provider_name.split(" (")[0]}</a>'
        if provider.key_url
        else "your provider"
    )
    st.markdown(
        '<div class="guide"><div class="gt">Get started</div><ol>'
        f"<li>Create a free API key at {key_link}.</li>"
        "<li>Paste it into the API key field in the sidebar (top left arrow if collapsed).</li>"
        "<li>Ask a question or pick an example above.</li></ol></div>",
        unsafe_allow_html=True,
    )
elif issue:
    st.warning(issue)

question = st.session_state.get("question", "").strip()
if st.session_state.pop("run_requested", False) and not issue and question:
    result = run_crew(question, settings, board_slot)
    st.session_state.result = result
    if result is None:
        st.stop()
    st.rerun()  # redraw with the finished pipeline strip and results in place

if result:
    st.divider()
    st.markdown('<div class="lbl">Results</div>', unsafe_allow_html=True)
    st.subheader(result["question"])
    render_result(result)

st.markdown(
    f'<div class="foot">Open source &middot; <a href="{REPO_URL}">GitHub</a></div>',
    unsafe_allow_html=True,
)
