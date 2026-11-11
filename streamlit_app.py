"""ScholarCrew web app (Streamlit). Visitors bring their own API key; nothing is stored."""

from __future__ import annotations

import sys
from pathlib import Path

# Lets the app run on Streamlit Community Cloud without installing the package first.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

from scholarcrew.graph import stream_crew
from scholarcrew.providers import PROVIDERS, check_llm, problems, settings_for

st.set_page_config(page_title="ScholarCrew", page_icon="🔎", layout="centered")

st.title("🔎 ScholarCrew")
st.caption(
    "A supervisor coordinates researcher, analyst and writer agents to answer a question "
    "with citations from arXiv and the web."
)

with st.sidebar:
    st.header("Your LLM")
    provider_name = st.selectbox("Provider", list(PROVIDERS))
    provider = PROVIDERS[provider_name]
    base_url = st.text_input("Base URL", value=provider.base_url, key=f"url-{provider_name}")
    model = st.text_input("Model", value=provider.default_model, key=f"model-{provider_name}")
    api_key = st.text_input("API key", type="password", help="Used only for this session.")
    if provider.key_url:
        st.markdown(f"No key? [Get a free one]({provider.key_url}).")
    st.caption(
        "Your key stays in this browser session. It is not saved, logged, or shared, and it "
        "is sent only to the provider you chose."
    )

settings = settings_for(base_url, api_key, model)
issue = problems(settings)

question = st.text_input(
    "Research question",
    max_chars=300,
    placeholder="How do researchers reduce hallucination in retrieval-augmented generation?",
)
go = st.button("Research", type="primary", disabled=bool(issue) or not question.strip())

if issue:
    st.info(issue)

if go:
    with st.status("Checking your key and model...", expanded=True) as status:
        error = check_llm(settings)
        if error:
            status.update(label="Could not reach the model", state="error")
            st.error(error)
            st.stop()

        report = ""
        status.update(label="Agents are working...")
        try:
            from scholarcrew.llm import make_llm

            for node, update in stream_crew(question.strip(), make_llm(settings), settings):
                for line in update.get("log", []):
                    st.write(f"**{node}**: {line}")
                report = update.get("report", report)
        except Exception as exc:  # noqa: BLE001 - show the problem instead of a stack trace
            status.update(label="Something went wrong", state="error")
            st.error(f"{type(exc).__name__}: {str(exc).replace(settings.api_key, '***')[:300]}")
            st.stop()
        status.update(label="Done", state="complete", expanded=False)

    if report:
        st.markdown(report)
        st.download_button("Download report (.md)", report, file_name="report.md")
    else:
        st.warning("No report was produced. Try rephrasing the question.")
