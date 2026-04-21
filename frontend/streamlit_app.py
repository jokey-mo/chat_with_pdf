from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Iterator

import httpx
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Agro-RAG", page_icon="🌱", layout="wide")


def api_get(path: str, **kwargs):
    r = requests.get(f"{API_URL}{path}", timeout=30, **kwargs)
    r.raise_for_status()
    return r.json()


def api_post(path: str, json_body: dict | None = None):
    r = requests.post(f"{API_URL}{path}", json=json_body, timeout=30)
    r.raise_for_status()
    return r.json()


def api_delete(path: str):
    r = requests.delete(f"{API_URL}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def stream_chat(notebook_id: int, messages: list[dict]) -> Iterator[dict]:
    with httpx.stream(
        "POST",
        f"{API_URL}/chat",
        json={"notebook_id": notebook_id, "messages": messages},
        timeout=None,
    ) as r:
        for line in r.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue


def sidebar_notebooks():
    st.sidebar.header("📚 Notebooks")
    notebooks = api_get("/notebooks")
    if not notebooks:
        st.sidebar.info("Нет ноутбуков. Создайте ниже.")
    else:
        names = {f"{nb['id']}: {nb['name']}": nb for nb in notebooks}
        choice = st.sidebar.radio("Выбрать", list(names.keys()), key="nb_choice")
        st.session_state["current_nb"] = names[choice]

    with st.sidebar.expander("➕ Создать ноутбук", expanded=not notebooks):
        with st.form("create_nb"):
            name = st.text_input("Название", "Hydroponics Yield")
            topic = st.text_area("Запрос (тема)", "hydroponics lettuce yield nutrient solution")
            sources = st.multiselect(
                "Источники", ["openalex", "agrirxiv"], default=["openalex"]
            )
            cron = st.text_input("Cron (пусто = ручной запуск)", "")
            model = st.text_input("OpenRouter модель (пусто = default)", "")
            submitted = st.form_submit_button("Создать")
            if submitted:
                api_post(
                    "/notebooks",
                    {
                        "name": name,
                        "topic_query": topic,
                        "sources": sources,
                        "schedule_cron": cron or None,
                        "openrouter_model": model or None,
                    },
                )
                st.rerun()


def render_citations(citations: list[dict]):
    if not citations:
        return
    st.markdown("##### Источники")
    for c in citations:
        authors = ", ".join(c.get("authors") or [])[:120]
        year = c.get("year") or ""
        doi = c.get("doi")
        title = c.get("paper_title")
        marker = c.get("marker")
        paper_id = c.get("paper_id")
        link = f"https://doi.org/{doi}" if doi else f"{API_URL}/papers/{paper_id}/pdf"
        st.markdown(
            f"- **[{marker}]** [{title}]({link}) — {authors} ({year})"
        )


def chat_view(nb: dict):
    st.title(f"🌱 {nb['name']}")
    st.caption(f"Topic: {nb['topic_query']}")

    if "chats" not in st.session_state:
        st.session_state["chats"] = {}
    chat_key = f"nb{nb['id']}"
    history: list[dict] = st.session_state["chats"].setdefault(chat_key, [])

    for m in history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("citations"):
                render_citations(m["citations"])

    prompt = st.chat_input("Задайте вопрос по статьям…")
    if prompt:
        history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            cite_box = st.empty()
            full = ""
            citations: list[dict] = []
            for event in stream_chat(
                nb["id"], [{"role": m["role"], "content": m["content"]} for m in history]
            ):
                if event["type"] == "citations":
                    citations = event.get("citations", [])
                    with cite_box.container():
                        render_citations(citations)
                elif event["type"] == "delta":
                    full += event.get("text", "")
                    placeholder.markdown(_linkify_markers(full, citations))
            history.append({"role": "assistant", "content": full, "citations": citations})


def _linkify_markers(text: str, citations: list[dict]) -> str:
    by_marker = {c["marker"]: c for c in citations}

    def repl(m: re.Match) -> str:
        marker = m.group(1)
        c = by_marker.get(marker)
        if not c:
            return m.group(0)
        doi = c.get("doi")
        link = f"https://doi.org/{doi}" if doi else f"{API_URL}/papers/{c['paper_id']}/pdf"
        return f"[{m.group(0)}]({link})"

    return re.sub(r"\[(P\d+)(?:\s+p\.\d+)?\]", repl, text)


def ingestion_dashboard(nb: dict):
    with st.expander("⚙️ Ингестия и статьи"):
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("▶ Запустить ингестию", type="primary"):
                res = api_post(f"/notebooks/{nb['id']}/ingest")
                st.success(f"Задача: {res['job_id']}")
            if st.button("🗑 Удалить ноутбук"):
                api_delete(f"/notebooks/{nb['id']}")
                st.rerun()
        with col2:
            runs = api_get(f"/notebooks/{nb['id']}/runs")
            if runs:
                st.markdown("**Последние прогоны**")
                st.dataframe(
                    [
                        {
                            "started": r["started_at"],
                            "finished": r["finished_at"] or "-",
                            "found": r["n_found"],
                            "new": r["n_new"],
                            "ok": r["n_embedded"],
                            "fail": r["n_failed"],
                        }
                        for r in runs[:10]
                    ],
                    use_container_width=True,
                )
        papers = api_get(f"/notebooks/{nb['id']}/papers")
        if papers:
            st.markdown(f"**Статьи ({len(papers)})**")
            st.dataframe(
                [
                    {
                        "id": p["id"],
                        "title": p["title"][:110],
                        "year": p["year"],
                        "source": p["source"],
                        "status": p["status"],
                        "doi": p["doi"] or "",
                    }
                    for p in papers
                ],
                use_container_width=True,
            )


def main():
    sidebar_notebooks()
    nb = st.session_state.get("current_nb")
    if not nb:
        st.info("Создайте или выберите ноутбук слева.")
        return
    ingestion_dashboard(nb)
    chat_view(nb)


if __name__ == "__main__":
    main()
