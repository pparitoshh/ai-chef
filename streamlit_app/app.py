"""AI Chef Streamlit UI: chat-style recommendations with feedback.

Talks to the FastAPI backend (POST /recommend, POST /feedback).

Run:  uv run streamlit run streamlit_app/app.py
"""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Chef", page_icon="🍳")
st.title("🍳 AI Chef")
st.caption("Tell me what you feel like eating — cuisine, dish type, diet, "
           "and your cooking level — and I'll find the right recipe for you.")

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, data?}]
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = set()  # conversation_ids already rated


def recipe_card(r: dict):
    st.markdown(
        f"**{r['name'].title()}**  \n"
        f"🌍 {r['cuisine']} · 🍽️ {r['dish_type']} · 🥗 {r['diet']} · "
        f"📶 {r['skill_level']} · ⏱️ {r['minutes']} min"
    )


# --- history ------------------------------------------------------------------
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("data"):
            data = msg["data"]
            if data["filters"]:
                st.caption("filters: " + ", ".join(
                    f"{k}={v}" for k, v in data["filters"].items()))
            with st.expander("Recommended dishes", expanded=True):
                for r in data["recipes"]:
                    recipe_card(r)
            conv_id = data["conversation_id"]
            if conv_id not in st.session_state.feedback_given:
                c1, c2, _ = st.columns([1, 1, 8])
                if c1.button("👍", key=f"up_{conv_id}"):
                    requests.post(f"{API_URL}/feedback",
                                  json={"conversation_id": conv_id, "feedback": 1})
                    st.session_state.feedback_given.add(conv_id)
                    st.rerun()
                if c2.button("👎", key=f"down_{conv_id}"):
                    requests.post(f"{API_URL}/feedback",
                                  json={"conversation_id": conv_id, "feedback": -1})
                    st.session_state.feedback_given.add(conv_id)
                    st.rerun()
            else:
                st.caption("Thanks for the feedback!")

# --- input --------------------------------------------------------------------
if question := st.chat_input("e.g. quick vegetarian indian dinner, I'm a beginner"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            resp = requests.post(f"{API_URL}/recommend",
                                 json={"question": question}, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            st.markdown(data["answer"])
            if data["filters"]:
                st.caption("filters: " + ", ".join(
                    f"{k}={v}" for k, v in data["filters"].items()))
            with st.expander("Recommended dishes", expanded=True):
                for r in data["recipes"]:
                    recipe_card(r)
            st.session_state.messages.append(
                {"role": "assistant", "content": data["answer"], "data": data})
            st.rerun()  # re-render so feedback buttons appear
        except requests.RequestException as e:
            st.error(f"API error: {e}")
