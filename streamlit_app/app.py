"""AI Chef Streamlit UI: conversational multi-turn recommendations.

Asks 4 guided questions (cuisine, dish type, diet, skill level),
then recommends matching recipes with similar dishes.

Talks to the FastAPI backend (POST /recommend, POST /feedback).

Run:  uv run streamlit run streamlit_app/app.py
"""

import os
import re
import sys
import uuid
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# streamlit run only puts this file's own directory on sys.path — add the
# repo root too so `from api import ...` resolves (e.g. on Streamlit Cloud).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000"))
# "api" (default, talks to FastAPI + Postgres) or "memory" (in-process search,
# no database needed — used for the Streamlit Cloud deployment for now).
SEARCH_BACKEND = st.secrets.get("SEARCH_BACKEND", os.getenv("SEARCH_BACKEND", "api"))

if SEARCH_BACKEND == "memory":
    from api import memory_search, rag_memory

st.set_page_config(page_title="AI Chef", page_icon="🍳")
st.title("🍳 AI Chef")
st.caption("Let's find you the perfect recipe. Answer a few quick questions.")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "answers" not in st.session_state:
    st.session_state.answers = {}  # {cuisine, dish_type, diet, skill_level}
if "current_step" not in st.session_state:
    st.session_state.current_step = 0  # 0-3 for questions, 4 for recommendation
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = set()
if "recommendation_id" not in st.session_state:
    st.session_state.recommendation_id = None
if "question_asked_for_step" not in st.session_state:
    st.session_state.question_asked_for_step = -1
if "recipe_details" not in st.session_state:
    st.session_state.recipe_details = {}  # {recipe_id: detail dict}
if "expanded_recipes" not in st.session_state:
    st.session_state.expanded_recipes = set()  # recipe_ids currently expanded

# Conversational questions and valid responses
QUESTIONS = [
    {
        "step": 0,
        "prompt": "🌍 What cuisine are you craving?",
        "help": "(e.g., Indian, Mexican, Italian, Asian, Mediterranean)",
        "key": "cuisine",
    },
    {
        "step": 1,
        "prompt": "🍽️ What kind of dish are you craving?",
        "help": "(e.g., carbonara pasta, butter chicken, tacos, stir-fry, soup, "
                "a quick 30-min dinner, or just say curry / dessert / one-pot)",
        "key": "dish_type",
    },
    {
        "step": 2,
        "prompt": "🥗 What protein do you eat?",
        "help": "(e.g., vegan, vegetarian, chicken, pork, beef, everything)",
        "key": "diet",
    },
    {
        "step": 3,
        "prompt": "👨‍🍳 What's your cooking level?",
        "help": "(e.g., beginner, intermediate, advanced)",
        "key": "skill_level",
    },
    {
        "step": 4,
        "prompt": "✏️ Anything else you'd like to add?",
        "help": "(optional — e.g., \"extra spicy masala curry\", \"no onions\", "
                "\"something creamy\". Or just type skip)",
        "key": "extra_details",
    },
]


def fetch_recipe_detail(recipe_id: str):
    if recipe_id not in st.session_state.recipe_details:
        if SEARCH_BACKEND == "memory":
            detail = memory_search.get_recipe_detail(recipe_id)
        else:
            resp = requests.get(f"{API_URL}/recipes/{recipe_id}", timeout=30)
            resp.raise_for_status()
            detail = resp.json()
        st.session_state.recipe_details[recipe_id] = detail
    return st.session_state.recipe_details[recipe_id]


def recipe_card(r: dict, key_prefix: str):
    recipe_id = r["id"]
    st.markdown(
        f"🌍 {r['cuisine']} · 🍽️ {r['dish_type']} · 🥗 {r['diet']} · "
        f"📶 {r['skill_level']} · ⏱️ {r['minutes']} min"
    )
    if st.button(f"👉 {r['name'].title()}", key=f"{key_prefix}_{recipe_id}"):
        if recipe_id in st.session_state.expanded_recipes:
            st.session_state.expanded_recipes.discard(recipe_id)
        else:
            st.session_state.expanded_recipes.add(recipe_id)
        st.rerun()

    if recipe_id in st.session_state.expanded_recipes:
        try:
            detail = fetch_recipe_detail(recipe_id)
            with st.container(border=True):
                if detail.get("calories"):
                    st.caption(f"🔥 {detail['calories']:.0f} calories")
                st.markdown("**Ingredients:**")
                st.markdown("\n".join(f"- {i}" for i in detail["ingredients"]))
                st.markdown("**Steps:**")
                st.markdown("\n".join(
                    f"{i}. {s}" for i, s in enumerate(detail["steps"], 1)))
        except requests.RequestException as e:
            st.error(f"Couldn't load recipe: {e}")
    st.divider()


def get_current_question():
    if st.session_state.current_step < len(QUESTIONS):
        return QUESTIONS[st.session_state.current_step]
    return None


def normalize_response(text: str, key: str) -> str:
    text = text.strip().lower()
    if key == "extra_details" and text in {"skip", "none", "no", "n/a", "na", "-"}:
        return ""
    text = re.sub(r'\b(i eat|i\'m|i am|i prefer)\s+', '', text)
    text = re.sub(r'\s*(please|thanks|etc\.?)\s*', '', text)
    return text.strip()


def add_assistant_message(content: str, data: dict = None):
    st.session_state.messages.append({
        "role": "assistant",
        "content": content,
        "data": data
    })


def add_user_message(content: str):
    st.session_state.messages.append({
        "role": "user",
        "content": content
    })


# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show recommendation data if present
        if msg["role"] == "assistant" and msg.get("data"):
            data = msg["data"]
            if data.get("filters"):
                st.caption("Filters: " + ", ".join(
                    f"{k}={v}" for k, v in data["filters"].items()))

            conv_id = data.get("conversation_id")
            with st.expander("📋 Recommended dishes", expanded=True):
                for r in data.get("recipes", []):
                    recipe_card(r, key_prefix=conv_id)

            # Feedback buttons
            if conv_id and conv_id not in st.session_state.feedback_given:
                c1, c2, _ = st.columns([1, 1, 8])
                if c1.button("👍 Helpful", key=f"up_{conv_id}"):
                    if SEARCH_BACKEND != "memory":
                        requests.post(f"{API_URL}/feedback",
                                      json={"conversation_id": conv_id, "feedback": 1})
                    st.session_state.feedback_given.add(conv_id)
                    st.rerun()
                if c2.button("👎 Not helpful", key=f"down_{conv_id}"):
                    if SEARCH_BACKEND != "memory":
                        requests.post(f"{API_URL}/feedback",
                                      json={"conversation_id": conv_id, "feedback": -1})
                    st.session_state.feedback_given.add(conv_id)
                    st.rerun()
            elif conv_id in st.session_state.feedback_given:
                st.caption("✨ Thanks for the feedback!")

# Determine what to ask/do next
if st.session_state.current_step < len(QUESTIONS):
    current_q = get_current_question()

    # Add the question to history exactly once per step, so it persists
    if st.session_state.question_asked_for_step != st.session_state.current_step:
        add_assistant_message(f"{current_q['prompt']}\n\n_{current_q['help']}_")
        st.session_state.question_asked_for_step = st.session_state.current_step
        st.rerun()

    # Get user response
    if user_input := st.chat_input(f"Your answer..."):
        add_user_message(user_input)

        # Process the answer
        normalized = normalize_response(user_input, current_q["key"])
        st.session_state.answers[current_q["key"]] = normalized
        st.session_state.current_step += 1
        st.rerun()

elif st.session_state.current_step == len(QUESTIONS):
    # All questions answered — make recommendation
    with st.chat_message("assistant"), st.spinner("Finding recipes..."):
        try:
            # Build query from collected answers
            parts = [
                st.session_state.answers.get("cuisine", ""),
                st.session_state.answers.get("dish_type", ""),
                st.session_state.answers.get("diet", ""),
                "I'm a", st.session_state.answers.get("skill_level", ""),
            ]
            extra = st.session_state.answers.get("extra_details", "")
            if extra:
                parts.append(extra)
            query = " ".join(p for p in parts if p).strip()

            if SEARCH_BACKEND == "memory":
                rewritten = rag_memory.rewrite_query(query)
                out = rag_memory.generate_answer(query, filters=rewritten["filters"])
                if not out["recipes"] and rewritten["filters"]:
                    out = rag_memory.generate_answer(query)
                data = {
                    "conversation_id": str(uuid.uuid4()),
                    "answer": out["answer"],
                    "recipes": [r.__dict__ for r in out["recipes"]],
                    "filters": rewritten["filters"],
                }
            else:
                resp = requests.post(
                    f"{API_URL}/recommend",
                    json={"question": query},
                    timeout=120
                )
                resp.raise_for_status()
                data = resp.json()

            # Display recommendation
            st.markdown(data["answer"])

            if data.get("filters"):
                st.caption("Filters: " + ", ".join(
                    f"{k}={v}" for k, v in data["filters"].items()))

            with st.expander("📋 Recommended dishes", expanded=True):
                for r in data.get("recipes", []):
                    recipe_card(r, key_prefix=data.get("conversation_id"))

            # Add to history and show feedback
            add_assistant_message(data["answer"], data)
            st.session_state.current_step = len(QUESTIONS) + 1
            st.rerun()

        except Exception as e:
            st.error(f"❌ API error: {e}")
            if st.button("🔄 Try again"):
                st.session_state.messages = st.session_state.messages[:-1]
                st.rerun()

else:
    # Recommendation already shown, ask if they want another
    st.info("✨ Want another recommendation? Type **reset** to start over.")
    if user_input := st.chat_input("Your answer..."):
        if "reset" in user_input.lower():
            st.session_state.messages = []
            st.session_state.answers = {}
            st.session_state.current_step = 0
            st.session_state.question_asked_for_step = -1
            st.rerun()
        else:
            add_user_message(user_input)
            add_assistant_message("😊 To get another recommendation, type **reset** to start over.")
            st.rerun()
