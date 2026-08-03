# AI Chef — Project Plan

A personal AI cooking assistant that recommends dishes
based on **cuisine**, **dish type**, and **culinary skill level**, then walks
the user through the recipe. Built and maintained as a long-term personal
project.

> Status: **Planning** — this document is the single source of truth for scope
> and architecture. Update it as decisions change.

---

## 1. Problem Statement

People often don't know what to cook. Generic recipe sites overwhelm users with
thousands of options and don't account for:

- what cuisine they're craving,
- what kind of dish they want (one-pot, quick snack, dessert, ...),
- how skilled they actually are in the kitchen.

**AI Chef** solves this with a short conversational flow:

1. *"What do you want to eat today?"* → cuisine (Indian, Mexican, Italian, ...)
2. *"What kind of dish?"* → dish type (one-pot, 30-min, dessert, ...)
3. *"What protein do you eat?"* → diet (vegan / vegetarian / chicken / pork /
   beef / I eat everything)
4. *"What's your cooking level?"* → skill (beginner / can cook / master chef)
5. AI Chef recommends matching dishes with full recipes, plus **similar dishes**
   the user might also like.

Under the hood it's a RAG application: a recipe knowledge base + hybrid
retrieval + an LLM that presents recipes matched to the user's constraints.

---

## 2. Tech Stack

Fully free-tier friendly. Only two hosted externals: **Groq** (LLM) and
**Vercel** (deployment).

| Layer | Choice | Why |
|---|---|---|
| LLM | **Groq** (`llama-3.1-8b-instant`, judge: `llama-3.3-70b-versatile`) | Free tier, OpenAI-compatible API |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Free, runs locally |
| Knowledge base | **PostgreSQL + pgvector** | Vector search + full-text search + app/analytics DB in one container |
| Retrieval | Hybrid: pgvector (semantic) + Postgres FTS (keyword), optional cross-encoder re-rank | Best of keyword + semantic matching |
| API | **FastAPI** | Typed, async, auto-docs at `/docs` |
| UI | **Streamlit** (chat-style flow + 👍/👎 feedback) | Fast to build, easy to demo |
| Monitoring | PostgreSQL tables + **Grafana** (provisioned dashboards) | Track quality, cost, usage over time |
| Containerization | Docker Compose (api, streamlit, postgres, grafana) | One-command local run |
| Ingestion | Python script → optional Prefect flow | Simple first, automate later |
| Evaluation | Jupyter notebooks (Hit Rate, MRR, LLM-as-a-judge) | Data-driven retrieval/model choices |
| Deployment | FastAPI → **Vercel**, Streamlit → Streamlit Cloud, DB → Neon free tier | Free hosting for the live demo |
| Package manager | `uv` | Fast, reproducible (`uv.lock`) |

---

## 3. Architecture

```
┌────────────────────────────────────────────────────────────────┐
│ Streamlit UI (chat)                                            │
│  cuisine → dish type → skill level  →  recipe cards + similar  │
│  👍/👎 feedback buttons                                        │
└──────────────┬─────────────────────────────────┬───────────────┘
               │ POST /recommend                  │ POST /feedback
               ▼                                  ▼
┌────────────────────────────────────────────────────────────────┐
│ FastAPI                                                        │
│  1. query rewriting / filter extraction (Groq, structured out) │
│  2. hybrid search: pgvector ⊕ full-text, filters, re-rank      │
│  3. build prompt → Groq LLM answer                             │
│  4. inline LLM-as-judge relevance + token/cost tracking        │
│  5. log conversation → Postgres                                │
└──────────────┬─────────────────────────────────────────────────┘
               ▼
┌────────────────────────────────────────────────────────────────┐
│ PostgreSQL + pgvector                                          │
│  ├── recipes        (knowledge base: text + embedding + meta)  │
│  ├── conversations  (question, answer, relevance, tokens, ms)  │
│  └── feedback       (conversation_id, +1/-1)                   │
└──────────────┬─────────────────────────────────────────────────┘
               ▼
┌────────────────────────────────────────────────────────────────┐
│ Grafana (provisioned datasource + dashboard, ≥5 charts)        │
└────────────────────────────────────────────────────────────────┘
```

Everything above runs from a single `docker compose up`.

---

## 4. Dataset

**Real Kaggle recipe dataset** (decided). Candidates, in order of preference:

1. **Food.com — RAW_recipes** (~230k recipes, Hugging Face mirror
   `Cassiedu66/ai-blessed_raw_recipes`; Kaggle needs auth)
   - Fields: name, minutes, tags, calories, steps, ingredients
   - Tags already contain cuisines (`indian`, `mexican`), dish types
     (`one-dish-meal`, `30-minutes-or-less`), diet (`vegan`, `vegetarian`,
     `meat`) and difficulty hints (`beginner-cook`, `easy`) → derive
     `cuisine`, `dish_type`, `skill_level`, `proteins` (chicken/pork/beef
     detected from ingredients; vegan/vegetarian from tags)
2. Indian Food datasets (~6–9k) — fallback / supplement for cuisine coverage

Plan: sample a manageable subset (**10k recipes**) covering several cuisines,
derive structured fields, embed `name + ingredients + steps`.

**Ground truth for retrieval evaluation**: generate 1–2 user questions per
sampled recipe with Groq.

---

## 5. Feature Checklist

Quality bar for the project — every feature is evaluated before it ships.

| Area | Implementation |
|---|---|
| Documentation | README: problem, users, flow diagram, examples, screenshots |
| RAG flow | pgvector knowledge base + Groq LLM, both in the flow |
| Retrieval evaluation | Compare **multiple** approaches: text-only vs vector-only vs hybrid vs hybrid+re-rank → Hit Rate + MRR, best one used in prod |
| LLM evaluation | LLM-as-a-judge (RELEVANT/PARTLY/NON_RELEVANT) comparing **2 models** (llama-3.1-8b vs llama-3.3-70b) × 2 prompt variants |
| Interface | FastAPI REST API **and** Streamlit chat UI |
| Ingestion pipeline | `pipeline/ingest.py` (download→clean→derive→embed→load); upgrade to Prefect flow later |
| Monitoring | User feedback (👍/👎) **and** Grafana dashboard: requests over time, relevance distribution, feedback ratio, response time p50/p95, token usage & cost, top cuisines |
| Containerization | One `docker-compose.yaml`: api + streamlit + postgres(pgvector) + grafana (auto-provisioned) |
| Reproducibility | `uv.lock`, `.env.example`, download script, step-by-step README |
| Hybrid search | pgvector + FTS with reciprocal rank fusion |
| Re-ranking | Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-ranks top-20 → top-5 |
| Query rewriting | Groq rewrites conversational input into an optimized search query + extracted filters |
| Cloud deployment | FastAPI on Vercel, Streamlit on Streamlit Cloud, DB on Neon |

---

## 6. Repository Structure

```
ai-chef/
├── PLAN.md                     # ← this file
├── README.md                   # problem, setup, usage, screenshots (final)
├── pyproject.toml / uv.lock
├── .env.example
├── docker-compose.yaml
├── Dockerfile                  # FastAPI image
├── Dockerfile.streamlit        # Streamlit image
├── api/
│   ├── main.py                 # FastAPI: /recommend, /feedback, /health
│   ├── rag.py                  # query rewrite → search → prompt → Groq → judge
│   ├── search.py               # text / vector / hybrid / re-rank
│   ├── db.py                   # conversations + feedback logging
│   └── schemas.py              # Pydantic models
├── streamlit_app/
│   └── app.py                  # chat flow + recipe cards + feedback
├── pipeline/
│   ├── ingest.py               # Kaggle → clean → derive fields → embed → pgvector
│   └── generate_ground_truth.py
├── notebooks/
│   ├── 01-data-exploration.ipynb
│   ├── 02-retrieval-evaluation.ipynb   # hit rate / MRR across approaches
│   └── 03-rag-evaluation.ipynb         # LLM-as-judge, model comparison
├── grafana/
│   ├── provisioning/           # datasource + dashboard auto-provisioning
│   └── dashboards/dashboard.json
├── scripts/
│   └── init_db.sql             # pgvector extension + tables
├── data/                       # gitignored, download via script
└── tests/
```

---

## 7. Build Roadmap

- [ ] **Phase 0 — Setup**: uv project, `.env` (GROQ_API_KEY), repo skeleton
- [ ] **Phase 1 — Data**: download dataset, explore in notebook, derive
      `cuisine` / `dish_type` / `skill_level`, pick subset
- [ ] **Phase 2 — Ingestion**: `ingest.py` → Postgres+pgvector; verify counts
- [ ] **Phase 3 — Retrieval + eval**: ground truth via Groq; Hit Rate/MRR for
      text vs vector vs hybrid (+re-rank); pick winner
- [ ] **Phase 4 — RAG + eval**: prompt templates, Groq answers, LLM-as-judge,
      8b vs 70b comparison; pick winner
- [ ] **Phase 5 — FastAPI**: `/recommend`, `/feedback`, `/health`; logging
- [ ] **Phase 6 — Streamlit**: conversational flow, recipe cards, similar
      dishes, feedback buttons
- [ ] **Phase 7 — Monitoring**: inline judge, Grafana dashboard (≥5 charts),
      auto-provisioning
- [ ] **Phase 8 — Docker Compose**: all 4 services, one-command start
- [ ] **Phase 9 — Docs**: full README (problem, architecture, setup, usage,
      screenshots, evaluation summary)
- [ ] **Phase 10 — Bonus (optional)**: Prefect ingestion flow; Vercel +
      Streamlit Cloud + Neon deployment

---

## 8. Environment Variables

```env
GROQ_API_KEY=...            # console.groq.com (free)
POSTGRES_HOST=postgres
POSTGRES_DB=ai_chef
POSTGRES_USER=chef
POSTGRES_PASSWORD=...
EMBEDDING_MODEL=all-MiniLM-L6-v2
GROQ_MODEL=llama-3.1-8b-instant
GROQ_JUDGE_MODEL=llama-3.3-70b-versatile
```

## 9. Open Questions / Decisions Log

| Date | Decision | Choice |
|---|---|---|
| 2026-08-03 | Dataset | Food.com RAW_recipes (231k) via HF mirror `Cassiedu66/ai-blessed_raw_recipes` (Kaggle needs auth; other HF mirrors were broken) |
| 2026-08-03 | Sample | 10k recipes, sqrt-proportional stratified across 18 cuisines (seed 42) |
| 2026-08-03 | Filters | cuisine + dish_type + **protein/diet** (vegan/vegetarian/chicken/pork/beef/everything) + skill_level — all derived in `pipeline/prepare_data.py` |
| 2026-08-03 | Diet logic | tags win; else infer from ingredients (no meat → vegetarian/vegan via dairy-egg check); ~0.14% conflicts (mock meats), acceptable |
| 2026-08-03 | Interface | FastAPI + Streamlit (no Next.js) |
| 2026-08-03 | Database | PostgreSQL + pgvector (no Supabase) |
| 2026-08-03 | LLM | Groq free tier |
| 2026-08-03 | Deployment | Vercel (API) — bonus phase only |
| — | Ingestion: script vs Prefect? | Start with script, decide at Phase 10 |
