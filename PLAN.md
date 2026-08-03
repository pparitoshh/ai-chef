# AI Chef — Capstone Project Plan

LLM Zoomcamp capstone project. An AI cooking assistant that recommends dishes
based on **cuisine**, **dish type**, and **culinary skill level**, then walks
the user through the recipe.

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
3. *"What's your cooking level?"* → skill (beginner / can cook / master chef)
4. AI Chef recommends matching dishes with full recipes, plus **similar dishes**
   the user might also like.

Under the hood it's a RAG application: a recipe knowledge base + hybrid
retrieval + an LLM that presents recipes matched to the user's constraints.

---

## 2. Tech Stack

Course stack (mirrors llm-zoomcamp modules 2–7), with only two externals:
**Groq** (LLM) and **Vercel** (deployment).

| Layer | Choice | Why |
|---|---|---|
| LLM | **Groq** (`llama-3.1-8b-instant`, judge: `llama-3.3-70b-versatile`) | Free tier, OpenAI-compatible API |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Free, local — same as course module 2 |
| Knowledge base | **PostgreSQL + pgvector** | Vector search + full-text search + monitoring DB in one container (course module 2) |
| Retrieval | Hybrid: pgvector (semantic) + Postgres FTS (keyword), optional cross-encoder re-rank | Hybrid search = best-practice point |
| API | **FastAPI** | Interface (2 pts) |
| UI | **Streamlit** (chat-style flow + 👍/👎 feedback) | Interface, easy demo video |
| Monitoring | PostgreSQL tables + **Grafana** (≥5 charts, provisioned) | Course module 5 |
| Containerization | Docker Compose (api, streamlit, postgres, grafana) | 2 pts |
| Ingestion | Python script → optional Prefect flow | 1–2 pts |
| Evaluation | Jupyter notebooks (Hit Rate, MRR, LLM-as-a-judge) | Course module 4 |
| Deployment (bonus) | FastAPI → **Vercel**, Streamlit → Streamlit Cloud, DB → Neon free tier | +2 bonus pts |
| Package manager | `uv` | Course standard |

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

1. **Food.com — Recipes and Reviews** (~230k recipes, via `kagglehub`)
   - Fields: name, ingredients, steps, minutes, tags, description
   - Tags already contain cuisines (`indian`, `mexican`), dish types
     (`one-pot`, `30-minutes-or-less`), and difficulty hints → derive
     `cuisine`, `dish_type`, `skill_level` (from #steps + minutes + tags)
2. Indian Food datasets (~6–9k) — fallback / supplement for cuisine coverage

Plan: sample a manageable subset (~10–20k recipes) covering several cuisines,
derive structured fields, embed `name + description + ingredients`.

**Ground truth for retrieval evaluation**: generate 1–2 user questions per
sampled recipe with Groq (same approach as course module 4 / project example).

---

## 5. Components → Evaluation Criteria Map

| Criterion (max pts) | Our implementation |
|---|---|
| Problem description (2) | README: problem, users, flow diagram, examples |
| Retrieval flow (2) | pgvector knowledge base + Groq LLM, both in the flow |
| Retrieval evaluation (2) | Compare **multiple** approaches: text-only vs vector-only vs hybrid vs hybrid+re-rank → Hit Rate + MRR, best one used in prod |
| LLM evaluation (2) | LLM-as-a-judge (RELEVANT/PARTLY/NON_RELEVANT) comparing **2 models** (llama-3.1-8b vs llama-3.3-70b) × 2 prompt variants |
| Interface (2) | FastAPI REST API **and** Streamlit chat UI |
| Ingestion pipeline (1→2) | `pipeline/ingest.py` (download→clean→derive→embed→load); upgrade to Prefect flow for 2 pts |
| Monitoring (2) | User feedback (👍/👎) **and** Grafana dashboard ≥5 charts: requests over time, relevance distribution, feedback ratio, response time p50/p95, token usage & cost, top cuisines |
| Containerization (2) | One `docker-compose.yaml`: api + streamlit + postgres(pgvector) + grafana (auto-provisioned) |
| Reproducibility (2) | `uv.lock`, `.env.example`, seed data / download script, step-by-step README |
| Hybrid search (+1) | pgvector + FTS with reciprocal rank fusion |
| Re-ranking (+1) | Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-ranks top-20 → top-5 |
| Query rewriting (+1) | Groq rewrites conversational input into an optimized search query + extracted filters |
| Cloud deployment (+2, bonus) | FastAPI on Vercel, Streamlit on Streamlit Cloud, DB on Neon |

**Target: 18 base + 3 best-practice (+2 deployment bonus).**

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
| 2026-08-03 | Dataset | Real Kaggle dataset (Food.com preferred) |
| 2026-08-03 | Interface | FastAPI + Streamlit (no Next.js) |
| 2026-08-03 | Database | PostgreSQL + pgvector (no Supabase) |
| 2026-08-03 | LLM | Groq free tier |
| 2026-08-03 | Deployment | Vercel (API) — bonus phase only |
| — | Ingestion: script vs Prefect? | Start with script, decide at Phase 10 |
