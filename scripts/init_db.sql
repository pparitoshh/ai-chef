-- AI Chef database schema (knowledge base + monitoring).
-- Runs once on first postgres container start (docker-entrypoint-initdb.d).

CREATE EXTENSION IF NOT EXISTS vector;

-- Knowledge base -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS recipes (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    cuisine     TEXT NOT NULL,
    dish_type   TEXT NOT NULL,
    diet        TEXT NOT NULL,
    proteins    JSONB NOT NULL DEFAULT '[]',
    skill_level TEXT NOT NULL,
    minutes     INTEGER,
    calories    FLOAT,
    n_steps     INTEGER,
    ingredients JSONB NOT NULL,
    steps       JSONB NOT NULL,
    tags        JSONB NOT NULL,
    -- denormalized plain text used for embedding + full-text search
    ingredients_text TEXT NOT NULL DEFAULT '',
    embedding   vector(384),
    -- full-text search column for hybrid retrieval (name + ingredients)
    fts tsvector GENERATED ALWAYS AS (
        to_tsvector(
            'english',
            name || ' ' || cuisine || ' ' || dish_type || ' ' || ingredients_text
        )
    ) STORED
);

CREATE INDEX IF NOT EXISTS recipes_fts_idx ON recipes USING gin (fts);
CREATE INDEX IF NOT EXISTS recipes_cuisine_idx ON recipes (cuisine);
CREATE INDEX IF NOT EXISTS recipes_diet_idx ON recipes (diet);
CREATE INDEX IF NOT EXISTS recipes_skill_idx ON recipes (skill_level);
-- HNSW index for vector search (created after data load is also fine)
CREATE INDEX IF NOT EXISTS recipes_embedding_idx
    ON recipes USING hnsw (embedding vector_cosine_ops);

-- Monitoring ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS conversations (
    id                    TEXT PRIMARY KEY,
    question              TEXT NOT NULL,
    answer                TEXT NOT NULL,
    model_used            TEXT NOT NULL,
    response_time         FLOAT NOT NULL,
    relevance             TEXT,
    relevance_explanation TEXT,
    prompt_tokens         INTEGER,
    completion_tokens     INTEGER,
    total_tokens          INTEGER,
    filters               JSONB,
    timestamp             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback (
    id              SERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    feedback        INTEGER NOT NULL CHECK (feedback IN (1, -1)),
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT now()
);
