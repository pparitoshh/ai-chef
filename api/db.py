"""Conversation + feedback logging to Postgres (monitoring data for Grafana)."""

import json
import uuid

from api.search import get_connection


def log_conversation(question: str, answer: str, model_used: str,
                     response_time_s: float, relevance: str | None,
                     prompt_tokens: int | None, completion_tokens: int | None,
                     total_tokens: int | None, filters: dict | None) -> str:
    conv_id = str(uuid.uuid4())
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO conversations
                (id, question, answer, model_used, response_time, relevance,
                 prompt_tokens, completion_tokens, total_tokens, filters)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (conv_id, question, answer, model_used, response_time_s, relevance,
             prompt_tokens, completion_tokens, total_tokens,
             json.dumps(filters or {})),
        )
    return conv_id


def log_feedback(conversation_id: str, feedback: int) -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO feedback (conversation_id, feedback) VALUES (%s, %s)",
            (conversation_id, feedback),
        )
