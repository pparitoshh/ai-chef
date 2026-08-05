"""Pydantic request/response models for the FastAPI app."""

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    question: str = Field(..., examples=["I want a quick vegetarian indian dinner"])
    k: int = Field(5, ge=1, le=10)


class RecipeOut(BaseModel):
    id: str
    name: str
    cuisine: str
    dish_type: str
    diet: str
    skill_level: str
    minutes: int | None


class RecipeDetail(RecipeOut):
    ingredients: list[str]
    steps: list[str]
    calories: float | None


class RecommendResponse(BaseModel):
    conversation_id: str
    answer: str
    recipes: list[RecipeOut]
    rewritten_query: str
    filters: dict
    relevance: str
    total_tokens: int | None
    response_time_ms: float


class FeedbackRequest(BaseModel):
    conversation_id: str
    feedback: int = Field(..., ge=-1, le=1)  # +1 thumbs up, -1 thumbs down


class FeedbackResponse(BaseModel):
    status: str = "ok"
