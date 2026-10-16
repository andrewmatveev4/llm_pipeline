from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["question", "complaint", "request", "news", "other"]
Sentiment = Literal["positive", "neutral", "negative"]


class LLMOutput(BaseModel):
    summary: str = Field(min_length=1)
    category: Category
    sentiment: Sentiment
    key_points: list[str] = Field(min_length=3, max_length=3)
    final_answer: str = Field(min_length=1)
