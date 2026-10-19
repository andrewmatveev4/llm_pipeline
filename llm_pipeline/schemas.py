from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["support", "feedback", "complaint", "sales", "general_question", "other"]
Sentiment = Literal["positive", "neutral", "negative"]


class LLMOutput(BaseModel):
    summary: str = Field(min_length=1)
    category: Category
    intent: str = Field(min_length=1, max_length=80)
    sentiment: Sentiment
    key_points: list[str] = Field(min_length=3, max_length=3)
    final_answer: str = Field(min_length=1)
