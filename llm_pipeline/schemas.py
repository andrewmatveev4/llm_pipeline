from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["support", "feedback", "complaint", "sales", "general_question", "other"]
Sentiment = Literal["positive", "neutral", "negative"]
Urgency = Literal["low", "medium", "high"]


class Classification(BaseModel):
    summary: str = Field(min_length=1)
    category: Category
    intent: str = Field(min_length=1, max_length=80)
    sentiment: Sentiment
    key_points: list[str] = Field(min_length=3, max_length=3)


class PipelineResult(Classification):
    route: Category
    final_answer: str = Field(min_length=1)


class Meaning(BaseModel):
    summary: str = Field(min_length=1)
    key_points: list[str] = Field(min_length=3, max_length=3)
    language: str = Field(min_length=2, max_length=30)


class RequestType(BaseModel):
    category: Category
    intent: str = Field(min_length=1, max_length=80)
    sentiment: Sentiment


class Details(BaseModel):
    product: str = Field(min_length=1)
    problem: str = Field(min_length=1)
    requested_action: str = Field(min_length=1)
    urgency: Urgency
    facts: list[str]
    missing_info: list[str]


class SelfCheck(BaseModel):
    contradictions: list[str]
    lost_details: list[str]


class ChainState(BaseModel):
    text: str
    meaning: Meaning | None = None
    request: RequestType | None = None
    details: Details | None = None
    answer: str | None = None
    check: SelfCheck | None = None
    passed: bool | None = None
    failed_step: str | None = None
    error: str | None = None
    fallback_reason: str | None = None
