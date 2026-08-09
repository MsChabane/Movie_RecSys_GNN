from typing import List
from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    movie_id: str
    score: float
    title: str
    genres: str


class RecommendationResponse(BaseModel):
    user_id: str
    is_known_user: bool
    top_k: int
    recommendations: List[RecommendationItem]


class RecommendRequest(BaseModel):
    user_id: str = Field(..., min_length=1, example="user_187")
    top_k: int = Field(default=5, ge=1, le=100)
    filter_seen: bool = Field(default=True)
