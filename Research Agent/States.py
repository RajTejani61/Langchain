"""
State Definitions and Pydantic Schemas for Research Scoping.

This defines the state objects and structured schemas used forthe research agent.
"""

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing import List, Literal

class AgnentState(MessagesState):
    
    research_question : List[str]
    research_chunk : List[str]
    final_doc : str
    
    evaluation_score : float
    improvement_type : Literal["no_improvement", "rewrite_questions", "rewrite_document"] | None
    improvement_suggestion : str
    retry_question : int 
    retry_document : int

class Research_Questions(BaseModel):
	questions : List[str]

class Evaluate_research(BaseModel):
    relevance_score: float = Field(default=0, description="Score from 0 to 1 indicating how well the research matches the query")
    coverage_score: float = Field(default=0, description="Score from 0 to 1 indicating depth and completeness of the research")
    overall_score: float = Field(default=0, description="Final weighted score from 0 to 1")
    # is_improvement_needed: bool = Field(description="Whether improvement is needed")
    improvement_type : Literal["no_improvement", "rewrite_questions", "rewrite_document"] = Field(description="What action should be taken next")
    improvement_suggestion: str | None = Field(description="How to improve the research query or document if score is low")
