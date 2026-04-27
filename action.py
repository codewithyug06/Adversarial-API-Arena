from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class TaskAction(BaseModel):
    """Action submitted by the task agent each step."""

    action_type: Literal["query", "final_answer", "accuse"]
    query: str | None = None          # required when action_type="query"
    answer: str | None = None         # required when action_type="final_answer"
    accusation_reason: str | None = None  # optional context for action_type="accuse"

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _validate_fields(self) -> "TaskAction":
        if self.action_type == "query" and not self.query:
            raise ValueError("query field is required when action_type='query'")
        if self.action_type == "final_answer" and self.answer is None:
            raise ValueError("answer field is required when action_type='final_answer'")
        return self


class AdversaryAction(BaseModel):
    """Action submitted by the adversary agent each step."""

    response_mode: Literal["truthful", "manipulated"]
    # Degree of manipulation; 0.0 = barely corrupted, 1.0 = fully corrupted.
    # Validated server-side — out-of-range values are rejected before any state mutation.
    manipulation_degree: float = Field(default=0.0, ge=0.0, le=1.0)
    # If provided, the adversary supplies its own corrupted response string.
    # Otherwise, the server auto-generates corruption from the degree.
    corrupted_response: str | None = None

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _validate_manipulation(self) -> "AdversaryAction":
        if self.response_mode == "truthful" and self.manipulation_degree != 0.0:
            # Silently zero out degree when mode is truthful — avoids subtle bugs
            # where a trainer passes degree=0.5 but mode="truthful" and expects no corruption.
            object.__setattr__(self, "manipulation_degree", 0.0)
        return self
