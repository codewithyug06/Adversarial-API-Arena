from __future__ import annotations
from pydantic import BaseModel


class QueryResponsePair(BaseModel):
    step: int
    query: str
    response: str

    model_config = {"frozen": True}


class TaskObservation(BaseModel):
    """
    What the task agent sees. Never contains ground_truth, manipulation_log,
    suspicion_score, or any adversary-side data. Immutable after construction.
    """

    episode_id: str
    task_description: str
    api_history: list[QueryResponsePair]
    step_count: int
    max_steps: int
    # Reward for the most recent step (0.0 on reset).
    reward: float
    # Terminal flags (gym-style split for TRL/GRPO compatibility).
    terminated: bool   # episode ended due to agent action (answer / accuse)
    truncated: bool    # episode ended due to step limit or server timeout

    model_config = {"frozen": True}

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class AdversaryObservation(BaseModel):
    """
    What the adversary sees. Includes ground truth and suspicion signal.
    Never returned on task-agent endpoints.
    """

    episode_id: str
    task_description: str
    # The task agent's most recent query (None before any query step).
    current_query: str | None
    ground_truth: str
    suspicion_score: float          # EMA suspicion, [0.0, 1.0]
    manipulation_log: list[float]   # manipulation_degree values per manipulated step
    step_count: int
    max_steps: int
    reward: float
    terminated: bool
    truncated: bool

    model_config = {"frozen": True}

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class StepResult(BaseModel):
    """
    Wraps TaskObservation in a gym-style (obs, reward, terminated, truncated, info)
    tuple for TRL / OpenEnv trainer compatibility.
    """

    observation: TaskObservation
    reward: float
    terminated: bool
    truncated: bool
    info: dict = {}

    model_config = {"frozen": True}
