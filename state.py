from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from observation import (
    TaskObservation,
    AdversaryObservation,
    QueryResponsePair,
    StepResult,
)


@dataclass
class ManipulationRecord:
    step: int
    degree: float
    response_mode: str
    corrupted_response: Optional[str]


@dataclass
class EpisodeState:
    """
    Server-internal episode state. NEVER serialized directly to any agent.
    Ground truth and manipulation log are stripped before building observations.

    All fields mutated only by environment.py — never by app.py or external code.
    """

    episode_id: str
    task_description: str
    ground_truth: str
    correctness_mode: str               # "exact" | "fuzzy" | "semantic"
    max_steps: int = 20
    step_count: int = 0
    api_history: list[dict] = field(default_factory=list)
    suspicion_score: float = 0.0
    manipulation_log: list[ManipulationRecord] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    termination_reason: Optional[str] = None
    # Staged by adversary_step(), consumed once by the next _handle_query().
    _pending_adversary_action: Optional[dict] = field(default=None, repr=False)
    # Latest query from task agent, exposed to adversary via GET /adversary/state.
    _latest_query: Optional[str] = field(default=None, repr=False)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    # ------------------------------------------------------------------
    # Observation builders — explicit whitelists, no dynamic field copy
    # ------------------------------------------------------------------

    def to_task_observation(self, reward: float) -> TaskObservation:
        """Returns a task-agent-safe view. All adversary fields excluded."""
        return TaskObservation(
            episode_id=self.episode_id,
            task_description=self.task_description,
            api_history=[
                QueryResponsePair(
                    step=p["step"],
                    query=p["query"],
                    response=p["response"],
                )
                for p in self.api_history
            ],
            step_count=self.step_count,
            max_steps=self.max_steps,
            reward=reward,
            terminated=self.terminated,
            truncated=self.truncated,
        )

    def to_step_result(self, reward: float) -> StepResult:
        obs = self.to_task_observation(reward)
        return StepResult(
            observation=obs,
            reward=reward,
            terminated=self.terminated,
            truncated=self.truncated,
            info={
                "step_count": self.step_count,
                "termination_reason": self.termination_reason,
            },
        )

    def to_adversary_observation(
        self, reward: float, current_query: Optional[str] = None
    ) -> AdversaryObservation:
        return AdversaryObservation(
            episode_id=self.episode_id,
            task_description=self.task_description,
            current_query=current_query or self._latest_query,
            ground_truth=self.ground_truth,
            suspicion_score=self.suspicion_score,
            manipulation_log=[r.degree for r in self.manipulation_log],
            step_count=self.step_count,
            max_steps=self.max_steps,
            reward=reward,
            terminated=self.terminated,
            truncated=self.truncated,
        )
