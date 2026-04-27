from __future__ import annotations
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ---------------------------------------------------------------------------
# Adjust sys.path so relative imports work when launched from Finale-meta/
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from action import TaskAction, AdversaryAction
from observation import TaskObservation, AdversaryObservation, StepResult
from server.environment import AdversarialArenaEnvironment
from server.rewards import RewardEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("adversarial_api_arena.app")

# ---------------------------------------------------------------------------
# Environment factory — one instance shared per process
# ---------------------------------------------------------------------------
_env: Optional[AdversarialArenaEnvironment] = None


def _get_env() -> AdversarialArenaEnvironment:
    global _env
    if _env is None:
        _env = AdversarialArenaEnvironment(
            max_steps=int(os.getenv("MAX_STEPS", "20")),
            episode_timeout_seconds=float(os.getenv("EPISODE_TIMEOUT", "300")),
            suspicion_threshold=float(os.getenv("SUSPICION_THRESHOLD", "0.7")),
            log_dir=Path(os.getenv("LOG_DIR", "outputs/logs")),
            seed=int(v) if (v := os.getenv("SEED")) else None,
        )
    return _env


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_env()
    logger.info("Adversarial API Arena server ready")
    yield
    logger.info("Adversarial API Arena server shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Adversarial API Arena",
    description=(
        "Multi-agent adversarial RL environment. "
        "Task agent completes factual tasks by querying a mock API. "
        "Adversary earns reward by manipulating responses while staying undetected."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Liveness probe."""
    env = _get_env()
    return {
        "status": "ok",
        "episode_id": env.current_episode_id,
        "version": "0.1.0",
    }


# ===========================================================================
# Task-agent endpoints  (OpenEnv standard interface)
# ===========================================================================

@app.post("/reset", response_model=TaskObservation)
def reset(
    seed: Optional[int] = Query(default=None, description="Optional RNG seed for reproducibility"),
    episode_id: Optional[str] = Query(default=None, description="Optional episode ID"),
) -> TaskObservation:
    """
    Start a new episode. Returns the initial TaskObservation.
    Samples a task from the bank, resets all state.
    Ground truth is NEVER included in the response.
    """
    env = _get_env()
    return env.reset(seed=seed, episode_id=episode_id)


@app.post("/step", response_model=StepResult)
def step(action: TaskAction) -> StepResult:
    """
    Execute one task-agent action.

    action_type options:
      - "query"        : send a natural-language query to the mock API
      - "final_answer" : submit your answer to the task (terminates episode)
      - "accuse"       : flag the API as manipulated (terminates episode)

    Returns StepResult with observation, reward, terminated, truncated, info.
    The observation NEVER contains ground_truth or adversary metadata.
    """
    env = _get_env()
    return env.step(action)


@app.get("/state", response_model=TaskObservation)
def get_state() -> TaskObservation:
    """Non-destructive read of the current task-agent observation."""
    env = _get_env()
    s = env._require_state()
    with env._lock:
        return s.to_task_observation(reward=0.0)


# ===========================================================================
# Adversary endpoints  (separate path — task agent cannot access these)
# ===========================================================================

@app.post("/adversary/step", response_model=AdversaryObservation)
def adversary_step(action: AdversaryAction) -> AdversaryObservation:
    """
    Adversary stages its action for the NEXT query the task agent makes.

    Call this BEFORE the task agent calls POST /step with action_type="query".
    If not called, the mock API defaults to truthful for that query step.

    manipulation_degree is validated [0.0, 1.0]. Out-of-range values are rejected.
    Adversary earns no per-step reward; reward is computed at episode termination.
    """
    env = _get_env()
    return env.adversary_step(action)


@app.get("/adversary/state", response_model=AdversaryObservation)
def adversary_state() -> AdversaryObservation:
    """
    Non-destructive read of the adversary observation.
    Includes ground_truth, suspicion_score, and manipulation_log.
    """
    env = _get_env()
    return env.adversary_observe()


@app.get("/adversary/reward")
def adversary_terminal_reward() -> dict:
    """
    Compute and return the adversary's terminal reward for the current (ended) episode.
    Must be called after the task agent's terminal step.
    """
    env = _get_env()
    s = env._require_state()
    if not s.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is not done yet. Adversary reward is only available at termination.",
        )
    reward = env.compute_adversary_terminal_reward()
    return {
        "episode_id": s.episode_id,
        "adversary_reward": reward,
        "suspicion_score": s.suspicion_score,
        "manipulation_count": len(s.manipulation_log),
        "termination_reason": s.termination_reason,
    }


# ===========================================================================
# Inspector endpoint  (post-hoc analysis — includes all adversary metadata)
# ===========================================================================

@app.get("/episode/log")
def episode_log() -> list[dict]:
    """
    Returns the full step-by-step log for the current episode.
    Includes adversary mode, manipulation_degree, and suspicion_score per step.
    Intended for human inspection and reward-hacking detection.
    The log is read-only and append-only during an episode.
    """
    env = _get_env()
    return env.get_episode_log()


@app.get("/episode/info")
def episode_info() -> dict:
    """Summary of the current episode (no ground truth exposed to task agent path)."""
    env = _get_env()
    s = env._require_state()
    with env._lock:
        return {
            "episode_id": s.episode_id,
            "step_count": s.step_count,
            "max_steps": s.max_steps,
            "done": s.done,
            "terminated": s.terminated,
            "truncated": s.truncated,
            "termination_reason": s.termination_reason,
            "suspicion_score": s.suspicion_score,
            "manipulation_count": len(s.manipulation_log),
        }


# ---------------------------------------------------------------------------
# Entry point for local development
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "server.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=bool(os.getenv("DEV", "")),
        workers=int(os.getenv("WORKERS", "1")),
    )
