"""
inspect.py
==========
Look at actual model outputs — not just metrics.

Usage:
  # Run a few episodes with your current model and print everything:
  python train/inspect.py --env-url http://localhost:8000 --n 5

  # Use a saved checkpoint:
  python train/inspect.py --model train/checkpoints/step_0010 --n 5

  # Compare two models side-by-side:
  python train/inspect.py --model-a unsloth/Qwen2.5-0.5B-Instruct \
                           --model-b train/checkpoints/step_0010 \
                           --n 5 --seed 42

What it prints:
  - The task description
  - Each model generation (raw completion + parsed action)
  - The API response received
  - The reward signal
  - Whether the final answer was correct
  - A side-by-side diff when comparing two models
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from action import TaskAction, AdversaryAction
from observation import TaskObservation, StepResult
from train.rollout import (
    EnvironmentRollout,
    format_chat,
    parse_action,
    SYSTEM_PROMPT,
    Trajectory,
)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-url", default="http://localhost:8000")
    p.add_argument("--model", default=None,
                   help="Model ID or checkpoint path (uses greedy text policy if None)")
    p.add_argument("--model-a", default=None, help="First model for comparison")
    p.add_argument("--model-b", default=None, help="Second model for comparison")
    p.add_argument("--n", type=int, default=5, help="Number of episodes to inspect")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--adversary-mode", default="random",
                   choices=["truthful", "random", "fixed"])
    p.add_argument("--no-unsloth", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Greedy text policy (no model — for baseline testing)
# ---------------------------------------------------------------------------

class GreedyTextPolicy:
    """Simple rule-based policy for baseline comparison."""
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def generate(self, prompt: str) -> str:
        # First action: always query the task directly
        if "no queries yet" in prompt:
            # Extract task from prompt
            import re
            m = re.search(r"Task: (.+)", prompt)
            task = m.group(1).strip() if m else "What is the answer?"
            return json.dumps({"action_type": "query", "query": task})
        # Second+ action: submit the last API response as the answer
        import re
        responses = re.findall(r"A: (.+?)(?:\n|$)", prompt)
        if responses:
            return json.dumps({"action_type": "final_answer", "answer": responses[-1].strip()})
        return json.dumps({"action_type": "final_answer", "answer": "unknown"})


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str, args):
    if not args.no_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            return model, tokenizer
        except ImportError:
            pass

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
    return model, tokenizer


def make_generate_fn(model, tokenizer, max_new_tokens: int):
    import torch

    @torch.no_grad()
    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # greedy for deterministic inspection
            pad_token_id=tokenizer.eos_token_id,
        )
        new_ids = ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    return generate


# ---------------------------------------------------------------------------
# Episode runner that captures everything for inspection
# ---------------------------------------------------------------------------

def run_inspect_episode(
    env_url: str,
    generate_fn,
    tokenizer_or_none,
    seed: int,
    adversary_mode: str,
) -> dict:
    """Run one episode and return full inspection data."""
    sess = requests.Session()

    def post(path, **kw):
        return sess.post(f"{env_url}{path}", **kw).json()

    def get(path):
        return sess.get(f"{env_url}{path}").json()

    obs_data = post("/reset", params={"seed": seed})
    obs = TaskObservation(**obs_data)

    episode = {
        "episode_id": obs.episode_id,
        "task_description": obs.task_description,
        "seed": seed,
        "steps": [],
        "total_reward": 0.0,
    }

    import random

    while not obs.done:
        # Stage adversary
        if adversary_mode == "truthful":
            degree, mode = 0.0, "truthful"
        elif adversary_mode == "random":
            degree = random.uniform(0, 1)
            mode = "manipulated" if degree > 0.3 else "truthful"
        else:
            degree, mode = 0.6, "manipulated"

        post("/adversary/step", json={"response_mode": mode, "manipulation_degree": degree})

        # Build prompt
        if tokenizer_or_none:
            prompt = format_chat(
                obs.task_description,
                [h.model_dump() for h in obs.api_history],
                tokenizer_or_none,
            )
        else:
            from train.rollout import build_prompt
            prompt = build_prompt(obs.task_description, [h.model_dump() for h in obs.api_history])

        # Generate
        t0 = time.time()
        completion = generate_fn(prompt)
        gen_time = time.time() - t0

        # Parse action
        action = parse_action(completion)

        # Force answer if too many steps
        if action.action_type == "query" and obs.step_count >= 8:
            last = obs.api_history[-1].response if obs.api_history else "unknown"
            action = TaskAction(action_type="final_answer", answer=last[:300])
            completion = json.dumps({"action_type": "final_answer", "answer": last[:300]})

        # Execute
        result_data = post("/step", json=action.model_dump())
        result = StepResult(
            observation=TaskObservation(**result_data["observation"]),
            reward=result_data["reward"],
            terminated=result_data["terminated"],
            truncated=result_data["truncated"],
            info=result_data.get("info", {}),
        )

        # Get API response from history if it was a query
        api_response = None
        if action.action_type == "query" and result.observation.api_history:
            api_response = result.observation.api_history[-1].response

        episode["steps"].append({
            "step": obs.step_count + 1,
            "action_type": action.action_type,
            "completion": completion,
            "action": action.model_dump(),
            "api_response": api_response,
            "reward": result.reward,
            "adversary_mode": mode,
            "manipulation_degree": degree,
            "gen_time": gen_time,
            "terminated": result.terminated,
            "truncated": result.truncated,
        })
        episode["total_reward"] += result.reward
        obs = result.observation

    # Get ground truth from adversary endpoint (for inspection)
    try:
        adv = get("/adversary/state")
        episode["ground_truth"] = adv.get("ground_truth", "?")
        episode["final_suspicion"] = adv.get("suspicion_score", 0.0)
        episode["manipulation_log"] = adv.get("manipulation_log", [])
    except Exception:
        episode["ground_truth"] = "?"
        episode["final_suspicion"] = 0.0

    sess.close()
    return episode


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"


def print_episode(ep: dict, model_label: str = "model"):
    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"{BOLD}Episode {ep['seed'] + 1} — {ep['episode_id'][:12]}...{RESET}")
    print(f"{CYAN}Task:{RESET} {ep['task_description']}")
    print(f"{CYAN}Ground truth:{RESET} {ep['ground_truth']}")
    print(f"{DIM}Manipulation events: {ep.get('manipulation_log', [])}{RESET}")
    print(f"{DIM}Final suspicion score: {ep.get('final_suspicion', 0):.3f}{RESET}")
    print()

    for s in ep["steps"]:
        reward = s["reward"]
        if reward > 0:
            reward_str = f"{GREEN}+{reward:.2f}{RESET}"
        elif reward == -0.1:
            reward_str = f"{DIM}{reward:.2f}{RESET}"
        else:
            reward_str = f"{RED}{reward:.2f}{RESET}"

        adv_str = ""
        if s["adversary_mode"] == "manipulated":
            adv_str = f" {YELLOW}[adversary: deg={s['manipulation_degree']:.2f}]{RESET}"

        print(f"  Step {s['step']} | {BOLD}{s['action_type']}{RESET} | reward: {reward_str}{adv_str}")
        print(f"    {DIM}Raw completion:{RESET} {s['completion'][:200]!r}")

        if s["action_type"] == "query":
            print(f"    {CYAN}Query:{RESET}      {s['action'].get('query', '')[:120]}")
            if s["api_response"]:
                print(f"    {CYAN}API response:{RESET} {s['api_response'][:120]}")
        elif s["action_type"] == "final_answer":
            ans = s["action"].get("answer", "")
            gt = ep["ground_truth"]
            # Simple color-coded correctness hint
            correct_hint = ""
            if gt and gt.lower() in ans.lower():
                correct_hint = f" {GREEN}(looks correct ✓){RESET}"
            elif gt:
                correct_hint = f" {RED}(may be wrong ✗){RESET}"
            print(f"    {CYAN}Answer:{RESET}     {ans[:120]}{correct_hint}")
        elif s["action_type"] == "accuse":
            print(f"    {CYAN}Reason:{RESET}     {s['action'].get('accusation_reason', '')[:120]}")

        print(f"    {DIM}Gen time: {s['gen_time']:.2f}s{RESET}")

    print(f"\n  {BOLD}Total reward: {ep['total_reward']:.3f}{RESET}")
    print()


# ---------------------------------------------------------------------------
# Comparison mode
# ---------------------------------------------------------------------------

def compare_models(env_url, model_a_fn, model_b_fn, tok_a, tok_b, n, seed_start, adversary_mode):
    print(f"\n{BOLD}MODEL COMPARISON{RESET}: A vs B across {n} episodes\n")
    a_rewards, b_rewards = [], []

    for i in range(n):
        seed = seed_start + i
        print(f"\n{BOLD}--- Episode {i+1} (seed={seed}) ---{RESET}")

        ep_a = run_inspect_episode(env_url, model_a_fn, tok_a, seed, adversary_mode)
        ep_b = run_inspect_episode(env_url, model_b_fn, tok_b, seed, adversary_mode)

        a_rewards.append(ep_a["total_reward"])
        b_rewards.append(ep_b["total_reward"])

        print(f"  Task: {ep_a['task_description']}")
        print(f"  Ground truth: {ep_a['ground_truth']}")

        for j, (sa, sb) in enumerate(zip(ep_a["steps"], ep_b["steps"])):
            print(f"\n  [Step {j+1}]")
            print(f"    {CYAN}Model A{RESET} ({sa['action_type']}): {sa['completion'][:120]!r}  reward={sa['reward']:.2f}")
            print(f"    {CYAN}Model B{RESET} ({sb['action_type']}): {sb['completion'][:120]!r}  reward={sb['reward']:.2f}")

    avg_a = sum(a_rewards) / len(a_rewards)
    avg_b = sum(b_rewards) / len(b_rewards)
    winner = "A" if avg_a > avg_b else ("B" if avg_b > avg_a else "tie")
    print(f"\n{BOLD}Summary:{RESET}")
    print(f"  Model A avg reward: {avg_a:.3f}")
    print(f"  Model B avg reward: {avg_b:.3f}")
    print(f"  Winner: {GREEN if winner != 'tie' else ''}{winner}{RESET}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    # Verify server
    try:
        r = requests.get(f"{args.env_url}/health", timeout=5)
        print(f"Env server: {r.json()}")
    except Exception as e:
        print(f"ERROR: Cannot reach {args.env_url}: {e}")
        sys.exit(1)

    # Comparison mode
    if args.model_a and args.model_b:
        print(f"Loading model A: {args.model_a}")
        model_a, tok_a = load_model(args.model_a, args)
        fn_a = make_generate_fn(model_a, tok_a, args.max_new_tokens)
        print(f"Loading model B: {args.model_b}")
        model_b, tok_b = load_model(args.model_b, args)
        fn_b = make_generate_fn(model_b, tok_b, args.max_new_tokens)
        compare_models(args.env_url, fn_a, fn_b, tok_a, tok_b, args.n, args.seed, args.adversary_mode)
        return

    # Single-model or greedy-text inspection
    if args.model:
        print(f"Loading model: {args.model}")
        model, tokenizer = load_model(args.model, args)
        generate_fn = make_generate_fn(model, tokenizer, args.max_new_tokens)
        label = Path(args.model).name
    else:
        print("No model specified. Using greedy text policy (rule-based baseline).")
        policy = GreedyTextPolicy()
        generate_fn = policy.generate
        tokenizer = None
        label = "greedy-text-baseline"

    print(f"\nRunning {args.n} inspection episodes (seed {args.seed}+)...\n")

    for i in range(args.n):
        ep = run_inspect_episode(
            env_url=args.env_url,
            generate_fn=generate_fn,
            tokenizer_or_none=tokenizer,
            seed=args.seed + i,
            adversary_mode=args.adversary_mode,
        )
        print_episode(ep, model_label=label)

    print(f"\nInspection complete — {args.n} episodes.")


if __name__ == "__main__":
    main()
