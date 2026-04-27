"""
train_grpo.py
=============
Train the task agent LLM with GRPO using Unsloth + TRL.

SETUP (run once):
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install trl>=0.12.0 datasets>=2.18.0 peft>=0.12.0 accelerate>=0.28.0 rich

START ENV SERVER FIRST:
  cd Finale-meta && uvicorn server.app:app --port 8000

RUN TRAINING:
  cd Finale-meta && python train/train_grpo.py \
    --env-url http://localhost:8000 \
    --model unsloth/Qwen2.5-0.5B-Instruct \
    --episodes-per-step 8 \
    --steps 20 \
    --adversary-mode random

What this does:
  - Loads a small quantized model (0.5B default)
  - Runs environment episodes to collect (prompt, completion, reward) pairs
  - Uses GRPO to update the model toward higher reward
  - Prints actual model completions at each checkpoint so you can inspect quality
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.rollout import EnvironmentRollout, build_prompt, SYSTEM_PROMPT, parse_action
from train.curriculum import CurriculumManager, CurriculumRollout, diagnose_zero_reward
from train.hack_inspector import HackInspector, print_report

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-url", default="http://localhost:8000")
    p.add_argument("--model", default="unsloth/Qwen2.5-0.5B-Instruct",
                   help="HuggingFace model ID (Unsloth quantized preferred)")
    p.add_argument("--output-dir", default="train/checkpoints")
    p.add_argument("--episodes-per-step", type=int, default=8,
                   help="Rollout episodes per GRPO update step")
    p.add_argument("--steps", type=int, default=20,
                   help="Total training steps (each step = rollout + update)")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--adversary-mode", default="random",
                   choices=["truthful", "random", "fixed"],
                   help="Adversary behaviour during training rollouts (ignored when --curriculum)")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inspect-every", type=int, default=5,
                   help="Print sample completions every N steps")
    p.add_argument("--hack-check-every", type=int, default=10,
                   help="Run hack inspection report every N steps (0 to disable)")
    p.add_argument("--curriculum", action="store_true",
                   help="Enable curriculum learning (auto-adjusts difficulty based on reward)")
    p.add_argument("--start-level", type=int, default=0,
                   help="Starting curriculum level (0=warmup, 3=hard)")
    p.add_argument("--no-unsloth", action="store_true",
                   help="Fall back to plain HuggingFace (no Unsloth) — slower")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(args):
    if not args.no_unsloth:
        try:
            from unsloth import FastLanguageModel, is_bfloat16_supported
            print(f"Loading {args.model} with Unsloth (4-bit)...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=args.lora_r,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_alpha=args.lora_r,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=args.seed,
            )
            return model, tokenizer, "unsloth"
        except ImportError:
            print("Unsloth not installed. Falling back to plain HuggingFace.")

    # Plain HuggingFace fallback
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType

    print(f"Loading {args.model} with HuggingFace (4-bit)...")
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer, "hf"


# ---------------------------------------------------------------------------
# Inference function (used during rollout)
# ---------------------------------------------------------------------------
def make_generate_fn(model, tokenizer, max_new_tokens: int, backend: str):
    import torch

    if backend == "unsloth":
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)

    @torch.no_grad()
    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    return generate


def reenable_training(model, backend: str):
    """Switch model back to training mode after inference."""
    if backend == "unsloth":
        from unsloth import FastLanguageModel
        FastLanguageModel.for_training(model)


# ---------------------------------------------------------------------------
# Reward functions for GRPO
# These receive (prompts, completions) and return list[float].
# Multiple independent functions — reduces reward hacking.
# ---------------------------------------------------------------------------

def reward_format(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """Reward valid JSON action format. Independent of correctness."""
    rewards = []
    for c in completions:
        try:
            data = json.loads(c.strip())
            if "action_type" in data:
                rewards.append(0.2)
            else:
                rewards.append(-0.1)
        except Exception:
            rewards.append(-0.2)
    return rewards


def reward_action_type(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """Reward choosing a specific action type when appropriate."""
    rewards = []
    for prompt, c in zip(prompts, completions):
        try:
            data = json.loads(c.strip())
            atype = data.get("action_type", "")
            # Small positive for any valid action type, extra for final_answer
            if atype == "final_answer" and data.get("answer", "").strip():
                rewards.append(0.1)
            elif atype == "query" and data.get("query", "").strip():
                rewards.append(0.05)
            elif atype == "accuse":
                rewards.append(0.0)  # neutral — we don't know if correct yet
            else:
                rewards.append(-0.1)
        except Exception:
            rewards.append(-0.1)
    return rewards


def make_env_reward_fn(rollout: EnvironmentRollout, tokenizer, max_new_tokens: int):
    """
    Returns a reward function that runs one environment episode per (prompt, completion).

    This is the MAIN reward signal: execute the completion in the env, let the env run
    to completion with a greedy policy, return the episode reward.

    To avoid running a full multi-step episode per training sample (which would be slow),
    we only execute the FIRST action from the completion and assign:
      - For query: the per-step reward (-0.1 time penalty)
      - For final_answer: the terminal reward (correctness check)
      - For accuse: the accusation reward

    The environment already computed these server-side, so we trust the env's reward.
    """
    import requests

    def reward_env(
        prompts: list[str],
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            try:
                action = parse_action(completion)
                # Reset and run one step in the env
                rollout._post("/reset")
                result_data = rollout._post("/step", json=action.model_dump())
                reward = result_data.get("reward", 0.0)
                rewards.append(float(reward))
            except Exception as e:
                print(f"  [env_reward] error: {e}")
                rewards.append(-0.5)
        return rewards

    return reward_env


# ---------------------------------------------------------------------------
# Dataset generation from environment rollouts
# ---------------------------------------------------------------------------

def collect_rollout_dataset(
    rollout: EnvironmentRollout,
    generate_fn,
    tokenizer,
    n_episodes: int,
    seed_offset: int = 0,
):
    """Run episodes and collect (prompt, completion, reward) as a HF Dataset."""
    from datasets import Dataset

    rows = []
    trajectories = rollout.run_batch(
        model_generate_fn=generate_fn,
        tokenizer=tokenizer,
        n_episodes=n_episodes,
        seed_offset=seed_offset,
    )
    for traj in trajectories:
        for step in traj.steps:
            rows.append({
                "prompt": step.prompt,
                "completion": step.completion,
                "reward": step.reward,
                "episode_id": traj.episode_id,
            })

    return Dataset.from_list(rows), trajectories


# ---------------------------------------------------------------------------
# Output inspection — the "look at outputs, not just metrics" part
# ---------------------------------------------------------------------------

def inspect_trajectories(trajectories, step_num: int, n_show: int = 3):
    """Print actual model completions and rewards for human inspection."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
        console = Console()
    except ImportError:
        console = None

    print(f"\n{'='*70}")
    print(f"  OUTPUT INSPECTION — Training step {step_num}")
    print(f"{'='*70}")

    for i, traj in enumerate(trajectories[:n_show]):
        if not traj.steps:
            continue

        print(f"\n[Episode {i+1}] {traj.episode_id[:8]}...")
        print(f"  Task: {traj.task_description}")
        print(f"  Steps: {len(traj.steps)} | Total reward: {traj.total_task_reward:.3f}")
        print(f"  Termination: {traj.termination_reason}")
        print(f"  Manipulation count: {traj.manipulation_count} | Suspicion: {traj.final_suspicion:.3f}")
        print()

        for j, step in enumerate(traj.steps):
            action = step.action
            status = "✓" if step.reward > 0 else ("·" if step.reward == -0.1 else "✗")
            print(f"  [{status}] Step {j+1} — {action.action_type} — reward: {step.reward:.2f}")

            if action.action_type == "query":
                print(f"      Query:    {getattr(action, 'query', '')[:120]}")
                # Show the API response from the prompt context
                try:
                    import re
                    responses = re.findall(r'A: (.+?)(?:\n|$)', step.prompt)
                    if responses:
                        print(f"      API resp: {responses[-1][:120]}")
                except Exception:
                    pass
            elif action.action_type == "final_answer":
                print(f"      Answer:   {getattr(action, 'answer', '')[:120]}")
            elif action.action_type == "accuse":
                print(f"      Reason:   {getattr(action, 'accusation_reason', '')[:120]}")

            print(f"      Raw completion: {step.completion[:200]!r}")

    print(f"\n{'='*70}\n")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    # ---- Load model
    model, tokenizer, backend = load_model_and_tokenizer(args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Verify env server is up
    try:
        import requests
        r = requests.get(f"{args.env_url}/health", timeout=5)
        print(f"Env server: {r.json()}")
    except Exception as e:
        print(f"ERROR: Cannot reach env server at {args.env_url}: {e}")
        print("Start it with: uvicorn server.app:app --port 8000")
        sys.exit(1)

    # ---- Curriculum or fixed-difficulty rollout
    curriculum: Optional[CurriculumManager] = None
    if args.curriculum:
        curriculum = CurriculumManager(start_level=args.start_level, seed=args.seed)
        active_rollout = CurriculumRollout(env_url=args.env_url, curriculum=curriculum)
        print(f"Curriculum enabled. Starting at {curriculum.current_level.name}.")
    else:
        active_rollout = EnvironmentRollout(
            env_url=args.env_url,
            max_queries_before_answer=6,
            adversary_mode=args.adversary_mode,
        )

    # ---- GRPOConfig
    from trl import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        max_new_tokens=args.max_new_tokens,
        max_prompt_length=1024,
        num_generations=4,
        temperature=0.7,
        logging_steps=1,
        save_steps=args.steps // 2,
        seed=args.seed,
        report_to="none",
    )

    # ---- Reward functions — multiple independent signals
    # env_reward_fn takes a plain EnvironmentRollout; for curriculum we pass the wrapper
    _base_rollout = active_rollout if not curriculum else EnvironmentRollout(
        env_url=args.env_url, adversary_mode="random"
    )
    env_reward_fn = make_env_reward_fn(_base_rollout, tokenizer, args.max_new_tokens)
    reward_fns = [env_reward_fn, reward_format, reward_action_type]

    # ---- Hack inspector (reused across steps)
    inspector = HackInspector()
    # Buffer of recent trajectories for hack checking
    recent_trajectories = []

    print(f"\nStarting training: {args.steps} steps × {args.episodes_per_step} episodes")
    print(f"Model: {args.model} | Backend: {backend}")
    if not curriculum:
        print(f"Adversary mode: {args.adversary_mode}")
    print()

    all_stats = []

    for step_num in range(1, args.steps + 1):
        if curriculum:
            print(f"\n--- Step {step_num}/{args.steps} [{curriculum.current_level.name}] ---")
        else:
            print(f"\n--- Step {step_num}/{args.steps} ---")

        # ---- Switch to inference mode for rollout
        generate_fn = make_generate_fn(model, tokenizer, args.max_new_tokens, backend)

        # ---- Collect rollout data
        t0 = time.time()
        trajectories = active_rollout.run_batch(
            model_generate_fn=generate_fn,
            tokenizer=tokenizer,
            n_episodes=args.episodes_per_step,
            seed_offset=(step_num - 1) * args.episodes_per_step,
        )
        rollout_time = time.time() - t0

        if not trajectories:
            print(f"  No data collected this step. Skipping update.")
            continue

        # Build HF dataset from trajectories
        from datasets import Dataset
        rows = [
            {"prompt": s.prompt, "completion": s.completion, "reward": s.reward,
             "episode_id": traj.episode_id}
            for traj in trajectories for s in traj.steps
        ]
        dataset = Dataset.from_list(rows) if rows else Dataset.from_list([])

        if len(dataset) == 0:
            print(f"  Empty dataset this step. Skipping update.")
            continue

        # ---- Stats
        avg_reward = sum(r["reward"] for r in rows) / len(rows)
        n_positive = sum(1 for t in trajectories if t.total_task_reward > 0)
        n_final_answer = sum(1 for t in trajectories for s in t.steps if s.action.action_type == "final_answer")
        n_accuse = sum(1 for t in trajectories for s in t.steps if s.action.action_type == "accuse")

        print(f"  Rollout: {len(trajectories)} episodes, {len(dataset)} steps in {rollout_time:.1f}s")
        print(f"  Avg step reward: {avg_reward:.3f}")
        print(f"  Episodes with positive return: {n_positive}/{len(trajectories)}")
        print(f"  final_answer: {n_final_answer}  accuse: {n_accuse}")

        # ---- Curriculum: record results and maybe change level
        if curriculum:
            curriculum.record_batch(trajectories)
            transition = curriculum.maybe_advance_or_simplify()
            if transition:
                print(f"  Curriculum: {transition}! Now at [{curriculum.current_level.name}]")
            else:
                print(f"  {curriculum.status()}")

            # Diagnose zero-reward stagnation
            zero_rate = sum(1 for t in trajectories if t.total_task_reward <= 0) / len(trajectories)
            if zero_rate > 0.8:
                print(f"  ⚠  Zero-reward rate {zero_rate:.0%} — diagnosing...")
                print(f"  {diagnose_zero_reward(trajectories)}")

        # ---- Accumulate trajectories for hack inspection
        recent_trajectories.extend(trajectories)
        if len(recent_trajectories) > 50:
            recent_trajectories = recent_trajectories[-50:]

        stats = {
            "step": step_num,
            "avg_reward": avg_reward,
            "n_episodes": len(trajectories),
            "n_steps": len(dataset),
            "n_positive_episodes": n_positive,
            "curriculum_level": curriculum.level_idx if curriculum else None,
        }

        # ---- Inspect outputs
        if step_num % args.inspect_every == 0 or step_num == 1:
            reenable_training(model, backend)
            inspect_trajectories(trajectories, step_num, n_show=2)

        # ---- Hack inspection
        if args.hack_check_every > 0 and step_num % args.hack_check_every == 0:
            print(f"\n  Running hack inspection on last {len(recent_trajectories)} episodes...")
            report = inspector.run(recent_trajectories)
            print_report(report)
            if report.has_critical:
                print("  ⚠  CRITICAL hack detected. Consider stopping training and investigating.")

        # ---- Switch back to training mode
        reenable_training(model, backend)

        # ---- GRPO update
        print(f"  Running GRPO update on {len(dataset)} steps...")
        t1 = time.time()

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_fns,
            args=grpo_config,
            train_dataset=dataset,
        )
        trainer.train()
        update_time = time.time() - t1
        print(f"  GRPO update done in {update_time:.1f}s")

        stats["update_time"] = update_time
        all_stats.append(stats)

        # ---- Save checkpoint at midpoint and end
        if step_num == args.steps // 2 or step_num == args.steps:
            ckpt_path = Path(args.output_dir) / f"step_{step_num:04d}"
            model.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))
            print(f"  Checkpoint saved: {ckpt_path}")

    # ---- Final hack inspection on all recent trajectories
    if args.hack_check_every > 0 and recent_trajectories:
        print(f"\nFinal hack inspection ({len(recent_trajectories)} episodes)...")
        final_report = inspector.run(recent_trajectories)
        print_report(final_report)

    # ---- Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Steps: {len(all_stats)}")
    if all_stats:
        first_r = all_stats[0]["avg_reward"]
        last_r = all_stats[-1]["avg_reward"]
        print(f"Avg reward: {first_r:.3f} (step 1) → {last_r:.3f} (step {args.steps})")
        print(f"Reward trend: {'↑ improving' if last_r > first_r else '↓ degrading' if last_r < first_r else '→ flat'}")
    if curriculum:
        print(f"\nCurriculum transitions:")
        for t in curriculum.transition_history():
            print(f"  ep {t['episode']}: {t['from']} → {t['to']} "
                  f"({t['direction']}, positive_rate={t['positive_rate_at_transition']:.2f})")
        print(f"Final level: {curriculum.current_level.name}")
    print(f"Checkpoints: {args.output_dir}/")

    active_rollout.close()


if __name__ == "__main__":
    main()
