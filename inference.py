"""
Baseline inference script for Adaptive Cyber Defense Simulator.
OpenEnv Hackathon submission — Meta x Scaler.

Usage:
    export API_BASE_URL=https://api.openai.com/v1
    export MODEL_NAME=gpt-4o-mini
    export HF_TOKEN=your_token_here
    python3 inference.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Optional

# ── path bootstrap (works when run from inside the package dir) ──────────────
_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
for _p in (_PARENT, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── environment variables ────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

# ── OpenAI client ────────────────────────────────────────────────────────────
from openai import OpenAI

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy-key",
)

# ── project imports ──────────────────────────────────────────────────────────
from adaptive_cyber_defense.models.action import Action, ActionInput, ACTION_PROFILES
from adaptive_cyber_defense.models.state  import EnvironmentState
from adaptive_cyber_defense.tasks         import EasyTask, MediumTask, HardTask

TASK_MAP = {"easy": EasyTask, "medium": MediumTask, "hard": HardTask}

# ── fallback action order (used when LLM call fails) ────────────────────────
_FALLBACK_ACTIONS = [
    Action.RUN_DEEP_SCAN,
    Action.BLOCK_IP,
    Action.ISOLATE_NODE,
    Action.PATCH_SYSTEM,
    Action.IGNORE,
]


# ── LLM Agent ────────────────────────────────────────────────────────────────

class LLMAgent:
    """
    Calls the LLM to choose an action each step.
    Falls back to a rule-based heuristic on timeout or parse failure.
    """

    def __init__(self, llm_client: OpenAI, model: str, timeout: float = 10.0) -> None:
        self.client  = llm_client
        self.model   = model
        self.timeout = timeout

    # ── prompt builder ───────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(state: EnvironmentState) -> str:
        threats_text = ""
        for t in state.active_threats[:5]:    # cap to keep prompt short
            threats_text += (
                f"  - {t.id}: stage={t.stage.name} severity={t.severity:.2f} "
                f"node={t.current_node} detected={'yes' if t.is_detected else 'no'}\n"
            )
        if not threats_text:
            threats_text = "  (none)\n"

        comp_text = ", ".join(state.compromised_nodes) or "none"

        actions_text = ""
        for action in Action:
            cost = ACTION_PROFILES[action].resource_cost
            actions_text += f"  {action.name} (cost={cost:.2f})\n"

        nodes_text = ""
        for nid, asset in list(state.assets.items())[:8]:
            nodes_text += (
                f"  {nid}: health={asset.health:.0%} "
                f"compromised={'yes' if asset.is_compromised else 'no'} "
                f"isolated={'yes' if asset.is_isolated else 'no'}\n"
            )

        return f"""You are an AI SOC analyst. Choose the best defensive action.

=== CURRENT STATE ===
Step            : {state.time_step}
Episode score   : {state.episode_score:.4f}
Threat severity : {state.threat_severity:.2%}
Resources left  : {state.resource_availability:.2%}
Compromised     : {comp_text}

Active threats:
{threats_text}
Network nodes:
{nodes_text}
Available actions:
{actions_text}

Reply with ONLY a JSON object — no explanation, no markdown:
{{"action": "<ACTION_NAME>", "target": "<node_id>"}}

Choose the action that best contains the highest-severity threat.
Valid ACTION_NAME values: {', '.join(a.name for a in Action)}
"""

    # ── LLM call (runs in thread for timeout support) ────────────────────────

    def _call_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    # ── response parser ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(text: str) -> Optional[ActionInput]:
        # Strip markdown fences if present
        text = text.strip().strip("```json").strip("```").strip()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            # Try to find a JSON object inside the response
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start == -1 or end == 0:
                return None
            try:
                obj = json.loads(text[start:end])
            except json.JSONDecodeError:
                return None

        action_name = str(obj.get("action", "")).upper().strip()
        target      = obj.get("target") or None

        try:
            action = Action[action_name]
        except KeyError:
            return None

        return ActionInput(action=action, target_node=target)

    # ── fallback heuristic ───────────────────────────────────────────────────

    @staticmethod
    def _fallback(state: EnvironmentState) -> ActionInput:
        """Rule-based fallback: target the highest-severity detected threat."""
        threats = sorted(
            state.active_threats,
            key=lambda t: t.effective_severity(),
            reverse=True,
        )
        for t in threats:
            if t.is_detected:
                for action in _FALLBACK_ACTIONS:
                    cost = ACTION_PROFILES[action].resource_cost
                    if state.resource_availability >= cost:
                        return ActionInput(action=action, target_node=t.current_node)

        # No detected threats or no affordable action → scan cheapest node
        node = next(iter(state.assets), "fw-01")
        return ActionInput(action=Action.RUN_DEEP_SCAN, target_node=node)

    # ── public interface ─────────────────────────────────────────────────────

    def select_action(self, state: EnvironmentState) -> ActionInput:
        """
        Ask the LLM for an action; fall back to heuristic on any failure.
        Uses a thread-based timeout to avoid blocking.
        """
        prompt = self._build_prompt(state)

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._call_llm, prompt)
                raw = future.result(timeout=self.timeout)

            action_input = self._parse_response(raw)
            if action_input is not None:
                # Validate the action is affordable; fall back if not
                cost = ACTION_PROFILES[action_input.action].resource_cost
                if state.resource_availability >= cost:
                    return action_input
        except (FuturesTimeoutError, Exception):
            pass  # any failure → use fallback

        return self._fallback(state)

    # compatibility shim for task.run()
    def choose(self, state: EnvironmentState, **_) -> ActionInput:
        return self.select_action(state)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_task(task_name: str, agent: LLMAgent, episodes: int = 3) -> float:
    """
    Run *episodes* episodes on the given task difficulty.
    Returns the average episode_score across all episodes.
    """
    task_cls = TASK_MAP[task_name]
    scores: list[float] = []

    for ep in range(episodes):
        task = task_cls()
        result = task.run(agent, seed=ep)
        scores.append(result.episode_score)
        print(
            f"  [{task_name}] ep {ep+1}/{episodes}  "
            f"score={result.episode_score:.4f}  "
            f"passed={'✓' if result.passed else '✗'}  "
            f"steps={result.steps_taken}",
            flush=True,
        )

    return sum(scores) / len(scores)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"API_BASE_URL : {API_BASE_URL}")
    print(f"MODEL_NAME   : {MODEL_NAME}")
    print(f"HF_TOKEN     : {'set' if HF_TOKEN else 'not set (using dummy-key)'}")
    print()

    agent = LLMAgent(client, MODEL_NAME, timeout=10.0)

    results = []
    t_start = time.time()

    for task_name in ["easy", "medium", "hard"]:
        print(f"--- Running task: {task_name} ---", flush=True)
        score = run_task(task_name, agent, episodes=3)
        record = {"task": task_name, "score": round(score, 4)}
        results.append(record)
        print(json.dumps(record), flush=True)
        print()

    elapsed = time.time() - t_start

    print("=== FINAL RESULTS ===")
    print(json.dumps(results, indent=2))
    avg = sum(r["score"] for r in results) / len(results)
    print(f"Average score : {avg:.4f}")
    print(f"Elapsed       : {elapsed:.1f}s")
