"""
Inference Script — Data Clean Env
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=fix_missing_values env=data_clean_env model=Qwen2.5-72B-Instruct
    [STEP] step=1 action=inspect reward=0.00 done=false error=null
    [STEP] step=2 action=fill_missing(age,mean) reward=0.10 done=false error=null
    [STEP] step=3 action=submit reward=0.85 done=true error=null
    [END] success=true steps=3 score=0.85 rewards=0.00,0.10,0.85
"""

import asyncio
import json
import os
import sys
import textwrap
import traceback
from typing import List, Optional

from openai import OpenAI

# Import the environment
from data_clean_env import DataCleanAction, DataCleanEnv

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # If you are using docker image
HF_TOKEN = os.getenv("HF_TOKEN")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = os.getenv("DATA_CLEAN_ENV_BENCHMARK", "data_clean_env")
MAX_STEPS = 15

TASKS = ["fix_missing_values", "dedup_and_normalize", "full_pipeline", "ml_impact"]

SYSTEM_PROMPT = textwrap.dedent("""\
You are a data cleaning agent. You are given a messy dataset and must clean it by issuing commands.

Available commands (respond with EXACTLY one JSON object per turn):
- {"command": "inspect"} — View dataset summary and detected issues
- {"command": "view_rows", "params": {"start": 0, "end": 10}} — View specific rows
- {"command": "fill_missing", "params": {"column": "col_name", "strategy": "mean|median|mode|value", "value": "..."}} — Fill missing values
- {"command": "drop_duplicates", "params": {"subset": ["col1", "col2"]}} — Remove duplicate rows
- {"command": "standardize", "params": {"column": "col_name", "format": "date_iso|phone_e164|lowercase|uppercase|title_case"}} — Standardize formats
- {"command": "fix_invalid", "params": {"column": "col_name", "rule": "positive|non_negative|range", "min": 0, "max": 5}} — Fix invalid values
- {"command": "drop_rows", "params": {"column": "col_name", "condition": "is_null|equals|contains", "value": "..."}} — Drop rows
- {"command": "replace_value", "params": {"column": "col_name", "old": "...", "new": "..."}} — Replace values
- {"command": "submit"} — Submit your cleaned dataset for final grading

IMPORTANT RULES:
1. ALWAYS respond with a single valid JSON object. No extra text.
2. First, use "inspect" to understand the dataset.
3. Fix issues systematically: missing values → duplicates → format standardization → invalid values.
4. When you have fixed all issues, use "submit".
5. You have a limited number of steps, so be efficient.
""")


def parse_llm_action(response_text: str) -> dict:
    """Parse the LLM response into an action dict."""
    text = response_text.strip()

    # Try to extract JSON from the response
    # Handle cases where LLM wraps JSON in markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Find the first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to extract command keyword
        text_lower = text.lower()
        if "submit" in text_lower:
            return {"command": "submit"}
        elif "inspect" in text_lower:
            return {"command": "inspect"}
        else:
            return {"command": "inspect"}  # safe default


def format_action_str(action_dict: dict) -> str:
    """Format action dict into a short string for logging."""
    cmd = action_dict.get("command", "unknown")
    params = action_dict.get("params", {})
    if params:
        param_parts = []
        for k, v in params.items():
            if isinstance(v, list):
                param_parts.append(",".join(str(x) for x in v))
            else:
                param_parts.append(str(v))
        return f"{cmd}({','.join(param_parts)})"
    return cmd


async def run_task(task_name: str, env: DataCleanEnv, client: OpenAI) -> float:
    """Run a single task and return the score."""
    rewards: List[float] = []
    score = 0.0
    steps = 0
    success = False
    last_error: Optional[str] = None

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")

    try:
        # Reset environment with the task
        result = await env.reset(task_name=task_name)
        obs = result.observation

        # Build initial context for the LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Task: {task_name}\n"
                f"Description: {obs.action_result}\n\n"
                f"Dataset Summary: {json.dumps(obs.dataset_summary, indent=2)}\n\n"
                f"Detected Issues:\n" + "\n".join(f"  - {issue}" for issue in obs.current_issues) + "\n\n"
                f"Data Preview:\n{obs.data_preview}\n\n"
                f"Please start cleaning. Respond with a JSON command."
            )},
        ]

        for step_num in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Call LLM
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.1,
                )
                llm_text = response.choices[0].message.content or '{"command": "inspect"}'
            except Exception as e:
                llm_text = '{"command": "submit"}'
                last_error = str(e)

            # Parse action
            action_dict = parse_llm_action(llm_text)
            action = DataCleanAction(
                command=action_dict.get("command", "inspect"),
                params=action_dict.get("params"),
            )

            # Step environment
            result = await env.step(action)
            obs = result.observation
            reward = float(result.reward or 0.0)
            done = result.done
            rewards.append(reward)
            steps = step_num

            # Check for error
            error_str = "null"
            if obs.metadata and obs.metadata.get("error"):
                error_str = str(obs.metadata["error"])
                last_error = error_str

            action_str = format_action_str(action_dict)
            print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={'true' if done else 'false'} error={error_str}")

            if done:
                score = obs.score_so_far
                success = score > 0.5
                break

            # Update conversation for next turn
            messages.append({"role": "assistant", "content": llm_text})
            messages.append({"role": "user", "content": (
                f"Result: {obs.action_result}\n\n"
                f"Dataset Summary: {json.dumps(obs.dataset_summary, indent=2)}\n\n"
                f"Remaining Issues:\n" + "\n".join(f"  - {issue}" for issue in obs.current_issues) + "\n\n"
                f"Current Score: {obs.score_so_far:.2f}\n"
                f"Steps remaining: {MAX_STEPS - step_num}\n\n"
                f"Continue cleaning or submit if done."
            )})

        # If we ran out of steps without submitting, submit now
        if not result.done:
            result = await env.step(DataCleanAction(command="submit"))
            obs = result.observation
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps += 1
            score = obs.score_so_far
            success = score > 0.5
            print(f"[STEP] step={steps} action=submit reward={reward:.2f} done=true error=null")

    except Exception as e:
        last_error = str(e)
        traceback.print_exc(file=sys.stderr)

    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(f"[END] success={'true' if success else 'false'} steps={steps} score={score:.2f} rewards={rewards_str}")

    return score


async def main():
    """Run inference on all tasks."""
    # Initialize OpenAI client
    client = OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
    )

    all_scores = {}

    for task_name in TASKS:
        # Connect to environment
        if LOCAL_IMAGE_NAME:
            env = await DataCleanEnv.from_docker_image(LOCAL_IMAGE_NAME)
        else:
            env = DataCleanEnv(base_url=os.getenv("ENV_URL", "http://localhost:8000"))

        try:
            async with env:
                score = await run_task(task_name, env, client)
                all_scores[task_name] = score
        except Exception as e:
            print(f"[END] success=false steps=0 score=0.00 rewards=0.00", file=sys.stdout)
            all_scores[task_name] = 0.0
            traceback.print_exc(file=sys.stderr)

    # Print summary to stderr (not part of mandatory output)
    print("\n=== INFERENCE SUMMARY ===", file=sys.stderr)
    for task, sc in all_scores.items():
        print(f"  {task}: {sc:.2f}", file=sys.stderr)
    avg = sum(all_scores.values()) / max(1, len(all_scores))
    print(f"  Average: {avg:.2f}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
