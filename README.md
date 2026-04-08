---
title: Data Clean Env Server
emoji: 🧹
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - data-cleaning
  - evaluation
---
# 🧹 Data Clean Env Environment

An OpenEnv environment that simulates real-world **data quality management** — one of the most common and costly challenges in enterprise data pipelines. AI agents learn to inspect, clean, and validate messy tabular datasets through structured cleaning commands. 

In each episode, the agent is presented with a messy dataset containing issues like missing values, duplicates, and semantic anomalies, and must iteratively clean it before submitting the final table.

## Quick Start
The simplest way to use the Data Clean environment is through the `DataCleanEnv` class:

```python
from data_clean_env import DataCleanAction, DataCleanEnv

try:
    # Create environment from Docker image
    env = DataCleanEnv.from_docker_image("data_clean_env:latest")

    # Load a dataset (e.g., the downstream ML-impact task)
    result = env.reset(
        task_name='ml_impact',
        seed=42
    )
    print(f"Dataset Summary: {result.observation.dataset_summary}")
    print(f"Initial Issues: {result.observation.current_issues}")

    # Inspect the data
    result = env.step(DataCleanAction(command="inspect"))
    print(f"Data Preview: {result.observation.data_preview}")

    # Clean the dataset
    result = env.step(DataCleanAction(
        command="fill_missing", 
        params={"column": "age", "strategy": "mean"}
    ))
    
    # Declare a data contract to get a bonus multiplier!
    result = env.step(DataCleanAction(
        command="declare_contract", 
        params={"column": "age", "rule": "positive"}
    ))

    # Submit for grading
    result = env.step(DataCleanAction(command="submit"))
    print(f"Score: {result.observation.score_so_far}")  
    print(f"Done: {result.done}")  # True (episode ended)

finally:
    # Always clean up
    env.close()
```

That's it! The `DataCleanEnv.from_docker_image()` method handles starting the Docker container, connecting, and cleaning up.

## Building the Docker Image

Before using the environment locally, you need to build the Docker image:

```bash
# From the data_clean_env directory
docker build -t data_clean_env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or push to a specific repository
openenv push --repo-id Mg0718/data_clean_env
```

After deployment, your space will be available at: `https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment.
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions.

## How to Test Manually in the Hugging Face Space (Playground)

When interacting with the Hugging Face space using the Web Interface (**Playground** widget), **you must initialize the dataset first.** 
If you simply execute `Step` without resetting, you will receive an `"Error executing: list index out of range"` or `"Dataset is empty"` because the tabular dataset hasn't been instantiated!

Here is how you test the exact dataset containing `["id", "name", "age", "department", "salary"]`:

1. **Step 1:** Under the **Params** box, type `{"task_name": "fix_missing_values"}` and click the **Reset** button. This securely loads a fresh batch of data into the active session.
2. **Step 2:** Fill out the input fields to run a command. For example, to preview the dataset:
   - **Command**: `view_rows`
   - **Params**: `{"start": 0, "end": 10}`
3. **Step 3:** Click the **Step** button. 
   - *Result*: The `dataset_summary` will show 20 rows and 5 columns, while the `data_preview` will cleanly output a string slice of the active rows!

**Other examples you can test sequentially via the "Step" button:**
- Fixing Missing Values: `Command`: `fill_missing` | `Params`: `{"column": "age", "strategy": "mean"}`
- Dropping Duplicates: `Command`: `drop_duplicates` | `Params`: `{"subset": ["email"]}`
- Submitting the task: `Command`: `submit` | `Params`: `{}`

## Environment Details

### Episode Structure

Each episode spans across multiple tabular actions:
- `reset(task_name, seed)` initializes a specific task variant and returns an observation.
- `step(action)` executes an operation directly on the dataset and returns a partial step observation.
- `step({"command": "submit"})` submits the current data state to the grader, returning the final score and marking `done: true`.

### Action Space

**`DataCleanAction`**: Contains the command and parameters for cleaning.

- `command` (str): One of: `inspect`, `view_rows`, `fill_missing`, `drop_duplicates`, `standardize`, `fix_invalid`, `drop_rows`, `replace_value`, `declare_contract`, `submit`.
- `params` (Dict): Command-specific parameters. Examples: `{"column": "age", "strategy": "mean"}` or `{"column": "price", "rule": "positive"}`.

### Observation Space

**`DataCleanObservation`**: Contains the current state of the dataset and task context.

- `action_result` (str) - Feedback message from the last execution rule (e.g. "Filled 4 missing values").
- `dataset_summary` (Dict) - The active rows, columns, feature names, and metadata.
- `current_issues` (List[str]) - Detected missing values, duplicates, and semantic anomalies.
- `data_preview` (str) - Slashed visual table structure of the dataset.
- `score_so_far` (float) - Normalized current score (0.0 to 1.0).
- `task_name` (str)  - Originating task name.

## Reward

The reward evaluates the progression of your dataset modifications compared to a golden reference.
Tasks evaluate row count, textual normalization, missingness, and invalid constraints. 

In the novel `ml_impact` task, scoring directly evaluates the downstream $F_1$ classification metric boost (Logistic Regression) against an unaltered dirty baseline.

## Tasks & Configurations

Use a specific task natively via configuration:

```python
result = env.reset(
    task_name='ml_impact', # Other options: fix_missing_values, dedup_and_normalize, full_pipeline
    seed=124               # Implements Procedural Corruption Data Splitting
)
```

## Running Locally

Run the server locally for development:

```bash
# Install dependencies
uv sync

# Run the env server locally
uvicorn server.app:app --reload
```

## Automated Test Coverage

Our `tests/` folder explicitly covers every major environment feature, scoring boundary logic, and generative scenario before space validation. We test these workflows manually in the Playground, and systematically via our CLI utilizing `uv run pytest tests/`.

**What we have explicitly tested and confirmed in our environment (`test_new_features.py`):**
1. `test_ml_impact_data_generation`: Validates procedural sequence generator for the `ml_impact` task datasets so that deterministic algorithms properly inject scaling anomaly errors across seeds.
2. `test_declare_contract`: Emulates the `step/reset` cycles sequentially validating that executing the `declare_contract` command successfully evaluates downstream rewards bounds conditionally. 
3. `test_ml_impact_grader_baseline`: Ensures uncleaned random datasets fed back into the `grade_ml_impact` thresholding yield an exact 0.0 baseline boundary properly avoiding arbitrary exploits.
4. `test_ml_impact_grader_perfect`: Injects pristine target data back against the inference predictor verifying F1 lift scores hit the target 1.0 constraint.
5. `test_semantic_issues_detected`: Simulates `2050` time errors and unit-type inaccuracies within the raw rows to ensure instances are trapped and properly communicated through the `_detect_issues()` parser.
