---
title: Data Clean Env
emoji: 🧹
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
---
# 🧹 Data Clean Env — Real-World Data Cleaning Environment for OpenEnv

An OpenEnv environment that simulates real-world **data quality management** — one of the most common and costly challenges in enterprise data pipelines. AI agents learn to inspect, clean, and validate messy tabular datasets through structured cleaning commands.

## Why Data Cleaning?

Data quality issues cost businesses an estimated **$12.9 million per year** on average. Data scientists spend **~80% of their time** cleaning and preparing data. This environment provides a realistic simulation for training AI agents to automate data cleaning pipelines.

## Environment Description

The agent receives a **messy dataset** containing various data quality issues:
- 🔲 **Missing values** (nulls, blanks)
- 🔁 **Duplicate rows** (exact and fuzzy duplicates)
- 📅 **Inconsistent date formats** (MM/DD/YYYY, DD/MM/YYYY, "Jan 1, 2024", etc.)
- 📞 **Inconsistent phone formats** (555-0101, (555)0101, +1-555-0101)
- 🔤 **Inconsistent text casing** (NYC vs nyc vs New York City)
- ❌ **Invalid values** (negative prices, ratings > 5.0, negative stock)

The agent must issue cleaning commands to produce a clean dataset, which is graded against a **golden reference**.

## Action Space

The agent sends a `DataCleanAction` with:

| Field | Type | Description |
|-------|------|-------------|
| `command` | `str` | The cleaning command to execute |
| `params` | `dict` (optional) | Command-specific parameters |

### Available Commands

| Command | Description | Example Params |
|---------|-------------|----------------|
| `inspect` | View dataset summary and detected issues | — |
| `view_rows` | View specific rows | `{"start": 0, "end": 10}` |
| `fill_missing` | Fill missing values | `{"column": "age", "strategy": "mean"}` |
| `drop_duplicates` | Remove duplicate rows | `{"subset": ["email"]}` |
| `standardize` | Standardize formats | `{"column": "date", "format": "date_iso"}` |
| `fix_invalid` | Fix invalid values | `{"column": "price", "rule": "positive"}` |
| `drop_rows` | Drop rows by condition | `{"column": "name", "condition": "is_null"}` |
| `replace_value` | Replace specific values | `{"column": "city", "old": "NYC", "new": "New York"}` |
| `submit` | Submit for final grading | — |

## Observation Space

After each action, the agent receives a `DataCleanObservation`:

| Field | Type | Description |
|-------|------|-------------|
| `action_result` | `str` | Result message from the last action |
| `dataset_summary` | `dict` | `{rows, columns, missing_count, duplicate_count}` |
| `current_issues` | `list[str]` | List of detected data quality issues |
| `data_preview` | `str` | Preview of dataset rows |
| `score_so_far` | `float` | Current partial score (0.0 - 1.0) |
| `task_name` | `str` | Current task name |

## Tasks

### Task 1: `fix_missing_values` (Easy)
- **Dataset**: 20-row employee records (5 columns)
- **Issues**: ~30% missing values across age, department, salary, name
- **Goal**: Fill all missing values appropriately
- **Expected difficulty**: Straightforward — basic `fill_missing` commands

### Task 2: `dedup_and_normalize` (Medium)
- **Dataset**: 35-row customer records (30 unique + 5 duplicates, 8 columns)
- **Issues**: Duplicate rows with varied casing, inconsistent date/phone formats, mixed text casing
- **Goal**: Remove duplicates and normalize all formats
- **Expected difficulty**: Moderate — requires recognizing fuzzy duplicates and applying multiple standardizations

### Task 3: `full_pipeline` (Hard)
- **Dataset**: 35-row product inventory (30 unique + 5 duplicates, 12 columns)
- **Issues**: ALL types combined — missing values, duplicates, invalid values (negative prices, out-of-range ratings), inconsistent dates, inconsistent category/status casing
- **Goal**: Complete data cleaning across all issue types
- **Expected difficulty**: Challenging — requires systematic multi-step cleaning strategy

## Reward Function

The reward function provides **partial progress signals**:

```
reward_per_step = improvement_in_score (issues fixed this step)
final_score = grade_task(cleaned_data, golden_data)  # 0.0 - 1.0
```

Grading criteria per task:
- **Easy**: Row count match (10%) + No nulls remaining (30%) + Value accuracy (60%)
- **Medium**: Correct row count (20%) + Date consistency (25%) + Text normalization (25%) + Value accuracy (30%)
- **Hard**: Row count (10%) + No nulls (15%) + No invalid values (15%) + Date consistency (15%) + Category normalization (15%) + Value accuracy (30%)

## Setup Instructions

### Prerequisites
- Python 3.10+
- Docker (for containerized testing)
- Hugging Face CLI (`pip install huggingface-hub`)

### Local Development

```bash
# Install dependencies
cd data_clean_env
uv sync

# Run the server locally
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal, run inference
export HF_TOKEN="your-hf-token"
python inference.py
```

### Docker

```bash
# Build the container
docker build -t data_clean_env:latest -f server/Dockerfile .

# Run the container
docker run -p 8000:8000 data_clean_env:latest

# Run inference against the container
export HF_TOKEN="your-hf-token"
export ENV_URL="http://localhost:8000"
python inference.py
```

### Deploy to Hugging Face Spaces

```bash
cd data_clean_env
openenv push --repo-id your-username/data-clean-env
```

## Baseline Scores

| Task | Baseline Score | Model |
|------|---------------|-------|
| `fix_missing_values` | ~0.65 | Qwen2.5-72B-Instruct |
| `dedup_and_normalize` | ~0.55 | Qwen2.5-72B-Instruct |
| `full_pipeline` | ~0.45 | Qwen2.5-72B-Instruct |

*Scores may vary based on model and LLM response quality.*

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | Hugging Face API token |
| `API_BASE_URL` | No | LLM API endpoint (default: HF router) |
| `MODEL_NAME` | No | Model identifier (default: Qwen2.5-72B-Instruct) |
| `DATA_CLEAN_TASK` | No | Default task on reset (default: fix_missing_values) |
| `IMAGE_NAME` | No | Docker image name for local container |

## Project Structure

```
data_clean_env/
├── __init__.py              # Package exports
├── models.py                # Pydantic Action/Observation models
├── client.py                # EnvClient implementation
├── inference.py             # Baseline inference script (mandatory)
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Dependencies
├── README.md                # This file
├── data/
│   ├── __init__.py
│   └── tasks.py             # Task data generators (dirty + golden)
└── server/
    ├── __init__.py
    ├── data_clean_env_environment.py  # Core Environment
    ├── graders.py            # Deterministic graders
    ├── app.py                # FastAPI application
    └── Dockerfile            # Container definition
```

## License

BSD 3-Clause License
