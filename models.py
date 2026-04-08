# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Data Clean Env Environment.

The data_clean_env environment simulates a real-world data cleaning pipeline.
An AI agent must inspect, clean, and fix messy tabular datasets through
structured cleaning commands.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, model_validator
import json


class DataCleanAction(Action):
    """Action for the Data Clean Env environment.

    The agent issues cleaning commands to transform messy data into clean data.

    Available commands:
        - inspect: View dataset summary statistics
        - view_rows: View specific rows (params: {"start": 0, "end": 10})
        - fill_missing: Fill missing values (params: {"column": "col", "strategy": "mean|median|mode|value", "value": ...})
        - drop_duplicates: Remove duplicate rows (params: {"subset": ["col1", "col2"]})
        - standardize: Standardize format (params: {"column": "col", "format": "date_iso|phone_e164|lowercase|uppercase|title_case"})
        - fix_invalid: Fix invalid entries (params: {"column": "col", "rule": "positive|range|not_future_date", "min": ..., "max": ...})
        - drop_rows: Drop rows by condition (params: {"column": "col", "condition": "is_null|equals|contains", "value": ...})
        - replace_value: Replace values (params: {"column": "col", "old": ..., "new": ...})
        - declare_contract: Declare a data schema constraint explicitly (params: {"column": "col", "rule": "unique|positive|non_null"})
        - submit: Submit cleaned dataset for grading (no params needed)
    """

    command: str = Field(..., description="Cleaning command to execute")
    params: Optional[Dict[str, Any]] = Field(
        default=None, description="Command-specific parameters"
    )

    @model_validator(mode="before")
    @classmethod
    def parse_params(cls, data: Any) -> Any:
        if isinstance(data, dict):
            params = data.get("params")
            if isinstance(params, str):
                try:
                    data["params"] = json.loads(params)
                except json.JSONDecodeError:
                    pass
        return data


class DataCleanObservation(Observation):
    """Observation from the Data Clean Env environment.

    After each action, the agent receives feedback about the current state
    of the dataset and the result of its cleaning action.
    """

    action_result: str = Field(
        default="", description="Result message from the last action"
    )
    dataset_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary: {rows, columns, column_names, missing_count, duplicate_count}",
    )
    current_issues: List[str] = Field(
        default_factory=list, description="List of detected data quality issues"
    )
    data_preview: str = Field(
        default="", description="Preview of first/requested rows as formatted text"
    )
    score_so_far: float = Field(
        default=0.0, description="Current partial score (0.0 - 1.0)"
    )
    task_name: str = Field(default="", description="Name of the current task")
