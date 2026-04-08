# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data Clean Env Environment Implementation.

A real-world data cleaning simulation where an AI agent must inspect,
clean, and fix messy tabular datasets using structured cleaning commands.

Tasks:
    - fix_missing_values (easy): Fill missing values in employee data
    - dedup_and_normalize (medium): Remove duplicates, normalize formats
    - full_pipeline (hard): Complete data cleaning with all issue types
"""

import copy
import json
import os
import re
import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DataCleanAction, DataCleanObservation
    from ..data.tasks import get_task_data, get_task_description, list_tasks
    from .graders import grade_task
except (ImportError, ModuleNotFoundError):
    try:
        from models import DataCleanAction, DataCleanObservation
        from data.tasks import get_task_data, get_task_description, list_tasks
        from server.graders import grade_task
    except (ImportError, ModuleNotFoundError):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from models import DataCleanAction, DataCleanObservation
        from data.tasks import get_task_data, get_task_description, list_tasks
        from server.graders import grade_task


class DataCleanEnvironment(Environment):
    """
    Data cleaning environment for training AI agents.

    The agent is presented with a messy dataset and must issue cleaning
    commands to produce a clean result. The environment tracks issues,
    provides partial reward signals, and grades the final submission
    against a golden reference.

    Environment contract:
        - reset(task_name=...) -> start a new episode with a specific task
        - step(action) -> execute a cleaning command, get observation
        - state() -> current episode metadata
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the data_clean_env environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._current_task: str = ""
        self._dirty_data: List[Dict[str, Any]] = []
        self._golden_data: List[Dict[str, Any]] = []
        self._working_data: List[Dict[str, Any]] = []
        self._initial_issue_count: int = 0
        self._max_steps: int = 30
        self._submitted: bool = False
        self._last_score: float = 0.0
        self._contracts: List[Dict[str, str]] = []

    def _detect_issues(self, data: List[Dict[str, Any]]) -> List[str]:
        """Detect data quality issues in the current working dataset."""
        issues = []
        if not data:
            return ["Dataset is empty"]

        columns = list(data[0].keys()) if data else []

        # Check for missing values
        missing_cols = {}
        for row in data:
            for col in columns:
                if row.get(col) is None:
                    missing_cols[col] = missing_cols.get(col, 0) + 1
        for col, count in missing_cols.items():
            issues.append(f"Missing values in '{col}': {count} rows")

        # Check for duplicates (by key columns — use all non-id columns)
        if len(data) > 1:
            seen_keys = set()
            dup_count = 0
            key_cols = [c for c in columns if c != "id"]
            for row in data:
                key = tuple(str(row.get(c, "")).strip().lower() for c in key_cols)
                if key in seen_keys:
                    dup_count += 1
                else:
                    seen_keys.add(key)
            if dup_count > 0:
                issues.append(f"Duplicate rows detected: {dup_count}")

        # Check for inconsistent date formats
        date_cols = [c for c in columns if "date" in c.lower()]
        for col in date_cols:
            non_iso = 0
            for row in data:
                v = row.get(col)
                if v is not None and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(v)):
                    non_iso += 1
            if non_iso > 0:
                issues.append(f"Inconsistent date format in '{col}': {non_iso} rows not ISO format")

        # Check for invalid numeric values
        numeric_checks = {
            "price": ("positive", 0),
            "salary": ("positive", 0),
            "stock": ("non-negative", 0),
            "age": ("range", (0, 150)),
            "rating": ("range", (0, 5.0)),
            "review_count": ("non-negative", 0),
            "weight_kg": ("positive", 0),
        }
        for col, (check_type, param) in numeric_checks.items():
            if col not in columns:
                continue
            invalid_count = 0
            for row in data:
                v = row.get(col)
                if v is None:
                    continue
                try:
                    v = float(v)
                    if check_type == "positive" and v <= 0:
                        invalid_count += 1
                    elif check_type == "non-negative" and v < 0:
                        invalid_count += 1
                    elif check_type == "range":
                        lo, hi = param
                        if v < lo or v > hi:
                            invalid_count += 1
                except (ValueError, TypeError):
                    invalid_count += 1
            if invalid_count > 0:
                issues.append(f"Invalid values in '{col}': {invalid_count} rows")

        # Check for inconsistent text casing (categorical columns)
        cat_cols = [c for c in columns if c in ("category", "department", "plan", "status", "city")]
        for col in cat_cols:
            values = set()
            for row in data:
                v = row.get(col)
                if v is not None:
                    values.add(str(v))
            # Check if we have mixed casing
            lower_values = set(v.lower() for v in values)
            if len(values) > len(lower_values):
                issues.append(f"Inconsistent casing in '{col}': {len(values)} variants for {len(lower_values)} unique values")

        # Check for semantic / contextual errors
        semantic_issues = 0
        for row in data:
            # 1. Salary magnitude outlier (e.g. 75 instead of 75000)
            salary = row.get("salary")
            if salary is not None and isinstance(salary, (int, float)) and 0 < salary < 1000:
                semantic_issues += 1
            
            # 2. Future launch date
            launch = row.get("launch_date")
            if launch is not None and isinstance(launch, str):
                try:
                    parts = launch.split("-")
                    if len(parts) == 3 and int(parts[0]) > 2025:
                        semantic_issues += 1
                except ValueError:
                    pass
            
            # 3. Unit inconsistency: weight_kg but value is suspiciously large
            weight = row.get("weight_kg")
            if weight is not None and isinstance(weight, (int, float)) and weight > 100.0:
                semantic_issues += 1

            # 4. Cross-column constraint: rating high but review_count = 0
            rating = row.get("rating")
            rc = row.get("review_count")
            if rating is not None and rc is not None:
                try:
                    if float(rating) > 3.0 and float(rc) == 0:
                        semantic_issues += 1
                except (ValueError, TypeError):
                    pass

        if semantic_issues > 0:
            issues.append(f"Semantic/Contextual violations detected: {semantic_issues} rows")

        return issues

    def _get_summary(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get dataset summary statistics."""
        if not data:
            return {"rows": 0, "columns": 0, "column_names": [], "missing_count": 0, "duplicate_count": 0}

        columns = list(data[0].keys())
        missing = sum(1 for row in data for col in columns if row.get(col) is None)

        seen = set()
        dup_count = 0
        key_cols = [c for c in columns if c != "id"]
        for row in data:
            key = tuple(str(row.get(c, "")).strip().lower() for c in key_cols)
            if key in seen:
                dup_count += 1
            else:
                seen.add(key)

        return {
            "rows": len(data),
            "columns": len(columns),
            "column_names": columns,
            "missing_count": missing,
            "duplicate_count": dup_count,
        }

    def _format_preview(self, data: List[Dict[str, Any]], start: int = 0, end: int = 5) -> str:
        """Format rows as a readable string table."""
        if not data:
            return "(empty dataset)"

        end = min(end, len(data))
        start = max(0, start)
        subset = data[start:end]

        if not subset:
            return "(no rows in range)"

        columns = list(subset[0].keys())
        # Simple tab-separated format
        lines = ["\t".join(columns)]
        for row in subset:
            vals = [str(row.get(c, "None")) for c in columns]
            lines.append("\t".join(vals))

        return "\n".join(lines) + f"\n\n[Showing rows {start}-{end-1} of {len(data)} total]"

    def _compute_partial_score(self) -> float:
        """Compute partial score based on issues fixed so far."""
        current_issues = len(self._detect_issues(self._working_data))
        if self._initial_issue_count == 0:
            return 1.0
        fixed = max(0, self._initial_issue_count - current_issues)
        return round(fixed / self._initial_issue_count, 2)

    def reset(self, seed: Optional[int] = None, task_name: Optional[str] = None, **kwargs) -> DataCleanObservation:
        """
        Reset the environment with a specific task.

        Args:
            seed: Optional random seed
            task_name: Task to load. One of: fix_missing_values, dedup_and_normalize, full_pipeline
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._submitted = False
        self._last_score = 0.0
        self._contracts = []

        # Default task from env var or use first task
        task_name = task_name or kwargs.get("task_name") or os.environ.get("DATA_CLEAN_TASK", "fix_missing_values")

        if task_name not in list_tasks():
            return DataCleanObservation(
                action_result=f"Unknown task: {task_name}. Available: {list_tasks()}",
                done=True,
                reward=0.0,
                task_name=task_name,
            )

        self._current_task = task_name
        self._dirty_data, self._golden_data = get_task_data(task_name, seed=seed)
        self._working_data = copy.deepcopy(self._dirty_data)

        issues = self._detect_issues(self._working_data)
        self._initial_issue_count = len(issues)

        task_desc = get_task_description(task_name)
        summary = self._get_summary(self._working_data)
        preview = self._format_preview(self._working_data, 0, 5)

        return DataCleanObservation(
            action_result=f"Task '{task_name}' loaded. {task_desc}",
            dataset_summary=summary,
            current_issues=issues,
            data_preview=preview,
            score_so_far=0.0,
            done=False,
            reward=0.0,
            task_name=task_name,
            metadata={
                "task": task_name,
                "difficulty": {"fix_missing_values": "easy", "dedup_and_normalize": "medium", "full_pipeline": "hard"}.get(task_name, "unknown"),
                "description": task_desc,
            },
        )

    def step(self, action: DataCleanAction) -> DataCleanObservation:
        """
        Execute a cleaning command.

        Available commands: inspect, view_rows, fill_missing, drop_duplicates,
        standardize, fix_invalid, drop_rows, replace_value, submit
        """
        self._state.step_count += 1

        if self._submitted:
            return DataCleanObservation(
                action_result="Episode already submitted. Call reset() to start a new task.",
                dataset_summary=self._get_summary(self._working_data),
                current_issues=[],
                score_so_far=self._last_score,
                done=True,
                reward=0.0,
                task_name=self._current_task,
            )

        if self._state.step_count > self._max_steps:
            # Auto-submit on max steps
            return self._handle_submit()

        command = action.command.lower().strip()
        params = action.params or {}

        try:
            if command == "inspect":
                result = self._handle_inspect()
            elif command == "view_rows":
                result = self._handle_view_rows(params)
            elif command == "fill_missing":
                result = self._handle_fill_missing(params)
            elif command == "drop_duplicates":
                result = self._handle_drop_duplicates(params)
            elif command == "standardize":
                result = self._handle_standardize(params)
            elif command == "fix_invalid":
                result = self._handle_fix_invalid(params)
            elif command == "drop_rows":
                result = self._handle_drop_rows(params)
            elif command == "replace_value":
                result = self._handle_replace_value(params)
            elif command == "declare_contract":
                result = self._handle_declare_contract(params)
            elif command == "submit":
                result = self._handle_submit()
            else:
                result = f"Unknown command: '{command}'. Available: inspect, view_rows, fill_missing, drop_duplicates, standardize, fix_invalid, drop_rows, replace_value, declare_contract, submit"
        except Exception as e:
            result = f"Error executing '{command}': {str(e)}"

        if isinstance(result, DataCleanObservation):
            return result

        # Compute current state
        issues = self._detect_issues(self._working_data)
        partial_score = self._compute_partial_score()
        prev_score = self._last_score
        self._last_score = partial_score

        # Reward = improvement in score
        reward = max(0, partial_score - prev_score)

        return DataCleanObservation(
            action_result=str(result),
            dataset_summary=self._get_summary(self._working_data),
            current_issues=issues,
            data_preview=self._format_preview(self._working_data, 0, 5),
            score_so_far=partial_score,
            done=False,
            reward=round(reward, 4),
            task_name=self._current_task,
            metadata={"step": self._state.step_count, "command": command},
        )

    def _handle_inspect(self) -> str:
        """Return dataset summary and detected issues."""
        summary = self._get_summary(self._working_data)
        issues = self._detect_issues(self._working_data)
        return f"Dataset: {summary['rows']} rows, {summary['columns']} columns. " \
               f"Missing: {summary['missing_count']}, Duplicates: {summary['duplicate_count']}. " \
               f"Issues found: {len(issues)}"

    def _handle_view_rows(self, params: Dict) -> str:
        """View specific rows."""
        start = int(params.get("start", 0))
        end = int(params.get("end", 10))
        return self._format_preview(self._working_data, start, end)

    def _handle_fill_missing(self, params: Dict) -> str:
        """Fill missing values in a column."""
        column = params.get("column")
        strategy = params.get("strategy", "value")
        fill_value = params.get("value")

        if not column:
            return "Error: 'column' parameter required"

        if column not in self._working_data[0]:
            return f"Error: column '{column}' not found. Available: {list(self._working_data[0].keys())}"

        filled = 0
        if strategy == "mean":
            values = [row[column] for row in self._working_data if row.get(column) is not None and isinstance(row.get(column), (int, float))]
            if values:
                mean_val = sum(values) / len(values)
                for row in self._working_data:
                    if row.get(column) is None:
                        row[column] = round(mean_val, 2)
                        filled += 1
        elif strategy == "median":
            values = sorted([row[column] for row in self._working_data if row.get(column) is not None and isinstance(row.get(column), (int, float))])
            if values:
                mid = len(values) // 2
                median_val = values[mid] if len(values) % 2 else (values[mid-1] + values[mid]) / 2
                for row in self._working_data:
                    if row.get(column) is None:
                        row[column] = round(median_val, 2)
                        filled += 1
        elif strategy == "mode":
            values = [row[column] for row in self._working_data if row.get(column) is not None]
            if values:
                from collections import Counter
                mode_val = Counter(values).most_common(1)[0][0]
                for row in self._working_data:
                    if row.get(column) is None:
                        row[column] = mode_val
                        filled += 1
        elif strategy == "value":
            if fill_value is None:
                return "Error: 'value' parameter required when strategy='value'"
            for row in self._working_data:
                if row.get(column) is None:
                    row[column] = fill_value
                    filled += 1
        else:
            return f"Error: unknown strategy '{strategy}'. Use: mean, median, mode, value"

        return f"Filled {filled} missing values in '{column}' using strategy '{strategy}'"

    def _handle_drop_duplicates(self, params: Dict) -> str:
        """Remove duplicate rows."""
        subset = params.get("subset")  # list of columns to check

        if subset:
            if isinstance(subset, str):
                subset = [subset]
        else:
            # Use all non-id columns
            subset = [c for c in self._working_data[0].keys() if c != "id"]

        seen = set()
        unique_rows = []
        removed = 0

        for row in self._working_data:
            key = tuple(str(row.get(c, "")).strip().lower() for c in subset)
            if key not in seen:
                seen.add(key)
                unique_rows.append(row)
            else:
                removed += 1

        self._working_data = unique_rows
        return f"Removed {removed} duplicate rows (checked columns: {subset})"

    def _handle_standardize(self, params: Dict) -> str:
        """Standardize format in a column."""
        column = params.get("column")
        fmt = params.get("format", "lowercase")

        if not column:
            return "Error: 'column' parameter required"

        if column not in self._working_data[0]:
            return f"Error: column '{column}' not found"

        changed = 0
        for row in self._working_data:
            v = row.get(column)
            if v is None:
                continue

            original = v
            if fmt == "lowercase":
                row[column] = str(v).lower()
            elif fmt == "uppercase":
                row[column] = str(v).upper()
            elif fmt == "title_case":
                row[column] = str(v).title()
            elif fmt == "date_iso":
                row[column] = self._parse_date_to_iso(str(v))
            elif fmt == "phone_e164":
                row[column] = self._normalize_phone(str(v))
            else:
                return f"Error: unknown format '{fmt}'. Use: lowercase, uppercase, title_case, date_iso, phone_e164"

            if row[column] != original:
                changed += 1

        return f"Standardized {changed} values in '{column}' to format '{fmt}'"

    def _parse_date_to_iso(self, date_str: str) -> str:
        """Parse various date formats to ISO YYYY-MM-DD."""
        s = date_str.strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            return s

        formats = [
            "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%Y/%m/%d",
            "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
            "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y",
        ]
        for fmt in formats:
            try:
                dt = datetime.datetime.strptime(s, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        # Try more flexible parsing
        # Handle "Sept" -> "Sep"
        s_fixed = s.replace("Sept ", "Sep ")
        for fmt in formats:
            try:
                dt = datetime.datetime.strptime(s_fixed, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return s  # return original if can't parse

    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone to +1-555-XXXX format."""
        digits = re.sub(r"[^\d]", "", phone)
        if len(digits) == 7:
            return f"+1-{digits[:3]}-{digits[3:]}"
        elif len(digits) == 10:
            return f"+1-{digits[:3]}-{digits[3:]}"
        elif len(digits) == 11 and digits[0] == "1":
            return f"+1-{digits[1:4]}-{digits[4:]}"
        return phone

    def _handle_fix_invalid(self, params: Dict) -> str:
        """Fix invalid values in a column."""
        column = params.get("column")
        rule = params.get("rule", "positive")
        fix_value = params.get("value")

        if not column:
            return "Error: 'column' parameter required"

        if column not in self._working_data[0]:
            return f"Error: column '{column}' not found"

        fixed = 0
        for row in self._working_data:
            v = row.get(column)
            if v is None:
                continue
            try:
                v = float(v)
            except (ValueError, TypeError):
                continue

            needs_fix = False
            if rule == "positive" and v <= 0:
                needs_fix = True
            elif rule == "non_negative" and v < 0:
                needs_fix = True
            elif rule == "range":
                lo = float(params.get("min", 0))
                hi = float(params.get("max", 100))
                if v < lo or v > hi:
                    needs_fix = True

            if needs_fix:
                if fix_value is not None:
                    row[column] = fix_value
                elif rule == "positive":
                    row[column] = abs(v) if v != 0 else 1
                elif rule == "non_negative":
                    row[column] = abs(v)
                elif rule == "range":
                    lo = float(params.get("min", 0))
                    hi = float(params.get("max", 100))
                    row[column] = max(lo, min(hi, abs(v)))
                fixed += 1

        return f"Fixed {fixed} invalid values in '{column}' using rule '{rule}'"

    def _handle_drop_rows(self, params: Dict) -> str:
        """Drop rows matching a condition."""
        column = params.get("column")
        condition = params.get("condition", "is_null")
        value = params.get("value")

        if not column:
            return "Error: 'column' parameter required"

        before = len(self._working_data)
        if condition == "is_null":
            self._working_data = [r for r in self._working_data if r.get(column) is not None]
        elif condition == "equals":
            self._working_data = [r for r in self._working_data if str(r.get(column, "")).lower() != str(value).lower()]
        elif condition == "contains":
            self._working_data = [r for r in self._working_data if str(value).lower() not in str(r.get(column, "")).lower()]
        else:
            return f"Error: unknown condition '{condition}'. Use: is_null, equals, contains"

        dropped = before - len(self._working_data)
        return f"Dropped {dropped} rows where '{column}' {condition} {value or ''}"

    def _handle_replace_value(self, params: Dict) -> str:
        """Replace specific values in a column."""
        column = params.get("column")
        old = params.get("old")
        new = params.get("new")

        if not column:
            return "Error: 'column' parameter required"
        if old is None or new is None:
            return "Error: 'old' and 'new' parameters required"

        replaced = 0
        for row in self._working_data:
            v = row.get(column)
            if v is not None and str(v).strip().lower() == str(old).strip().lower():
                row[column] = new
                replaced += 1

        return f"Replaced {replaced} occurrences of '{old}' with '{new}' in '{column}'"

    def _handle_declare_contract(self, params: Dict) -> str:
        """Declare a schema constraint check."""
        column = params.get("column")
        rule = params.get("rule")
        if not column or not rule:
            return "Error: 'column' and 'rule' parameters required"
        self._contracts.append({"column": column, "rule": rule})
        return f"Declared contract: {column} must be {rule}"

    def _check_contracts_satisfied(self) -> bool:
        """Check if all declared contracts are satisfied by the current dataset."""
        if not self._contracts or not self._working_data:
            return False
            
        columns = self._working_data[0].keys()
        for contract in self._contracts:
            col = contract["column"]
            rule = contract["rule"]
            if col not in columns:
                return False
                
            if rule == "unique":
                seen = set()
                for r in self._working_data:
                    v = r.get(col)
                    if v in seen: return False
                    seen.add(v)
            elif rule == "non_null":
                if any(r.get(col) is None for r in self._working_data):
                    return False
            elif rule == "positive":
                for r in self._working_data:
                    v = r.get(col)
                    if v is not None:
                        try:
                            if float(v) <= 0: return False
                        except (ValueError, TypeError):
                            return False
        return True

    def _handle_submit(self) -> DataCleanObservation:
        """Submit the cleaned dataset for grading."""
        self._submitted = True

        # Grade the submission
        final_score = grade_task(self._current_task, self._working_data, self._golden_data)
        
        # Apply contract bonus if declared and correctly enforced
        if self._contracts and self._check_contracts_satisfied():
            final_score = min(1.0, final_score + 0.1)

        self._last_score = final_score

        return DataCleanObservation(
            action_result=f"Submitted! Final score: {final_score:.2f}/1.00",
            dataset_summary=self._get_summary(self._working_data),
            current_issues=self._detect_issues(self._working_data),
            score_so_far=final_score,
            done=True,
            reward=final_score,
            task_name=self._current_task,
            metadata={
                "final_score": final_score,
                "task": self._current_task,
                "steps_used": self._state.step_count,
            },
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
