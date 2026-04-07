"""
Deterministic graders for the Data Clean Environment.

Each grader compares the agent's cleaned dataset against the golden reference
and returns a score between 0.0 and 1.0.
"""

import copy
import re
from typing import Any, Dict, List, Tuple


def _normalize_string(s: Any) -> str:
    """Normalize a string value for comparison."""
    if s is None:
        return ""
    return str(s).strip().lower()


def _normalize_date(date_str: Any) -> str:
    """Try to normalize a date string to YYYY-MM-DD format."""
    if date_str is None:
        return ""
    s = str(date_str).strip()

    # Already ISO format
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s

    # Try common formats
    import datetime
    formats = [
        "%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y",
        "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y",
        "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s


def _rows_match(row1: Dict, row2: Dict, columns: List[str], 
                normalize_strings: bool = True, normalize_dates: List[str] = None) -> float:
    """
    Compare two rows and return a match score (0.0 - 1.0).
    
    Each matching column adds to the score proportionally.
    """
    if not columns:
        return 1.0

    normalize_dates = normalize_dates or []
    matches = 0
    total = len(columns)

    for col in columns:
        v1 = row1.get(col)
        v2 = row2.get(col)

        if col in normalize_dates:
            v1 = _normalize_date(v1)
            v2 = _normalize_date(v2)
        elif normalize_strings and isinstance(v1, str) and isinstance(v2, str):
            v1 = _normalize_string(v1)
            v2 = _normalize_string(v2)
        
        # Handle None comparisons
        if v1 is None and v2 is None:
            matches += 1
        elif v1 is None or v2 is None:
            continue  # mismatch
        elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            if abs(float(v1) - float(v2)) < 0.01:
                matches += 1
        elif str(v1).strip().lower() == str(v2).strip().lower():
            matches += 1

    return matches / total if total > 0 else 0.0


def grade_easy(cleaned_data: List[Dict[str, Any]], golden_data: List[Dict[str, Any]]) -> float:
    """
    Grade the easy task: fix_missing_values.

    Scoring:
    - Row count match: 10%
    - No None values remaining: 30%
    - Value accuracy: 60%
    
    Returns: score between 0.0 and 1.0
    """
    if not cleaned_data:
        return 0.0

    score = 0.0
    columns = ["id", "name", "age", "department", "salary"]

    # Row count match (10%)
    if len(cleaned_data) == len(golden_data):
        score += 0.1

    # No None values remaining (30%)
    total_cells = 0
    non_null_cells = 0
    for row in cleaned_data:
        for col in columns:
            total_cells += 1
            if row.get(col) is not None:
                non_null_cells += 1
    if total_cells > 0:
        completeness = non_null_cells / total_cells
        score += 0.3 * completeness

    # Value accuracy (60%) - compare each row to golden
    if len(cleaned_data) > 0 and len(golden_data) > 0:
        row_scores = []
        for golden_row in golden_data:
            best_match = 0.0
            for cleaned_row in cleaned_data:
                if cleaned_row.get("id") == golden_row.get("id"):
                    match = _rows_match(cleaned_row, golden_row, columns)
                    best_match = max(best_match, match)
                    break
            row_scores.append(best_match)
        
        if row_scores:
            score += 0.6 * (sum(row_scores) / len(row_scores))

    return round(min(1.0, score), 2)


def grade_medium(cleaned_data: List[Dict[str, Any]], golden_data: List[Dict[str, Any]]) -> float:
    """
    Grade the medium task: dedup_and_normalize.

    Scoring:
    - Correct row count (no duplicates): 20%
    - Date format consistency (ISO): 25%
    - Text normalization: 25%
    - Value accuracy: 30%
    
    Returns: score between 0.0 and 1.0
    """
    if not cleaned_data:
        return 0.0

    score = 0.0
    columns = ["id", "name", "email", "phone", "signup_date", "city", "plan", "active"]
    date_cols = ["signup_date"]

    # Correct row count — should have 30 rows (original), not 35 (with dupes)
    expected_count = len(golden_data)
    if len(cleaned_data) == expected_count:
        score += 0.2
    elif len(cleaned_data) < expected_count + 5:
        # Partial credit for removing some dupes
        extra = max(0, len(cleaned_data) - expected_count)
        score += 0.2 * max(0, 1 - extra / 5)

    # Date format consistency (25%) — all should be YYYY-MM-DD
    date_scores = []
    for row in cleaned_data:
        d = row.get("signup_date", "")
        if d and re.match(r"^\d{4}-\d{2}-\d{2}$", str(d)):
            date_scores.append(1.0)
        else:
            date_scores.append(0.0)
    if date_scores:
        score += 0.25 * (sum(date_scores) / len(date_scores))

    # Text normalization (25%) — city should be title case, plan lowercase
    text_scores = []
    for row in cleaned_data:
        row_score = 0
        total = 0

        city = row.get("city", "")
        if city and city == city.title():
            row_score += 1
        total += 1

        plan = row.get("plan", "")
        if plan and plan == plan.lower():
            row_score += 1
        total += 1

        text_scores.append(row_score / total if total > 0 else 0)
    if text_scores:
        score += 0.25 * (sum(text_scores) / len(text_scores))

    # Value accuracy (30%)
    row_scores = []
    for golden_row in golden_data:
        best_match = 0.0
        for cleaned_row in cleaned_data:
            match = _rows_match(cleaned_row, golden_row, columns, 
                               normalize_strings=True, normalize_dates=date_cols)
            best_match = max(best_match, match)
        row_scores.append(best_match)
    if row_scores:
        score += 0.3 * (sum(row_scores) / len(row_scores))

    return round(min(1.0, score), 2)


def grade_hard(cleaned_data: List[Dict[str, Any]], golden_data: List[Dict[str, Any]]) -> float:
    """
    Grade the hard task: full_pipeline.

    Scoring:
    - Correct row count (removed duplicates): 10%
    - No missing values: 15%
    - No invalid values: 15%
    - Date consistency: 15%
    - Category/status normalization: 15%
    - Value accuracy: 30%
    
    Returns: score between 0.0 and 1.0
    """
    if not cleaned_data:
        return 0.0

    score = 0.0
    columns = ["id", "product_name", "category", "price", "stock", "supplier",
                "sku", "weight_kg", "rating", "review_count", "launch_date", "status"]
    date_cols = ["launch_date"]

    # Correct row count (10%)
    expected = len(golden_data)
    if len(cleaned_data) == expected:
        score += 0.1
    elif len(cleaned_data) < expected + 5:
        extra = max(0, len(cleaned_data) - expected)
        score += 0.1 * max(0, 1 - extra / 5)

    # No missing values (15%)
    total_cells = 0
    non_null = 0
    for row in cleaned_data:
        for col in columns:
            total_cells += 1
            if row.get(col) is not None:
                non_null += 1
    if total_cells > 0:
        score += 0.15 * (non_null / total_cells)

    # No invalid values (15%)
    valid_count = 0
    total_checks = 0
    for row in cleaned_data:
        # Price should be positive
        price = row.get("price")
        if price is not None:
            total_checks += 1
            if isinstance(price, (int, float)) and price > 0:
                valid_count += 1

        # Stock should be non-negative
        stock = row.get("stock")
        if stock is not None:
            total_checks += 1
            if isinstance(stock, (int, float)) and stock >= 0:
                valid_count += 1

        # Rating should be 0.0 - 5.0
        rating = row.get("rating")
        if rating is not None:
            total_checks += 1
            if isinstance(rating, (int, float)) and 0 <= rating <= 5.0:
                valid_count += 1

        # Review count should be non-negative
        rc = row.get("review_count")
        if rc is not None:
            total_checks += 1
            if isinstance(rc, (int, float)) and rc >= 0:
                valid_count += 1

    if total_checks > 0:
        score += 0.15 * (valid_count / total_checks)

    # Date consistency (15%)
    date_scores = []
    for row in cleaned_data:
        d = row.get("launch_date", "")
        if d and re.match(r"^\d{4}-\d{2}-\d{2}$", str(d)):
            date_scores.append(1.0)
        else:
            date_scores.append(0.0)
    if date_scores:
        score += 0.15 * (sum(date_scores) / len(date_scores))

    # Category/status normalization (15%)
    norm_scores = []
    valid_categories = {"electronics", "furniture", "stationery"}
    valid_statuses = {"active", "inactive", "discontinued"}
    for row in cleaned_data:
        row_score = 0
        total = 0

        cat = row.get("category", "")
        total += 1
        if cat and str(cat).lower() in valid_categories:
            row_score += 1

        status = row.get("status", "")
        total += 1
        if status and str(status).lower() in valid_statuses:
            row_score += 1

        norm_scores.append(row_score / total if total > 0 else 0)
    if norm_scores:
        score += 0.15 * (sum(norm_scores) / len(norm_scores))

    # Value accuracy (30%)
    row_scores = []
    for golden_row in golden_data:
        best_match = 0.0
        for cleaned_row in cleaned_data:
            match = _rows_match(cleaned_row, golden_row, columns,
                                normalize_strings=True, normalize_dates=date_cols)
            best_match = max(best_match, match)
        row_scores.append(best_match)
    if row_scores:
        score += 0.3 * (sum(row_scores) / len(row_scores))

    return round(min(1.0, score), 2)


def grade_task(task_name: str, cleaned_data: List[Dict[str, Any]], 
               golden_data: List[Dict[str, Any]]) -> float:
    """
    Grade a task submission.
    
    Args:
        task_name: Name of the task
        cleaned_data: The agent's cleaned dataset
        golden_data: The golden reference dataset
        
    Returns:
        Score between 0.0 and 1.0
    """
    graders = {
        "fix_missing_values": grade_easy,
        "dedup_and_normalize": grade_medium,
        "full_pipeline": grade_hard,
    }
    
    if task_name not in graders:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(graders.keys())}")
    
    return graders[task_name](cleaned_data, golden_data)
