import pytest
from data_clean_env.server.graders import grade_easy, grade_medium, grade_hard, grade_task
from data_clean_env.data.tasks import EASY_GOLDEN, MEDIUM_GOLDEN, HARD_GOLDEN

def test_grade_easy_perfect():
    # If cleaned_data exactly matches golden, it should get 1.0
    score = grade_easy(EASY_GOLDEN, EASY_GOLDEN)
    assert score == 1.0

def test_grade_easy_empty():
    score = grade_easy([], EASY_GOLDEN)
    assert score == 0.0

def test_grade_medium_perfect():
    score = grade_medium(MEDIUM_GOLDEN, MEDIUM_GOLDEN)
    assert score == 1.0

def test_grade_hard_perfect():
    score = grade_hard(HARD_GOLDEN, HARD_GOLDEN)
    assert score == 1.0

def test_grade_task_router():
    assert grade_task("fix_missing_values", EASY_GOLDEN, EASY_GOLDEN) == 1.0
    assert grade_task("dedup_and_normalize", MEDIUM_GOLDEN, MEDIUM_GOLDEN) == 1.0
    assert grade_task("full_pipeline", HARD_GOLDEN, HARD_GOLDEN) == 1.0
