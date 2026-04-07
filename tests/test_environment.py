import pytest
from data_clean_env.server.data_clean_env_environment import DataCleanEnvironment
from data_clean_env.models import DataCleanAction

def test_env_reset():
    env = DataCleanEnvironment()
    obs = env.reset(task_name="fix_missing_values")
    assert obs.task_name == "fix_missing_values"
    assert obs.score_so_far == 0.0
    assert not obs.done
    assert len(obs.current_issues) > 0 # Dirty data should have issues

def test_env_submit_early():
    env = DataCleanEnvironment()
    env.reset(task_name="fix_missing_values")
    # Submit without cleaning
    action = DataCleanAction(command="submit")
    obs = env.step(action)
    assert obs.done
    # Score should be > 0 (some data is already correct) but < 1.0
    assert 0.0 < obs.score_so_far < 1.0

def test_env_invalid_command():
    env = DataCleanEnvironment()
    env.reset(task_name="fix_missing_values")
    action = DataCleanAction(command="invalid_cmd")
    obs = env.step(action)
    assert "Unknown command" in obs.action_result
    assert not obs.done

def test_env_inspect():
    env = DataCleanEnvironment()
    env.reset(task_name="fix_missing_values")
    action = DataCleanAction(command="inspect")
    obs = env.step(action)
    assert "Dataset" in obs.action_result
    assert not obs.done

def test_auto_submit_on_max_steps():
    env = DataCleanEnvironment()
    env.reset(task_name="fix_missing_values")
    # Step 30 times
    for _ in range(30):
        obs = env.step(DataCleanAction(command="inspect"))
        if obs.done:
            break
    
    # 31st step is auto-submit
    obs = env.step(DataCleanAction(command="inspect"))
    assert obs.done
    assert "final score" in obs.action_result.lower()
