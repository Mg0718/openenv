# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data Clean Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import DataCleanAction, DataCleanObservation


class DataCleanEnv(
    EnvClient[DataCleanAction, DataCleanObservation, State]
):
    """
    Client for the Data Clean Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with DataCleanEnv(base_url="http://localhost:8000").sync() as client:
        ...     result = client.reset(task_name="fix_missing_values")
        ...     print(result.observation.action_result)
        ...
        ...     result = client.step(DataCleanAction(command="inspect"))
        ...     print(result.observation.dataset_summary)

    Example with Docker:
        >>> client = DataCleanEnv.from_docker_image("data_clean_env:latest")
        >>> try:
        ...     result = client.reset(task_name="fix_missing_values")
        ...     result = client.step(DataCleanAction(command="submit"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: DataCleanAction) -> Dict:
        """
        Convert DataCleanAction to JSON payload for step message.

        Args:
            action: DataCleanAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload = {"command": action.command}
        if action.params:
            payload["params"] = action.params
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[DataCleanObservation]:
        """
        Parse server response into StepResult[DataCleanObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with DataCleanObservation
        """
        obs_data = payload.get("observation", {})
        observation = DataCleanObservation(
            action_result=obs_data.get("action_result", ""),
            dataset_summary=obs_data.get("dataset_summary", {}),
            current_issues=obs_data.get("current_issues", []),
            data_preview=obs_data.get("data_preview", ""),
            score_so_far=obs_data.get("score_so_far", 0.0),
            task_name=obs_data.get("task_name", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
