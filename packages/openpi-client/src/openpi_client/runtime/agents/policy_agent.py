from typing_extensions import override

from openpi_client import base_policy as _base_policy
from openpi_client.runtime import agent as _agent


class PolicyAgent(_agent.Agent):
    """An agent that uses a policy to determine actions."""

    def __init__(self, policy: _base_policy.BasePolicy) -> None:
        self._policy = policy

    @override
    def get_action(self, observation: dict, noise: float = None) -> dict:
        return self._policy.infer(observation, noise)

    def reset(self) -> None:
        self._policy.reset()
        
    @override
    def get_prefix_rep(self, observation: dict) -> dict:
        return self._policy.get_prefix_rep(observation)