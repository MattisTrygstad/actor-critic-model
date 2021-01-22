
from abstract_classes.approximator import Approximator
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState
from utils.config_parser import Config


class Critic:

    def __init__(self, approximator: Approximator) -> None:
        self.approximator = approximator  # Table or NN
        self.eligibilities = {}  # State-based eligibilities

    def compute_temporal_difference_error(self, state: UniversalState, next_state: UniversalAction, reinforcement: int) -> float:
        self.approximator.initialize_state_value(state)
        self.approximator.initialize_state_value(next_state)

        return reinforcement + Config.critic_discount_factor * self.approximator.state_values[str(state)] - self.approximator.state_values[str(state)]

    def compute_state_values(self, temporal_difference_error: float) -> None:
        self.approximator.compute_state_values(temporal_difference_error, self.eligibilities)

    # new
    def compute_state_value(self, state: UniversalState, td_error: float) -> None:
        self.approximator.compute_state_value(state, td_error, self.eligibilities)

    def initialize_state_value(self, state: UniversalState) -> None:
        self.approximator.initialize_state_value(state)

    def reset_eligibilies(self) -> None:
        self.eligibilities.clear()

    def set_eligibility(self, state: UniversalState, value: int):
        self.eligibilities[str(state)] = value

    def decay_eligibilies(self):
        for key, eligibility in self.eligibilities.items():
            self.eligibilities[key] = Config.critic_discount_factor * Config.critic_decay_rate * eligibility
