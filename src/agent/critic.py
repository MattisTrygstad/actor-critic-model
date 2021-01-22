
from abstract_classes.approximator import Approximator
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState
from utils.config_parser import Config


class Critic:

    def __init__(self, approximator: Approximator) -> None:
        self.approximator = approximator  # Table or NN
        self.state_values = {}  # V(s)
        self.eligibilities = {}  # State-based eligibilities

    def compute_temporal_difference_error(self, state: UniversalState, next_state: UniversalAction, reinforcement: int) -> float:
        return reinforcement + Config.critic_discount_factor * self.approximator.state_values[str(state)] - self.approximator.state_values[str(next_state)]

    def reset_eligibilies(self) -> None:
        self.eligibilities.clear()

    def set_eligibility(self, state: UniversalState, value: int):
        self.eligibilities[str(state)] = value

    def decay_eligibilies(self):
        for key, eligibility in self.eligibilities.items():
            self.eligibilities[key] = Config.critic_discount_factor * Config.critic_decay_rate * eligibility
