
import numpy as np
from abstract_classes.approximator import Approximator
from environment.universal_state import UniversalState
from utils.config_parser import Config


class TableApproximator(Approximator):

    def __init__(self) -> None:
        super().__init__()

    def compute_state_values(self, temporal_difference_error: float, eligibilities: dict) -> None:
        for key, eligibility in eligibilities.items():
            print(key)
            self.state_values[key] += Config.critic_learning_rate * temporal_difference_error * eligibility

    def compute_state_value(self, state: UniversalState, td_error: float, eligibilities: dict) -> None:
        self.state_values[str(state)] = self.state_values[str(state)] + Config.critic_learning_rate * eligibilities[str(state)] * td_error

    def initialize_state_value(self, state: UniversalState) -> None:
        self.state_values.setdefault(str(state), np.random.uniform(-0.01, 0.01))
