
import numpy as np
from abstract_classes.approximator import Approximator
from environment.universal_state import UniversalState


class TableApproximator(Approximator):

    def __init__(self) -> None:
        super().__init__()

    def compute_state_values(self, temporal_difference_error: float, eligibilities: dict, learning_rate: float) -> None:
        for key, eligibility in eligibilities.items():
            self.state_values[key] += learning_rate * temporal_difference_error * eligibility

    def compute_state_value(self, state: UniversalState, td_error: float, eligibilities: dict, learning_rate: float) -> None:
        self.state_values[str(state)] = self.state_values[str(state)] + learning_rate * eligibilities[str(state)] * td_error

    def initialize_state_value(self, state: UniversalState) -> None:
        self.state_values.setdefault(str(state), np.random.uniform(-0.01, 0.01))
