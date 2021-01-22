
import numpy as np
from abstract_classes.approximator import Approximator
from environment.universal_state import UniversalState
from utils.config_parser import Config


class TableApproximator(Approximator):

    def __init__(self) -> None:
        super().__init__()

    def compute_state_values(self, temporal_difference_error: float, eligibilities: dict) -> None:
        for key, eligibility in eligibilities.items():
            self.state_values[key] += Config.critic_learning_rate * temporal_difference_error * eligibility

    def initialize_state_value(self, state: UniversalState) -> None:
        if str(state) not in self.state_values:
            self.state_values[str(state)] = np.random.uniform(-0.1, 0.1)
