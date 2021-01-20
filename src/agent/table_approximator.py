
import numpy as np
from abstract_classes.approximator import Approximator
from environment.universal_state import UniversalState
from utils.config_parser import Config


class TableApproximator(Approximator):

    def __init__(self) -> None:
        super().__init__()

    def compute_state_value(self, state: UniversalState, temporal_difference_error: float, eligibilities: dict) -> None:
        self.state_values[str(state)] += Config.actor_learning_rate * temporal_difference_error * eligibilities[str(state)]

    def initialize_state_value(self, state: UniversalState) -> None:
        self.state_values[str(state)] = np.random.uniform(-0.1, 0.1)
