
import numpy as np
from abstract_classes.approximator import Approximator
from environment.universal_state import UniversalState


class TableApproximator(Approximator):

    def __init__(self, discount_factor, decay_rate) -> None:
        super().__init__()
        self.discount_factor = discount_factor
        self.decay_rate = decay_rate
        self.eligibilities = {}  # State-based eligibilities

    def compute_state_values(self, td_error: float, learning_rate: float) -> None:
        for key in self.eligibilities:
            self.state_values[key] += learning_rate * self.eligibilities[key] * td_error

    def initialize_state_values(self, state: UniversalState) -> None:
        if str(state) not in self.state_values:
            self.state_values.setdefault(str(state), np.random.uniform(-0.01, 0.01))

    def reset_eligibilities(self) -> None:
        self.eligibilities.clear()

    def set_eligibility(self, state: UniversalState, value: float) -> None:
        self.eligibilities[str(state)] = value

    def decay_eligibilies(self):
        for key in self.eligibilities:
            self.eligibilities[key] *= self.decay_rate * self.discount_factor
