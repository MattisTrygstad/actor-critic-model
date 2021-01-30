
from abstract_classes.approximator import Approximator
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState


class Critic:

    def __init__(self, approximator: Approximator) -> None:
        self.approximator = approximator  # Table or NN

    def compute_temporal_difference_error(self, state: UniversalState, next_state: UniversalAction, reinforcement: int) -> float:
        self.approximator.initialize_state_value(state)
        self.approximator.initialize_state_value(next_state)

        return reinforcement + self.approximator.discount_factor * self.approximator.state_values[str(next_state)] - self.approximator.state_values[str(state)]

    def compute_state_values(self, td_error: float) -> None:
        self.approximator.compute_state_values(td_error)

    def initialize_state_value(self, state: UniversalState) -> None:
        self.approximator.initialize_state_value(state)

    def reset_eligibilities(self) -> None:
        self.approximator.reset_eligibilities()

    def set_eligibility(self, state: UniversalState, value: int):
        self.approximator.set_eligibility(state, value)

    def decay_eligibilities(self) -> None:
        self.approximator.decay_eligibilies()
