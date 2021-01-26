
from abstract_classes.approximator import Approximator
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState


class Critic:

    def __init__(self, approximator: Approximator, critic_discount_factor: float, critic_decay_rate, critic_learning_rate: float) -> None:
        self.critic_discount_factor = critic_discount_factor
        self.critic_decay_rate = critic_decay_rate
        self.critic_learning_rate = critic_learning_rate

        self.approximator = approximator  # Table or NN
        self.eligibilities = {}  # State-based eligibilities

    def compute_temporal_difference_error(self, state: UniversalState, next_state: UniversalAction, reinforcement: int) -> float:
        self.approximator.initialize_state_value(state)
        self.approximator.initialize_state_value(next_state)

        return reinforcement + self.critic_discount_factor * self.approximator.state_values[str(state)] - self.approximator.state_values[str(state)]

    def compute_state_value(self, state: UniversalState, td_error: float) -> None:
        self.approximator.compute_state_value(state, td_error, self.eligibilities, self.critic_learning_rate)

    def initialize_state_value(self, state: UniversalState) -> None:
        self.approximator.initialize_state_value(state)

    def reset_eligibilies(self) -> None:
        self.eligibilities.clear()

    def set_eligibility(self, state: UniversalState, value: int):
        self.eligibilities[str(state)] = value
