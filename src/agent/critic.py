
from abstract_classes.approximator import Approximator
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState
from utils.config_parser import Config


class Critic:

    def __init__(self, approximator: Approximator) -> None:
        self.approximator = approximator  # Table or NN
        self.state_values = {}  # V(s)
        self.eligibilities = {}  # State-based eligibilities

    def compute_temporal_difference_error(self, prev_state: UniversalState, state: UniversalAction, reinforcement: int) -> float:
        if str(prev_state) not in self.approximator.state_values:
            self.approximator.initialize_state_value(prev_state)

        if str(state) not in self.approximator.state_values:
            self.approximator.initialize_state_value(state)

        return reinforcement + Config.critic_discount_factor * self.approximator.state_values[prev_state] - self.approximator.state_values[state]

    def reset_eligibilies(self) -> None:
        self.eligibilities = {}

    def initialize_eligibility(self, state: UniversalState):
        self.eligibilities[str(state)] = 1

    def update_eligibility(self, state: UniversalState):
        self.eligibilities[str(state)] = Config.critic_discount_factor * Config.critic_decay_rate * self.eligibilities[str(state)]
