
import random
from typing import Tuple
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState
from utils.config_parser import Config


class Actor:
    def __init__(self) -> None:
        self.policies = {}  # Pi(s)
        self.eligibilities = {}  # SAP-based eligibilities
        self.state = None

    def reset_eligibilities(self) -> None:
        self.eligibilities.clear()

    def set_eligibility(self, state: UniversalState, action: UniversalAction) -> None:
        key = self.__generate_state_action_pair_key(state, action)
        self.eligibilities[key] = 1

    def decay_eligibilities(self) -> None:
        for key, eligibility in self.eligibilities.items():
            self.eligibilities[key] = Config.actor_discount_factor * Config.actor_decay_rate * eligibility

    def initialize_state_action_pairs(self, state: UniversalState, legal_actions: UniversalAction) -> None:
        for action in legal_actions:
            key = self.__generate_state_action_pair_key(state, action)
            if key not in self.eligibilities:
                self.eligibilities[key] = 0

    def generate_action(self, state: UniversalState, legal_actions: list, epsilon: float) -> UniversalAction:
        if random.uniform(0, 1) > epsilon:
            random_index = random.randint(0, len(legal_actions) - 1)
            chosen_action = legal_actions[random_index]

        else:
            action_policies = {}
            for chosen_action in legal_actions:
                key = self.__generate_state_action_pair_key(state, chosen_action)
                if key not in self.policies:
                    self.policies[key] = 0

                action_policies[chosen_action] = self.policies[key]

            chosen_action = max(action_policies, key=action_policies.get)

        universal_action = UniversalAction()
        universal_action.action = chosen_action
        return universal_action

    def compute_policies(self, td_error: float) -> None:
        for key, eligibility in self.eligibilities.items():
            if key not in self.policies:
                self.policies[key] = 0

            self.policies[key] += Config.actor_learning_rate * td_error * eligibility

    def __generate_state_action_pair_key(self, state: UniversalState, action: UniversalAction):
        return f'state: {str(state)}, action: {str(action)}'
