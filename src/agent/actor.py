
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
        self.eligibilities = {}

    def initialize_eligibility(self, state: UniversalState, action: UniversalAction) -> None:
        key = self.__generate_state_action_pair_key(state, action)
        self.eligibilities[key] = 1

    def update_eligibility(self, state: UniversalState, action: UniversalAction) -> None:
        key = self.__generate_state_action_pair_key(state, action)
        self.eligibilities[key] = Config.actor_discount_factor * Config.actor_decay_rate * self.eligibilities[key]

    def generate_action(self, state: UniversalState, legal_actions: list) -> UniversalAction:

        random_probability = Config.e_initial
        if random.uniform(0, 1) > random_probability:
            print('random')
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

    def compute_policy(self, state: UniversalState, action: UniversalAction, td_error: float) -> None:
        key = self.__generate_state_action_pair_key(state, action)
        if key not in self.policies:
            self.policies[key] = 0

        if key not in self.eligibilities:
            self.eligibilities[key] = 0

        self.policies[key] += Config.actor_learning_rate * td_error * self.eligibilities[key]

    def __generate_state_action_pair_key(self, state: UniversalState, action: UniversalAction):
        return f'state: {str(state)}, action: {str(action)}'
