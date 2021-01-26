
import random
from typing import Tuple
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState


class Actor:
    def __init__(self, actor_discount_factor: float, actor_decay_rate: float, actor_learning_rate: float) -> None:
        self.actor_discount_factor = actor_discount_factor
        self.actor_decay_rate = actor_decay_rate
        self.actor_learning_rate = actor_learning_rate

        self.policies = {}  # Pi(s)
        self.eligibilities = {}  # SAP-based eligibilities
        self.state = None

    def reset_eligibilities(self) -> None:
        self.eligibilities.clear()

    def set_eligibility(self, state: UniversalState, action: UniversalAction, value: int) -> None:
        self.eligibilities.setdefault(str(state), {})[str(action)] = value

    def generate_action(self, state: UniversalState, legal_actions: list, epsilon: float) -> UniversalAction:
        random_action = False
        if random.uniform(0, 1) < epsilon:
            random_index = random.randint(0, len(legal_actions) - 1)
            chosen_action = legal_actions[random_index]
            random_action = True

        else:
            action_policies = {}
            for action in legal_actions:
                self.policies.setdefault(str(state), {}).setdefault(str(action), 0)

                action_policies[action] = self.policies[str(state)][str(action)]

            chosen_action = max(action_policies, key=action_policies.get)

        universal_action = UniversalAction()
        universal_action.action = chosen_action
        return universal_action, random_action

    def compute_policy(self, state: UniversalState, action: UniversalAction, td_error: float) -> None:
        value = self.policies.setdefault(str(state), {}).setdefault(str(action), 0)
        self.policies[str(state)][str(action)] = value + self.actor_learning_rate * self.eligibilities[str(state)][str(action)] * td_error
