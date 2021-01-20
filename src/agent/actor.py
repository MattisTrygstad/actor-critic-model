
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState


class Actor:
    def __init__(self) -> None:
        self.policy = {}  # Pi(s)
        self.eligibilities = {}  # SAP-based eligibilities
        self.state = None

    def reset_eligibilities(self) -> None:
        self.eligibilities = {}

    def generate_action() -> UniversalAction:
        universal_action = UniversalAction()
        universal_action.action = ((6, 2), (4, 2))
        return universal_action

    def initialize_eligibility(self, state: UniversalState, action: UniversalAction) -> None:
        key = f'state: {str(state)}, action: {str(action)}'
        self.eligibilities[key] = 1
