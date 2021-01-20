
class Actor:
    def __init__(self) -> None:
        self.state_values = {}
        self.eligibilities = {}
        self.state = None

    def reset_eligibilities(self) -> None:
        self.eligibilities = {}
