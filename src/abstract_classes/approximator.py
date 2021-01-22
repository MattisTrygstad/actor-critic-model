

from abc import ABC, abstractmethod

from environment.universal_state import UniversalState


class Approximator(ABC):
    """
    Abstract class to be implemented by each critic
    """

    def __init__(self) -> None:
        self.state_values = {}

    @abstractmethod
    def compute_state_values(self, temporal_difference_error: float, eligibilities: dict) -> None:
        pass

    @abstractmethod
    def initialize_state_value(self, state_str: str) -> None:
        pass
