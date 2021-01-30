

from abc import ABC, abstractmethod
from typing import Union

from environment.universal_state import UniversalState


class Approximator(ABC):
    """
    Abstract class to be implemented by each critic
    """

    def __init__(self) -> None:
        self.state_values = {}
        self.eligibilities: Union[dict, list]

    @abstractmethod
    def compute_state_values(self, td_error: float, learning_rate: float) -> None:
        pass

    @abstractmethod
    def initialize_state_values(self, state_str: str) -> None:
        pass

    @abstractmethod
    def reset_eligibilities(self) -> None:
        pass

    @abstractmethod
    def set_eligibility(self, state: UniversalState, value: float) -> None:
        pass

    @abstractmethod
    def decay_eligibilies(self):
        pass
