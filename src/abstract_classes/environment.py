from abc import ABC, abstractmethod

from enums import Action
from environment.state import State


class Environment(ABC):
    """
    Abstract class to be implemented by each environment
    required tables.
    """

    @abstractmethod
    def execute_action(self, action: Action) -> State:
        pass

    def reverse_action() -> State:
        pass

    @abstractmethod
    def get_legal_moves(self) -> list:
        pass

    @abstractmethod
    def get_state(self) -> State:
        pass

    @abstractmethod
    def reset(self) -> State:
        pass

    @abstractmethod
    def visualize_board() -> None:
        pass
