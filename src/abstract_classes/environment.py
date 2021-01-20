from abc import ABC, abstractmethod
from environment.state import State


class Environment(ABC):
    """
    Abstract class to be implemented by each environment
    required tables.
    """

    @abstractmethod
    def execute_action(self, action: tuple) -> None:
        pass

    @abstractmethod
    def undo_action(self) -> None:
        pass

    @abstractmethod
    def get_legal_actions(self) -> list:
        pass

    @abstractmethod
    def check_win_condition(self) -> bool:
        pass

    @abstractmethod
    def get_state(self) -> State:
        pass

    @abstractmethod
    def reset(self) -> State:
        pass

    @abstractmethod
    def visualize(self) -> None:
        pass
