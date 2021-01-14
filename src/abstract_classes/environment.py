from abc import ABC, abstractmethod


class Environment(ABC):
    """
    Abstract class to be implemented by each environment
    required tables.
    """

    @abstractmethod
    def method(self):
        pass
