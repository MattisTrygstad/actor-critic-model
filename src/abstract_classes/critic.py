from abc import ABC, abstractmethod


class Critic(ABC):
    """
    Abstract class to be implemented by each agent
    required tables.
    """

    @abstractmethod
    def method(self):
        pass
