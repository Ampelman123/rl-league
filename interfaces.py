from abc import ABC, abstractmethod


class LeagueAgent(ABC):
    """
    Interface that agents must implement to participate in the League.
    """

    @abstractmethod
    def save(self, path: str, **kwargs):
        """Save the agent's state to the specified path."""
        pass

    @abstractmethod
    def load(self, path: str, **kwargs):
        """Load the agent's state from the specified path."""
        pass

    @abstractmethod
    def act(self, obs):
        """Return an action for the given observation."""
        pass
