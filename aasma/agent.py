import numpy as np
from abc import ABC, abstractmethod


class Agent(ABC):

    """
    Base agent class.
    Represents the concept of an autonomous agent.

    Attributes
    ----------
    name: str
        Name for identification purposes.
        
    observation: np.ndarray
       The most recent observation of the environment


    Methods
    -------
    see(observation)
        Collects an observation

    action(): int
        Abstract method.
        Returns an action, represented by an integer
        May take into account the observation (numpy.ndarray).

    References
    ----------
    ..[1] Michael Wooldridge "An Introduction to MultiAgent Systems - Second
    Edition", John Wiley & Sons, p 44.


    """

    def __init__(self, name: str):
        self.name = name
        self.observation = None

    def see(self, observation: np.ndarray):
        self.observation = observation

    @abstractmethod
    def action(self) -> int:
        raise NotImplementedError()
