# library imports
from enum import Enum

# project imports


class DistType(Enum):
    """
    Enum for dist types
    """

    Uniform = 1
    NORMAL = 2

# TODO: parameters and mapper functions between distributions and something we can work with


class AlgoState:
    """
    A helper class holding a dist and its parameters to represent a state in the distribution space
    """

    def __init__(self,
                 dist_type: DistType,
                 parameters: list):
        self.dist_type = dist_type
        self.parameters = parameters
