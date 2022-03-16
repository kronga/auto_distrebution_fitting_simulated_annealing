# library imports
from enum import IntEnum

# project imports


class DistType(IntEnum):
    """
    Enum for dist types
    """

    Uniform = 1
    NORMAL = 2
    EXP = 3
    WEIBULL_MIN = 4


class AlgoState:
    """
    A helper class holding a dist and its parameters to represent a state in the distribution space
    """

    def __init__(self,
                 dist_type: DistType,
                 parameters: list):
        self.dist_type = dist_type
        self.parameters = parameters

    def __repr__(self):
        return "<State: type = {}, parms = {}>".format(self.dist_type,
                                                       self.parameters)

    def __str__(self):
        return self.__repr__()
