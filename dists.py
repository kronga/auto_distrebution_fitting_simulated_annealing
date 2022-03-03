# library imports
from scipy import stats
from random import randint, choice

# project imports
from algo_state import AlgoState, DistType


class Dists:
    """
    A list of distributions with their parameters
    """

    # CONSTS #
    MIN_TEMP_TO_SWAP_DISTS = 50  # TODO: play with this number later
    PROBABILITY_TO_MOVE_DIST = 0.5

    DIST_COUNT = 2
    other_dists = {DistType.Uniform: [DistType.NORMAL],
                   DistType.NORMAL: [DistType.Uniform]}
    param_num = {DistType.Uniform: 1,
                 DistType.NORMAL: 2}

    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def random():
        """
        generates a random distribution with its properties
        """
        # TODO: the magic numbers in this method can be driven from the data with some logic for better results
        # random dist
        dist_pick = randint(1, Dists.DIST_COUNT)
        if dist_pick == DistType.Uniform:
            return AlgoState(dist_type=DistType.Uniform,
                             parameters=[randint(0, 100)])
        elif dist_pick == DistType.NORMAL:
            return AlgoState(dist_type=DistType.NORMAL,
                             parameters=[randint(0, 100), randint(0, 10)])
        else:
            raise Exception("We do not support in this type of distribution")

    @staticmethod
    def copy(state: AlgoState):
        """
        Copy a algo_state instance to new instance
        """
        return AlgoState(dist_type=state.dist_type,
                         parameters=state.parameters.copy())

    @staticmethod
    def to_string(state: AlgoState):
        """
        convert the distribution (e.g., algo_state instance) to the right string representation
        """
        if state.dist_type == DistType.Uniform:
            return "<Uniform: {}>".format(state.parameters[0])
        elif state.dist_type == DistType.NORMAL:
            return "<Normal: mean = {}, std = {}>".format(state.parameters[0],
                                                          state.parameters[1])
        else:
            raise Exception("We do not support in this type of distribution")

    @staticmethod
    def fitness(data: list,
                state: AlgoState):
        """
        Check how well a data is fitted in the state
        """
        if state.dist_type == DistType.Uniform:
            return stats.kstest(data, 'uniform')[
                1]  # the p-value of the Kolmogorov–Smirnov test for uniform distribution
        elif state.dist_type == DistType.NORMAL:
            return stats.kstest(data, 'norm', args=state.parameters)[
                1]  # the p-value of how we sure this is a normal dist
        else:
            raise Exception("We do not support in this type of distribution")

    @staticmethod
    def move(state: AlgoState,
             temperature: float):
        """
        find a state around the current state such that the temperature is now too much
        """
        # check if we want to try another dist
        if temperature > Dists.MIN_TEMP_TO_SWAP_DISTS and randint(0, 100) / 100 < Dists.PROBABILITY_TO_MOVE_DIST:
            if state.dist_type == DistType.Uniform:
                return AlgoState(choice(Dists.other_dists[state.dist_type]),
                                 parameters=[randint(0, 100), randint(0, 10)])
            elif state.dist_type == DistType.NORMAL:
                return AlgoState(choice(Dists.other_dists[state.dist_type]), parameters=[randint(0, 100)])
            else:
                raise Exception("We do not support in this type of distribution")
        else:
            # play along the parameters of this distribution
            if state.dist_type == DistType.Uniform:
                return AlgoState(dist_type=DistType.Uniform,
                                 parameters=[randint(0, 100)])
            elif state.dist_type == DistType.NORMAL:
                return AlgoState(dist_type=DistType.NORMAL,
                                 parameters=[randint(0, 100), randint(0, 10)])
            else:
                raise Exception("We do not support in this type of distribution")
