# library imports
import numpy as np
from random import randint, choice
from sklearn.metrics import mean_absolute_error, mean_squared_error

# project imports
from algo_state import AlgoState, DistType


class Dists:
    """
    A list of distributions with their parameters
    """

    # CONSTS #
    MIN_TEMP_TO_SWAP_DISTS = 300
    PROBABILITY_TO_MOVE_DIST = 0.5

    DIST_COUNT = 2
    ALL_DISTS = [DistType.Uniform, DistType.NORMAL]
    PARAM_NUM = {DistType.Uniform: 1,
                 DistType.NORMAL: 2,
                 # DistType.EXP: 2,
                 # DistType.WEIBULL_MIN: 3
                 }

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
                             parameters=[np.random.normal(0, 3)])
        elif dist_pick == DistType.NORMAL:
            return AlgoState(dist_type=DistType.NORMAL,
                             parameters=[np.random.normal(0, 3), abs(np.random.normal(0, 3))])
        # elif dist_pick == DistType.EXP:
        #     return AlgoState(dist_type=DistType.EXP,
        #                      parameters=[np.random.normal(0, 3), abs(np.random.normal(0, 3))])
        # elif dist_pick == DistType.WEIBULL_MIN:
        #     return AlgoState(dist_type=DistType.WEIBULL_MIN,
        #                      parameters=[abs(np.random.normal(0, 3)), abs(np.random.normal(0, 3)),
        #                                  abs(np.random.normal(0, 3))])
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
            return "<Uniform: loc = {}>".format(state.parameters[0])
        elif state.dist_type == DistType.NORMAL:
            return "<Normal: mean = {}, std = {}>".format(state.parameters[0],
                                                          state.parameters[1])
        # elif state.dist_type == DistType.EXP:
        #     return "<Expon: loc = {}, scale = {}>".format(state.parameters[0],
        #                                                  state.parameters[1])
        # elif state.dist_type == DistType.WEIBULL_MIN:
        #     return "<Weibull_min: c = {}, loc = {}, scale = {}>".format(state.parameters[0],
        #                                                                 state.parameters[1],
        #                                                                 state.parameters[2])
        else:
            raise Exception("We do not support in this type of distribution")

    @staticmethod
    def fitness(data: list,
                state: AlgoState):
        """
        Check how well a data is fitted in the state
        """
        if state.dist_type == DistType.Uniform:
            return mean_absolute_error(y_true=data, y_pred=[state.parameters[0] for _ in range(len(data))])

            # return stats.kstest(data, 'uniform')[0]  # the p-value of the Kolmogorov–Smirnov test for uniform distribution
        elif state.dist_type == DistType.NORMAL:
            data_mean = np.mean(data)
            data_std = np.std(data)
            return mean_squared_error([data_mean, data_std], state.parameters)

            # y_pred = np.random.normal(stsate.parameters[0], state.parameters[1], len(data))
            # return mean_squared_error(y_true=data, y_pred=y_pred) + mean_absolute_error(y_true=data, y_pred=y_pred)

        #     return stats.kstest(data, 'norm', args=state.parameters)[0]  # the p-value of how we sure this is a normal dist
        # elif state.dist_type == DistType.EXP:
        #     return stats.kstest(data, 'expon', args=state.parameters)[0]
        # elif state.dist_type == DistType.WEIBULL_MIN:
        #     return stats.kstest(data, 'weibull_min', args=state.parameters)[0]
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
            # get the list of dists we can chose from
            remain_dists = Dists.ALL_DISTS.copy()
            remain_dists.remove(state.dist_type)
            # chose dist to move to
            chosen_dist = choice(remain_dists)
            # pick number of params
            parameters = [parm_value + np.random.normal(0, 0.05) for parm_value in state.parameters]
            if len(parameters) < Dists.PARAM_NUM[chosen_dist]:
                parameters.extend([np.random.normal(1, 0.1) for _ in range(Dists.PARAM_NUM[chosen_dist] - len(parameters))])
            else:
                parameters = parameters[:Dists.PARAM_NUM[chosen_dist]]

            # edge case
            if chosen_dist == DistType.NORMAL and parameters[1] < 0:
                parameters[1] = parameters[1] * -1

            return AlgoState(dist_type=chosen_dist,
                             parameters=parameters)
        else:
            # play along the parameters of this distribution
            if state.dist_type == DistType.Uniform:
                return AlgoState(dist_type=DistType.Uniform,
                                 parameters=[state.parameters[0] + np.random.normal(0, 2)])
            elif state.dist_type == DistType.NORMAL:
                std = state.parameters[1] + np.random.normal(0, 0.05)
                return AlgoState(dist_type=DistType.NORMAL,
                                 parameters=[state.parameters[0] + np.random.normal(0, 0.5),
                                             std if std > 0 else std * -1])
            # elif state.dist_type == DistType.EXP:
            #     scale = state.parameters[1] + np.random.normal(0, 0.05)
            #     return AlgoState(dist_type=DistType.NORMAL,
            #                      parameters=[state.parameters[0] + np.random.normal(0, 0.5),
            #                                  scale if scale > 0 else scale * -1])
            # elif state.dist_type == DistType.WEIBULL_MIN:
            #     scale = state.parameters[1] + np.random.normal(0, 0.05)
            #     return AlgoState(dist_type=DistType.NORMAL,
            #                      parameters=[state.parameters[0] + np.random.normal(0, 0.5),
            #                                  scale if scale > 0 else scale * -1])
            else:
                raise Exception("We do not support in this type of distribution")
