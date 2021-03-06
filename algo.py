# library imports
import math
import pandas as pd
from timeit import timeit
from pandas.api.types import is_numeric_dtype

# project imports
from dists import Dists
from algo_state import AlgoState
from smart_gap_filler import SmartGapFiller


class Algo:
    """
    An algorithm that run on each column in a DF and allocates the best distribution (with parameters) for it
    using the simulated annealing algorithm
    """

    # COSNTS #
    MIN_TEMPERATURE = 10
    MAX_TEMPERATURE = 20000

    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def run(df: pd.DataFrame,
            fix_gaps: bool,
            k: int,
            iters: int = 300,
            debug: bool = False):
        """
        A single entry point for the algorithm
        :param df: the DF we would like to fill
        :param fix_gaps: run or not the gap filling algorithm
        :param k: the number of similar rows to take for the "smart fix gaps" algorithm
        :param iters: the number of iterations we allow for the simulated annealing algorithm
        :return: enrgies of best states,
                 times of performance,
                 a dict (json) object with keys as the names of the features and representation of the dists to each one


        """
        # if asked, fill gaps. Otherwise, clear rows with nulls
        if fix_gaps:
            df = SmartGapFiller.run(df=df,
                                    k=k)
        else:
            df.dropna(axis=0, inplace=True)
        # return answer
        times = []
        answers = []
        col_dic = {}
        for col in list(df):
            if is_numeric_dtype(df[col]):
                start = timeit()
                energy, result = Algo._feature_run(data=df[col].tolist(),
                                  iters=iters,
                                  debug=debug)
                answers.append(energy)
                col_dic[col] = result
                end = timeit()
                times.append(end - start)
        return answers, times, col_dic

    @staticmethod
    def _feature_run(data: list,
                     iters: int,
                     debug: bool = False):
        """
        Run the algorithm for a single feature (as a list object).
        It minimizes the energy of a system by simulated annealing.
        Parameters
        state : an initial arrangement of the system
        Returns
        (state, energy): the best state and energy found.
        :param data: the data to fit
        :param iters: the number of iterations we allow for the simulated annealing algorithm
        :return: energy of the final state, the final state
        """
        # technical vars
        step = 0
        temp_factor = -math.log(Algo.MAX_TEMPERATURE / Algo.MIN_TEMPERATURE)

        # initial state
        state = Dists.random()
        energy = Algo._energy(data=data,
                              state=state)
        # just to debug
        if debug:
            print("{}: {} with E = {:.3f}".format(step, state, energy))
        best_state = Dists.copy(state)
        best_energy = energy

        # Run as much as allowed
        while step < iters:
            # count this step
            step += 1
            # compute new temperature
            temp = Algo.MAX_TEMPERATURE * math.exp(temp_factor * step / iters)
            new_state = Algo._move(state=state,
                                   temperature=temp)
            new_energy = Algo._energy(data=data,
                                      state=new_state)
            delta_energy = new_energy - energy
            if delta_energy < 0:  # and math.exp(-delta_energy / temp) < random.random():
                # Accept new state and compare to best state
                state = Dists.copy(state=new_state)
                energy = new_energy
                if energy < best_energy:
                    best_state = Dists.copy(state=state)
                    best_energy = energy

            # just to debug
            if debug:
                print("{}: {} with E = {:.3f}".format(step, state, energy))
        # return answer
        return energy, Dists.to_string(state=best_state)

    @staticmethod
    def _energy(data: list,
                state: AlgoState):
        """
        computes the fitness of the state given the data
        """
        return Dists.fitness(data=data,
                             state=state)

    @staticmethod
    def _move(state: AlgoState,
              temperature: float):
        """
        A small state change to compute the differential of the current state
        """
        return Dists.move(state=state,
                          temperature=temperature)
