# library imports
import os
import json
import pandas as pd
from glob import glob

# project imports
from algo import Algo
import brute_force_fitter


class Main:
    """

    """

    # CONSTS #
    DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "data")
    RESULTS_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "results")
    TOP_K_ALGO = 5
    DEBUG_FLAG = True
    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def run_example():
        """
        This assumes we have a "/data/*.csv" files and run the smart gap filler + distribution fitting algorithm
        :return: a list of JSONs as the number of data files with the respective columns and the fitted ddistribution
        """
        # make sure we have the "results" folder
        try:
            os.mkdir(Main.RESULTS_FOLDER_PATH)
        except:
            pass
        # get data and run all
        for data_file_path in glob(os.path.join(Main.DATA_FOLDER_PATH, "*.csv")):
            df = pd.read_csv(data_file_path)
            answer = Algo.run(df=df,
                              fix_gaps=False,
                              k=Main.TOP_K_ALGO,
                              debug=Main.DEBUG_FLAG)
            with open(os.path.join(Main.RESULTS_FOLDER_PATH, os.path.basename(data_file_path).replace(".csv", ".json")), "w") as answer_file:
                json.dump(answer, answer_file)


if __name__ == '__main__':
    Main.run_example()
