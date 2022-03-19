# library imports
import os
import json
import pandas as pd
from glob import glob
import numpy as np
import random

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
    def finalData(num_dfs):
        df_list = []
        for _ in range(num_dfs):
            df_list.append(Main.createTestData())
        [df.to_csv(os.path.join(Main.DATA_FOLDER_PATH, "df_{}.csv".format(i)), index=False) for i, df in enumerate(df_list)]

    @staticmethod
    def createTestData():
        """
        Building the datasets for evaluating the method of simulated anealing 
        """
        const = random.uniform(-10, 10)
        row_num = random.randint(100, 10000)
        col_num = random.randint(5, 15)
        full_df = pd.DataFrame([const for _ in range(row_num)])
        normal_idx = random.sample(range(col_num), col_num // 2)
        for i in range(col_num):
            if i in normal_idx:
                full_df[i] = pd.DataFrame(np.random.normal(random.uniform(-100, 100), random.uniform(0, 0), row_num))
            else:
                const = random.uniform(-10, 10)
                full_df[i] = pd.DataFrame([const for i in range(row_num)])
        return full_df

    @staticmethod
    def run_example():
        """
        This assumes we have a "/data/*.csv" files and run the smart gap filler + distribution fitting algorithm
        :return: a list of JSONs as the number of data files with the respective columns and the fitted distributions
        """
        # make sure we have the "results" folder
        try:
            os.mkdir(Main.RESULTS_FOLDER_PATH)
        except:
            pass
        # get data and run all
        for data_file_path in glob(os.path.join(Main.DATA_FOLDER_PATH, "*.csv")):
            df = pd.read_csv(data_file_path)
            # df = df.apply(pd.to_numeric)
            answer = Algo.run(df=df,
                              fix_gaps=False,
                              k=Main.TOP_K_ALGO,
                              debug=Main.DEBUG_FLAG)
            with open(os.path.join(Main.RESULTS_FOLDER_PATH, os.path.basename(data_file_path).replace(".csv", ".json")), "w") as answer_file:
                json.dump(answer, answer_file)


if __name__ == '__main__':
    # Main.run_example()
    Main.finalData(50)
