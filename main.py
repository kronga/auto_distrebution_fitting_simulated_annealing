# library imports
import os
import json
import pandas as pd
import numpy as np
import random

# project imports
from algo import Algo
from naive import NaiveApproach


class Main:
    """

    """

    # CONSTS #
    DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "data")
    RESULTS_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "results")
    TOP_K_ALGO = 5
    DEBUG_FLAG = False

    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def run_example():
        """
        This assumes we have a "/data/*.csv" files and run the smart gap filler + distribution fitting algorithm
        :return: a list of JSONs as the number of data files with the respective columns and the fitted distributions
        """
        # make sure we have the "results" and "data" folder
        try:
            os.mkdir(Main.RESULTS_FOLDER_PATH)
        except:
            pass
        try:
            os.mkdir(Main.DATA_FOLDER_PATH)
        except:
            pass
        for p_normal in [0, 0.5, 1]:
            # generate data
            datasets = [Main.create_test_data(df_index=i,
                                              p_normal=p_normal) for i in range(10)]
            # save datasets for later comparison
            [df.to_csv(os.path.join(Main.DATA_FOLDER_PATH, "df_{}.csv".format(i)), index=False) for i, df in
             enumerate(datasets)]
            # run our approach and compute performance and time
            our_times = []
            naive_times = []
            our_scores = []
            naive_scores = []
            for index, df in enumerate(datasets):
                answer_sa, times_sa, results_dic = Algo.run(df=df,
                                               fix_gaps=False,
                                               k=Main.TOP_K_ALGO,
                                               debug=Main.DEBUG_FLAG)
                answer_naive, times_naive = NaiveApproach.run(df=df)
                # record
                our_times.extend(times_sa)
                naive_times.extend(times_naive)
                our_scores.extend(answer_sa)
                naive_scores.extend(answer_naive)
                print("Finish with dataset {}".format(index))
                with open(os.path.join(Main.RESULTS_FOLDER_PATH,"df_"+str(index)+ ".json"), "w") as answer_file:
                    json.dump(results_dic, answer_file)
            # summarize the results
            print("Answer for p={} for normal feature".format(p_normal))
            print("\n\nOur:\nPerformance = {:.6f} +- {:.6f},"
                  "Time = {:.6f} +- {:.6f}\n\n"
                  "Naive:\nPerformance = {:.6f} +- {:.6f},"
                  "Time = {:.6f} +- {:.6f}\n\n".format(np.nanmean(our_scores),
                                                       np.nanstd(our_scores),
                                                       np.nanmean(our_times),
                                                       np.nanstd(our_times),
                                                       np.nanmean(naive_scores),
                                                       np.nanstd(naive_scores),
                                                       np.nanmean(naive_times),
                                                       np.nanstd(naive_times)))


    @staticmethod
    def create_test_data(df_index: int,
                         p_normal: float):
        """
        Building the datasets for evaluating the method of simulated anealing
        """
        # size of dataset
        row_num = random.randint(100, 10000)
        col_num = random.randint(5, 15)
        # init meta data for later
        to_json_data = []
        # generate dataset
        full_df = pd.DataFrame()
        for i in range(col_num):
            if random.random() < p_normal:
                mean = random.uniform(-100, 100)
                std = random.uniform(1, 10)
                new_col = np.random.normal(mean, std, row_num)
                full_df[i] = new_col
                to_json_data.append(["normal", mean, std])
            else:
                const = random.uniform(-100, 100)
                new_col = np.asarray([const for _ in range(row_num)])
                full_df[i] = new_col
                to_json_data.append(["uniform", const])
        # save meta-data
        with open(os.path.join(Main.DATA_FOLDER_PATH, "df_meta_{}.csv".format(df_index)), "w") as json_file:
            json.dump(to_json_data,
                      json_file,
                      indent=2)
        # return the dataset
        return full_df


if __name__ == '__main__':
    Main.run_example()
