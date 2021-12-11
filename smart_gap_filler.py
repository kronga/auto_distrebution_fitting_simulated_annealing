# library imports
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# project imports


class SmartGapFiller:
    """
    A KNN-based feature-oriented filler, taking the similar top 'k' rows without
    some feature to each row with this data and fill it.

    * This algorithm is problematic when the nulls are uniformly distributed over the dataset or concentrates on small number of rows\cols
    """

    def __init__(self):
        pass

    @staticmethod
    def run(df: pd.DataFrame,
            k: int = 5):
        """
        A single entry point for the algorithm, fill the nulls in the DF column-size
        using a KNN algorithm on the other values fine by them
        :param df: the DF we would like to fill
        :param k: the number of similar rows to take for the average
        :return: a filled DF
        """
        # first, drop all lines with more than 2 nulls
        new_df = df.dropna(axis='rows', thresh=2)
        # find the columns that has at least one row with nulls
        cols_with_nulls = df.columns[df.isna().any()].tolist()
        for col in cols_with_nulls:
            # get sub data
            sub_new_df = new_df.drop([col], axis=1)
            # handle only rows with full data
            sub_new_df.dropna(inplace=True)
            # get KNN instance
            knn = NearestNeighbors(n_neighbors=k)
            knn.fit(sub_new_df.values)
            # query each row with null in the original dataframe
            for row_index, row in new_df.iteritems():
                # TODO: make sure this is the right method
                if row[col].isna():
                    # fix row
                    del row[col]
                    # get answer
                    similar_rows = knn.kneighbors([row],
                                                  return_distance=False)
                    # get indexes
                    similar_rows_indices = list(sub_new_df.index[similar_rows])
                    # get the average value of these values in the index and allocate back to the df
                    new_df.loc[row_index, col] = np.nanmean(new_df.loc[similar_rows_indices, col])
        return new_df
