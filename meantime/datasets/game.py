from .base import AbstractDataset

import pandas as pd


class GameDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'game'

    @classmethod
    def is_zipfile(cls):
        return False

    @classmethod
    def url(cls):
        return 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Video_Games.csv'

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.csv')
        df = pd.read_csv(file_path, header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
