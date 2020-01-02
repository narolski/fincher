import pandas as pd
from fastai.text import *


class FilmwebPreprocessor:
    """
    FilmwebPreprocessor performs pre-processing of a raw Filmweb+ dataset, returning ready-to-use databunches of
    chosen properties.
    """
    def __init__(self, csv_path: Path):
        """
        Initializes the FilmwebPreprocessor.
        :param csv_path: path to a CSV file containing the Filmweb+ dataset
        """
        self.dataset = pd.read_csv(csv_path)
        self.__preprocess()

    def __preprocess(self):
        """Performs a pre-processing of the dataset."""
        self.dataset.drop_duplicates(subset="review", inplace=True, keep="first")
        self.dataset['review'] = self.dataset['review'].astype(str)
        self.dataset = self.dataset[self.dataset['review'].apply(lambda row: len(row) > 300)]

    def get_entire_dataset(self) -> pd.DataFrame:
        """
        Returns the contents of an entire Filmweb+ dataset.
        :return: pd.DataFrame object containing a dataset
        """
        return self.dataset

    def get_labelled_train_test_split_dataset(self) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Returns the dataset split into training set and a test set.
        :return: tuple of format (training pd.DataFrame, test pd.DataFrame)
        """
        negative = self.dataset.loc[self.dataset['sentiment'] == 0]
        positive = self.dataset.loc[self.dataset['sentiment'] == 2]
        sample_pos = positive.sample(n=len(negative), random_state=42)

        split_idx = int(0.1 * len(negative))
        test_pos = sample_pos[0:split_idx]
        training_pos = sample_pos[split_idx:len(sample_pos)]
        test_neg = negative[0:split_idx]
        training_neg = negative[split_idx:len(negative)]

        training_df = pd.concat([training_pos, training_neg], sort=False)
        test_df = pd.concat([test_pos, test_neg], sort=False)

        return training_df, test_df


