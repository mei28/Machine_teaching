import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

TRAIN_DATA = '../input/train.csv'
TEST_DATA = '../input/test.csv'


def read_csv(path, header=None):
    """read csv file, return data frame

    Parameters
    ----------
    path : path 
        It is where the data is

    Returns
    -------
    return pandas df
    """
    df = pd.read_csv(path, header=header)
    return df


def load_train_data():
    df = read_csv(TRAIN_DATA)
    return df


def load_test_data():
    df = read_csv(TEST_DATA)
    return df


def split_data(df):
    """
    split data frame into train and test
    Parameters
    ----------
    data : pandas
        data frame

    Returns
    -------
    pandas
    return train ans test data frame
    """
    X = df.drop('Spe', axis=1)
    y = df['Spe']
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, shuffle=True
    )
    train_X.reset_index(drop=True, inplace=True)
    train_y.reset_index(drop=True, inplace=True)
    test_X.reset_index(drop=True, inplace=True)
    test_y.reset_index(drop=True, inplace=True)

    return train_X, test_X, train_y, test_y


if __name__ == "__main__":
    df = read_csv('output/weebil_vespula.csv')
