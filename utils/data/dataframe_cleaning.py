import numpy as np
import pandas as pd

def number_of_unique_values(array):
    return len(np.unique(array))


def is_binary_feature(feature):
    return number_of_unique_values(feature) == 2


def unique_values_per_feature(dataset: pd.DataFrame):
    n_unique_vals = []
    for i in dataset:
        feature = dataset[i]
        n_unique_vals.append(number_of_unique_values(feature))

    return n_unique_vals


def remove_single_value_features(dataset:pd.DataFrame, inplace=False):
    values_per_feature = unique_values_per_feature(dataset)
    single_value_features = [i for i,j in zip(dataset, values_per_feature) if j == 1]
    return dataset.drop(columns=single_value_features, inplace=inplace)


def classes_to_one_hot(y, n_classes=2):
    """Converts labels to one-hot classes"""
    y_one_hot = np.zeros((y.shape[0], n_classes))
    y_one_hot[np.arange(y.shape[0]), y] = 1
    return y_one_hot