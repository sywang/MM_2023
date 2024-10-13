import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def intersect_series_with_index(index: pd.Index, data: pd.Series) -> pd.Series:
    """
    returns series intersected with given index
    """
    aligned_series = data.loc[index.intersection(data.index)]
    return aligned_series

def intersect_df(dataframes=()):
    """
    returns all dataframe/series-s with intersected indexes in the same order
    :param dataframes: list of pd.DataFrame/pd.Series
    """
    shared_indices = set(dataframes[0].index)
    for df_index in range(1, len(dataframes)):
        shared_indices = shared_indices.intersection(dataframes[df_index].index)
    return [dataframes[df_index].loc[list(shared_indices)] for df_index in range(len(dataframes))]

def assign_quantiles(data, quantiles=[0.5]):
    """
    annotates samples of a numeric series by its quantile
    :param data: pd.Series with numeric-like data
    :param quantiles: quantiles marks to associate samples
    """
    quantiles = sorted(set([0] + quantiles + [1]))
    quantile_values = data.quantile(quantiles)
    labels = [f'{lower}q<x<={upper}q' for lower, upper in zip(quantiles[:-1], quantiles[1:])]
    return pd.cut(data, bins=quantile_values, labels=labels, include_lowest=True)