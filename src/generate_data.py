import os
import itertools
import json
import numpy as np
import pandas as pd
import traces
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from tqdm import tqdm


def read_data(dataset_folder):
    """Reads and concat all the csv files from a folder.

    Args:
        dataset_folder (str): Data folder's path
    
    Returns:
        pandas.DataFrame: solar irradiance dataset.
    """
    files = sorted(
        [
            file
            for (_, _, files) in os.walk(dataset_folder)
            for file in files
            if ".csv" in file and not file.startswith(".")
        ]
    )

    data = None
    for file in files:
        file_df = pd.read_csv(os.path.join(dataset_folder, file))
        file_df = file_df[file_df.columns[[0, 1, 2]]]
        file_df.columns = ["date", "timestamp", file.split("_")[-1].split(".")[0]]
        file_df.index = [file_df.pop(c) for c in ["date", "timestamp"]]

        if data is None:
            data = file_df
        else:
            data = pd.concat([data, file_df], axis=1, sort=False)
    print(dataset_folder)
    data.index = pd.to_datetime(
        data.index.get_level_values(0).astype(str).values
        + data.index.get_level_values(1).astype(str).values,
        format="%Y-%m-%d%H:%M:%S.%f",
    )
    data = data.fillna(method="ffill")
    return data


def separate_evenly(data, freq):
    """Samples an uneven time series in an evenly separated time series.

    Args:
        data (pandas.DataFrame): uneven time series
        freq (int): sample frequency in ms

    Returns:
        pd.DataFrame: New evenly separated dataframe.
    """
    t0 = data.index.min()
    tf = data.index.max()

    even_index = pd.date_range(t0, tf, freq="{}ms".format(freq))
    even_data = pd.DataFrame(index=even_index)

    for c in data.columns:
        ts = traces.TimeSeries(data[c].to_dict())
        even_data[c] = [ts.get(t) for t in even_index]

    return even_data


def remove_initial_final_zeros(data):
    """Filters out all-zeros rows from the begining and end of the time series.

    Args:
        data (pandas.DataFrame): Solar irradiance data

    Returns:
        pandas.DataFrame: filtered data
    """
    initial_zeros = 0
    for i in range(len(data)):
        if np.sum(data.iloc[i].values) > 0:
            break
        initial_zeros += 1

    final_zeros = 0
    for i in range(len(data) - 1, -1, -1):
        if np.sum(data.iloc[i].values) > 0:
            break
        final_zeros += 1

    return data.iloc[initial_zeros:-final_zeros]


def normalize(data, max_value=None):
    """Scales the solar irradiance data.

    Args:
        data (pandas.DataFrame): Data
        max_value (int, optional): Parameter for scaling the data. If None, it is
            computed from the data. Defaults to None.

    Returns:
        pandas.DataFrame: scaled data
    """
    if max_value is None:
        max_value = data.values.max()
    norm_params = {"min": 0.0, "max": max_value}

    return data / max_value, norm_params


def write_data_csv(data, dataset_folder, freq):
    filename = os.path.join(
        dataset_folder,
        "../",
        "{}_{}ms.csv".format(os.path.basename(os.path.normpath(dataset_folder)), freq),
    )
    data.to_csv(filename)


def write_norm_params(params, dataset_folder, freq):
    filename = os.path.join(
        dataset_folder,
        "../",
        "{}_{}ms_params.json".format(
            os.path.basename(os.path.normpath(dataset_folder)), freq
        ),
    )
    with open(filename, "w") as outfile:
        json.dump(params, outfile)


def genereta_dataset(dataset_folder, periods=[], norm_max=None):
    uneven_data = read_data(dataset_folder)
    for freq in periods:
        data = separate_evenly(uneven_data, freq)
        data = remove_initial_final_zeros(data)
        data, norm_params = normalize(data, norm_max)
        write_data_csv(data, dataset_folder, freq)
        write_norm_params(norm_params, dataset_folder, freq)


if __name__ == "__main__":

    periods = [100, 500, 1000]  # in ms
    clouds = ["Clear sky", "Overcast", "Variable", "Very variable"]
    sites = ["Alderville", "Varennes"]
    norm_max = None

    dataset_folders = [
        "../data/{}/{}/".format(x[0], x[1]) for x in itertools.product(clouds, sites)
    ]

    for path in tqdm(dataset_folders, position=0):
        genereta_dataset(path, periods, norm_max)