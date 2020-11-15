import os
import itertools
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

norm_max = 1400


def genereta_dataset(dataset_folder, periods=[]):
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

    data = data.fillna(method="ffill")

    t0 = datetime.strptime(data.index[0][0] + data.index[0][1], "%Y-%m-%d%H:%M:%S.%f")
    tf = datetime.strptime(data.index[-1][0] + data.index[-1][1], "%Y-%m-%d%H:%M:%S.%f")
    duration = tf - t0
    duration_seconds = duration.total_seconds()

    for timeseries_period in tqdm(periods, position=1):
        timestamps = [
            t0 + i * timedelta(milliseconds=timeseries_period)
            for i in range(int(duration_seconds * 1000 / timeseries_period))
        ]
        timeseries_data = pd.DataFrame(index=timestamps, columns=data.columns)
        i = 0
        prev_value = data.iloc[i]
        for ts in timestamps:
            for j in range(i, len(data)):
                row = data.iloc[j]
                row_t = datetime.strptime(
                    row.name[0] + row.name[1], "%Y-%m-%d%H:%M:%S.%f"
                )
                if row_t < ts:
                    prev_value = row.values
                    i += 1
                elif row_t == ts:
                    prev_value = row.values
                    i += 1
                    break
                elif row_t > ts:
                    break
            timeseries_data.loc[ts] = prev_value

        initial_zeros = 0
        for i in range(len(timeseries_data)):
            if np.sum(timeseries_data.iloc[i].values) > 0:
                break
            initial_zeros += 1

        final_zeros = 0
        for i in range(len(timeseries_data) - 1, -1, -1):
            if np.sum(timeseries_data.iloc[i].values) > 0:
                break
            final_zeros += 1

        timeseries_data = timeseries_data.iloc[initial_zeros:-final_zeros]

        timeseries_data = timeseries_data / norm_max

        timeseries_data.to_csv(
            os.path.join(
                dataset_folder,
                "../",
                os.path.basename(os.path.normpath(dataset_folder))
                + "_{}ms.csv".format(timeseries_period),
            )
        )

    data = data / norm_max
    data.to_csv(
        os.path.join(
            dataset_folder,
            "../",
            os.path.basename(os.path.normpath(dataset_folder)) + ".csv",
        )
    )


periods = [10, 100, 250, 500, 750, 1000]
clouds = ["Clear sky", "Overcast", "Variable", "Very variable"]
sites = ["Alderville", "Varennes"]

dataset_folders = [
    "../data/{}/{}/".format(x[0], x[1]) for x in itertools.product(clouds, sites)
]

for path in tqdm(dataset_folders, position=0):
    genereta_dataset(path, periods)
    break
