#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd
import pickle

from typing import Tuple, List
from typing import List
from dataclasses import dataclass
from pyod.models.hbos import HBOS


@dataclass
class CustomParameters:
    tol: float = 3.0
    random_state: int = 42
    target_channels: List[str] = None
    target_channel_indices: List[int] = None  # do not use, automatically handled


class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)

# functions should preserve the order provided in target_channels
def load_data(config: AlgorithmArgs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"Loading: {config.dataInput}")

    columns, data_columns, anomaly_columns = get_columns_info(config.dataInput)
    dataset = read_dataset(config.dataInput, data_columns, anomaly_columns)

    target_channels = get_valid_channels(config.customParameters.target_channels, data_columns)
    config.customParameters.target_channels = target_channels

    anomaly_cols = [f"is_anomaly_{ch}" for ch in target_channels]

    dataset = handle_anomaly_columns(dataset, anomaly_columns, anomaly_cols)
    dataset = dataset.loc[:, target_channels + anomaly_cols]

    data = dataset[target_channels].to_numpy()
    labels = dataset[anomaly_cols].to_numpy()

    return normalize_and_return(data, labels, config, config.modelOutput)


def get_columns_info(filepath: str) -> tuple[list[str], list[str], list[str]]:
    columns = pd.read_csv(filepath, index_col="timestamp", nrows=0).columns.tolist()
    anomaly_cols = [col for col in columns if col.startswith("is_anomaly")]
    data_cols = columns[:-len(anomaly_cols)] if anomaly_cols else columns
    return columns, data_cols, anomaly_cols


def read_dataset(filepath: str, data_cols: list[str], anomaly_cols: list[str]) -> pd.DataFrame:
    dtypes = {col: np.float32 for col in data_cols}
    dtypes.update({col: np.uint8 for col in anomaly_cols})
    return pd.read_csv(filepath, index_col="timestamp", parse_dates=True, dtype=dtypes)


def get_valid_channels(raw_channels: list[str], data_cols: list[str]) -> list[str]:
    if not raw_channels:
        print(f"No target_channels provided. Using all data columns: {data_cols}")
        return data_cols

    seen = set()
    valid = [ch for ch in raw_channels if ch in data_cols and not (ch in seen or seen.add(ch))]
    if not valid:
        print(f"No valid target channels found in dataset, falling back to all data columns.")
        return data_cols
    return valid


def handle_anomaly_columns(dataset: pd.DataFrame, anomaly_cols: list[str],
                           target_anomaly_cols: list[str]) -> pd.DataFrame:
    if len(anomaly_cols) == 1 and anomaly_cols[0] == "is_anomaly":
        for col in target_anomaly_cols:
            dataset[col] = dataset["is_anomaly"]
        dataset = dataset.drop(columns="is_anomaly")
    return dataset


def normalize_and_return(data: np.ndarray, labels: np.ndarray, config: AlgorithmArgs, model_output_path: str):
    means_path = str(model_output_path) + ".means.txt"
    stds_path = str(model_output_path) + ".stds.txt"

    if config.executionType == "train":
        means = [np.mean(data[:, i][labels[:, i] == 0]) for i in range(data.shape[1])]
        stds = [np.std(data[:, i][labels[:, i] == 0]) for i in range(data.shape[1])]
        stds = np.where(np.asarray(stds) == 0, 1, stds)

        np.savetxt(means_path, means)
        np.savetxt(stds_path, stds)

    elif config.executionType == "execute":
        means = np.atleast_1d(np.loadtxt(means_path))
        stds = np.atleast_1d(np.loadtxt(stds_path))

    return data, means, stds



# alphabetically sorted order of the channels
def get_columns_and_types(file_path: str) -> Tuple[List[str], List[str], dict]:
    columns = pd.read_csv(file_path, index_col="timestamp", nrows=0).columns.tolist()
    anomaly_columns = [col for col in columns if col.startswith("is_anomaly")]
    data_columns = columns[:-len(anomaly_columns)] if anomaly_columns else columns

    dtypes = {col: np.float32 for col in data_columns}
    dtypes.update({col: np.uint8 for col in anomaly_columns})

    return data_columns, anomaly_columns, dtypes


def load_dataset(file_path: str, dtypes: dict) -> pd.DataFrame:
    return pd.read_csv(file_path, index_col="timestamp", parse_dates=True, dtype=dtypes)


def filter_target_channels(raw_targets: List[str], data_columns: List[str]) -> List[str]:
    seen = set()
    filtered = [ch for ch in raw_targets if ch in data_columns and not (ch in seen or seen.add(ch))]
    return sorted(filtered) if filtered else sorted(data_columns)


def handle_global_anomaly_column(dataset: pd.DataFrame, anomaly_columns: List[str], target_channels: List[str]) -> pd.DataFrame:
    if len(anomaly_columns) == 1 and anomaly_columns[0] == "is_anomaly":
        for ch in target_channels:
            dataset[f"is_anomaly_{ch}"] = dataset["is_anomaly"]
        dataset = dataset.drop(columns="is_anomaly")
    return dataset


def compute_means_stds(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    means = [np.mean(data[:, i][labels[:, i] == 0]) for i in range(data.shape[1])]
    stds = [np.std(data[:, i][labels[:, i] == 0]) for i in range(data.shape[1])]
    stds = np.where(stds == 0, 1, stds)
    return np.asarray(means), np.asarray(stds)


def load_data(config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"Loading: {config.dataInput}")

    data_columns, anomaly_columns, dtypes = get_columns_and_types(config.dataInput)
    dataset = load_dataset(config.dataInput, dtypes)

    raw_targets = config.customParameters.target_channels or []
    filtered_targets = filter_target_channels(raw_targets, data_columns)

    if not raw_targets or not filtered_targets:
        print(f"Input channels not given or not present in the data, selecting all channels: {filtered_targets}")

    config.customParameters.target_channels = filtered_targets

    dataset = handle_global_anomaly_column(dataset, anomaly_columns, filtered_targets)

    all_anomaly_columns = [f"is_anomaly_{ch}" for ch in filtered_targets]
    dataset = dataset.loc[:, filtered_targets + all_anomaly_columns]

    labels = dataset[all_anomaly_columns].to_numpy()
    data = dataset[filtered_targets].to_numpy()

    means_path = f"{config.modelOutput}.means.txt"
    stds_path = f"{config.modelOutput}.stds.txt"

    if config.executionType == "train":
        means, stds = compute_means_stds(data, labels)
        np.savetxt(means_path, means)
        np.savetxt(stds_path, stds)
    elif config.executionType == "execute":
        means = np.atleast_1d(np.loadtxt(means_path))
        stds = np.atleast_1d(np.loadtxt(stds_path))
    else:
        raise ValueError(f"Unknown executionType: {config.executionType}")

    return data, means, stds


# original function load_data

# def load_data(config: AlgorithmArgs) -> np.ndarray:
#     print(f"Loading: {config.dataInput}")
#     columns = pd.read_csv(config.dataInput, index_col="timestamp", nrows=0).columns.tolist()
#     anomaly_columns = [x for x in columns if x.startswith("is_anomaly")]
#     data_columns = columns[:-len(anomaly_columns)]
#
#     dtypes = {col: np.float32 for col in data_columns}
#     dtypes.update({col: np.uint8 for col in anomaly_columns})
#     dataset = pd.read_csv(config.dataInput, index_col="timestamp", parse_dates=True, dtype=dtypes)
#
#     if config.customParameters.target_channels is None or len(
#             set(config.customParameters.target_channels).intersection(data_columns)) == 0:
#         config.customParameters.target_channels = data_columns
#         print(
#             f"Input channels not given or not present in the data, selecting all the channels: {config.customParameters.target_channels}")
#         all_used_channels = [x for x in data_columns if x in set(config.customParameters.target_channels)]
#         all_used_anomaly_columns = [f"is_anomaly_{channel}" for channel in all_used_channels]
#     else:
#         config.customParameters.target_channels = [x for x in config.customParameters.target_channels if x in data_columns]
#
#         # Remove unused columns from dataset
#         all_used_channels = [x for x in data_columns if x in set(config.customParameters.target_channels)]
#         all_used_anomaly_columns = [f"is_anomaly_{channel}" for channel in all_used_channels]
#         if len(anomaly_columns) == 1 and anomaly_columns[0] == "is_anomaly":  # Handle datasets with only one global is_anomaly column
#             for c in all_used_anomaly_columns:
#                 dataset[c] = dataset["is_anomaly"]
#             dataset = dataset.drop(columns="is_anomaly")
#         dataset = dataset.loc[:, all_used_channels + all_used_anomaly_columns]
#
#     labels = dataset[all_used_anomaly_columns].to_numpy()
#     dataset = dataset[all_used_channels].to_numpy()
#     meansOutput = str(config.modelOutput) + ".means.txt"
#     stdsOutput = str(config.modelOutput) + ".stds.txt"
#     if config.executionType == "train":
#         train_means = [np.mean(dataset[:, i][labels[:, i] == 0]) for i in range(dataset.shape[-1])]
#         np.savetxt(meansOutput, train_means)
#
#         train_stds = [np.std(dataset[:, i][labels[:, i] == 0].astype(float)) for i in range(dataset.shape[-1])]
#         train_stds = np.asarray(train_stds)
#         train_stds = np.where(train_stds == 0, 1, train_stds)  # do not divide constant signals by zero
#         np.savetxt(stdsOutput, train_stds)
#     elif config.executionType == "execute":
#         train_means = np.atleast_1d(np.loadtxt(meansOutput))
#         train_stds = np.atleast_1d(np.loadtxt(stdsOutput))
#
#
#     return dataset, train_means, train_stds



def train(config: AlgorithmArgs):
    load_data(config)  # generate train means and stds


def execute(config: AlgorithmArgs):
    data, train_means, train_stds = load_data(config)

    scores = ((data > train_means + config.customParameters.tol * train_stds) |
              (data < train_means - config.customParameters.tol * train_stds)).astype(np.uint8)
    np.savetxt(config.dataOutput, scores, delimiter=",")


if __name__ == "__main__":

    config = AlgorithmArgs.from_sys_args()
    print(f"Config: {config}")

    if config.executionType == "train":
        train(config)
    elif config.executionType == "execute":
        execute(config)
    else:
        raise ValueError(f"Unknown execution type '{config.executionType}'; expected either 'train' or 'execute'!")
