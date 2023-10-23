import pandas as pd
from pathlib import Path
import numpy as np
import subprocess
import os
import argparse

from generate_data_for_training import StandardScaler


def create_h5(dataset: str, filename: str) -> pd.DataFrame:
    # Read the csv file
    csv_path = Path(dataset, filename)
    df = pd.read_csv(csv_path)

    # Set the index
    df.set_index("Unnamed: 0", inplace=True)
    df.index.name = "Time"
    df.index = pd.to_datetime(df.index)

    # Save the dataframe as a h5 file
    df.to_hdf(Path(dataset, f"{dataset}_his_all.h5"), key="t", mode="w")
    print(df.shape)

    return df


def generate_train_val_test(args, filename: str) -> None:
    data = np.load(Path(args.dataset, filename))["data"]

    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
    y_offsets = np.arange(1, (seq_length_y + 1), 1)

    # data, idx = generate_data_and_idx(df, x_offsets, y_offsets, args.tod, args.dow)
    num_samples, num_nodes, _ = data.shape
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    print("idx min & max:", min_t, max_t)
    idx = np.arange(min_t, max_t, 1)

    print("final data shape:", data.shape, "idx shape:", idx.shape)

    num_samples = len(idx)
    num_train = round(num_samples * 0.6)
    num_val = round(num_samples * 0.2)

    # split idx
    idx_train = idx[:num_train]
    idx_val = idx[num_train : num_train + num_val]
    idx_test = idx[num_train + num_val :]

    # normalize
    x_train = data[: idx_val[0] - args.seq_length_x, :, 0]
    scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
    data[..., 0] = scaler.transform(data[..., 0])

    # save
    out_dir = Path(args.dataset, args.years)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "his.npz", data=data, mean=scaler.mean, std=scaler.std
    )

    np.save(out_dir / "idx_train", idx_train)
    np.save(out_dir / "idx_val", idx_val)
    np.save(out_dir / "idx_test", idx_test)


if __name__ == "__main__":
    # METR_LA
    print("### METR_LA ###")
    df = create_h5("metr_la", "METR-LA.csv")
    subprocess.run(
        "python generate_data_for_training.py --dataset metr_la --years all", shell=True
    )

    # PEMS_BAY
    print("### PEMS_BAY ###")
    df = create_h5("pems_bay", "PEMS-BAY.csv")
    subprocess.run(
        "python generate_data_for_training.py --dataset pems_bay --years all",
        shell=True,
    )

    # PEMS04
    print("### PEMS04 ###")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="pems04", help="dataset name")
    parser.add_argument("--years", type=str, default="all", help="just set all here")
    parser.add_argument("--seq_length_x", type=int, default=12, help="sequence Length")
    parser.add_argument("--seq_length_y", type=int, default=12, help="sequence Length")
    parser.add_argument("--tod", type=int, default=1, help="time of day")
    parser.add_argument("--dow", type=int, default=1, help="day of week")
    args = parser.parse_args()
    generate_train_val_test(args, "PEMS04.npz")

    # PEMS04
    print("### PEMS08 ###")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="pems08", help="dataset name")
    parser.add_argument("--years", type=str, default="all", help="just set all here")
    parser.add_argument("--seq_length_x", type=int, default=12, help="sequence Length")
    parser.add_argument("--seq_length_y", type=int, default=12, help="sequence Length")
    parser.add_argument("--tod", type=int, default=1, help="time of day")
    parser.add_argument("--dow", type=int, default=1, help="day of week")
    args = parser.parse_args()
    generate_train_val_test(args, "PEMS08.npz")

    print("@@@ Finished! @@@")
