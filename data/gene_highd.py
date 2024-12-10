# Copyright (C) 2024 - 2024 Ruojian Li， Inc. All Rights Reserved
# @Time    : 2024/3/8 21:30
# @Email   : ruojian_li@163.com
# License: MIT
import torch
from pypots.data import masked_fill, mcar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


def gene_highd(path, artificially_missing_rate: float = 0.1):
    assert (
            0 <= artificially_missing_rate < 1
    ), "artificially_missing_rate must be in [0,1)"


    with open(
            path,
            "rb") as f:
        dataset = pickle.load(f)
    X = dataset

    all_recordID = X["RecordID"].unique()
    train_set_ids, test_set_ids = train_test_split(all_recordID, test_size=0.2)
    train_set_ids, val_set_ids = train_test_split(train_set_ids, test_size=0.2)
    train_set_ids.sort()
    val_set_ids.sort()
    test_set_ids.sort()

    train_set = X[X["RecordID"].isin(train_set_ids)].sort_values(["RecordID", "Time"])
    val_set = X[X["RecordID"].isin(val_set_ids)].sort_values(["RecordID", "Time"])
    test_set = X[X["RecordID"].isin(test_set_ids)].sort_values(["RecordID", "Time"])

    train_set = train_set.drop(["RecordID", "Time"], axis=1)
    val_set = val_set.drop(["RecordID", "Time"], axis=1)
    test_set = test_set.drop(["RecordID", "Time"], axis=1)

    train_X, val_X, test_X = (
        train_set.to_numpy(),
        val_set.to_numpy(),
        test_set.to_numpy(),
    )

    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)

    train_X = train_X.reshape(len(train_set_ids), -1, 96)
    val_X = val_X.reshape(len(val_set_ids), -1, 96)
    test_X = test_X.reshape(len(test_set_ids), -1, 96)

    data = {
        "n_classes": 2,
        "n_steps": 400,
        "n_features": train_X.shape[-1],
        "train_X": train_X,
        "val_X": val_X,
        "test_X": test_X,
        "scaler": scaler,
    }

    if artificially_missing_rate > 0:
        val_X_intact, val_X, val_X_missing_mask, val_X_indicating_mask = mcar(
            val_X, artificially_missing_rate
        )
        val_X = masked_fill(val_X, 1 - val_X_missing_mask, torch.nan)
        test_X_intact, test_X, test_X_missing_mask, test_X_indicating_mask = mcar(
            test_X, artificially_missing_rate
        )
        test_X = masked_fill(test_X, 1 - test_X_missing_mask, torch.nan)
        data["val_X"] = val_X
        data["test_X"] = test_X
        data["test_X_intact"] = test_X_intact
        data["test_X_indicating_mask"] = test_X_indicating_mask
        data["val_X_intact"] = val_X_intact
        data["val_X_indicating_mask"] = val_X_indicating_mask
    return data
