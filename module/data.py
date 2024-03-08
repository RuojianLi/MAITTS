# Copyright (C) 2024 - 2024 Ruojian Li， Inc. All Rights Reserved
# @Time    : 2024/3/8 21:30
# @Email   : ruojian_li@163.com
# License: MIT
from typing import Union, Iterable
import torch
from pycorruptor import mcar
from pypots.data import BaseDataset


class DatasetForSAITS(BaseDataset):
    def __init__(
        self,
        data: Union[dict, str],
        return_labels: bool = True,
        file_type: str = "h5py",
        rate: float = 0.2,
    ):
        super().__init__(data, return_labels, file_type)
        self.rate = rate

    def _fetch_data_from_array(self, idx: int) -> Iterable:
        X = self.X[idx].to(torch.float32)
        X_intact, X, missing_mask, indicating_mask = mcar(X, rate=self.rate)

        sample = [
            torch.tensor(idx),
            X_intact,
            X,
            missing_mask,
            indicating_mask,
        ]

        if self.y is not None and self.return_labels:
            sample.append(self.y[idx].to(torch.long))

        return sample

    def _fetch_data_from_file(self, idx: int) -> Iterable:
        if self.file_handle is None:
            self.file_handle = self._open_file_handle()

        X = torch.from_numpy(self.file_handle["X"][idx]).to(torch.float32)
        X_intact, X, missing_mask, indicating_mask = mcar(X, rate=self.rate)

        sample = [
            torch.tensor(idx),
            X_intact,
            X,
            missing_mask,
            indicating_mask,
        ]
        if "y" in self.file_handle.keys() and self.return_labels:
            sample.append(torch.tensor(self.file_handle["y"][idx], dtype=torch.long))

        return sample
