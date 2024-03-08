# Copyright (C) 2024 - 2024 Ruojian Li， Inc. All Rights Reserved
# @Time    : 2024/3/8 21:30
# @Email   : ruojian_li@163.com
# License: MIT
from typing import Tuple, Optional, Callable, Union
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.mixer_conv_block import TimesBlock
from layers.self_attention import EncoderLayer, PositionalEncoding
from pypots.data import BaseDataset
from pypots.imputation.base import BaseNNImputer
from pypots.optim import Adam
from pypots.optim.base import Optimizer
from pypots.utils.metrics import cal_mae
from torch.utils.data import DataLoader

from module.data import DatasetForSAITS


class MAITTS(BaseNNImputer):
    def __init__(
            self,
            n_steps: int,
            n_features: int,
            n_layers: int,
            n_mixer_layers,
            d_model: int,
            d_inner: int,
            n_heads: int,
            d_k: int,
            d_v: int,
            dropout: float = 0,
            attn_dropout: float = 0,
            diagonal_attention_mask: bool = True,
            ORT_weight: int = 1,
            MIT_weight: int = 1,
            batch_size: int = 32,
            epochs: int = 100,
            patience: Optional[int] = None,
            customized_loss_func: Callable = cal_mae,
            optimizer: Optional[Optimizer] = Adam(),
            num_workers: int = 0,
            device: Optional[Union[str, torch.device, list]] = None,
            saving_path: Optional[str] = None,
            model_saving_strategy: Optional[str] = "best",
    ):
        super().__init__(
            batch_size,
            epochs,
            patience,
            num_workers,
            device,
            saving_path,
            model_saving_strategy,
        )
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.diagonal_attention_mask = diagonal_attention_mask
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.n_mixer_layers = n_mixer_layers
        self.model = _MAITTS(
            self.n_layers,
            self.n_steps,
            self.n_features,
            self.d_model,
            self.d_inner,
            self.n_heads,
            self.d_k,
            self.d_v,
            self.dropout,
            self.attn_dropout,
            self.n_mixer_layers,
            self.diagonal_attention_mask,
            self.ORT_weight,
            self.MIT_weight,
        )
        self._print_model_size()
        self._send_model_to_given_device()
        self.customized_loss_func = customized_loss_func
        self.optimizer = optimizer
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        (
            indices,
            X_intact,
            X,
            missing_mask,
            indicating_mask,
        ) = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "X_intact": X_intact,
            "missing_mask": missing_mask,
            "indicating_mask": indicating_mask,
        }

        return inputs

    def _assemble_input_for_validating(self, data) -> dict:
        indices, X, missing_mask = self._send_data_to_given_device(data)
        inputs = {
            "X": X,
            "missing_mask": missing_mask,
        }
        return inputs

    def _assemble_input_for_testing(self, data) -> dict:
        return self._assemble_input_for_validating(data)

    def fit(
            self,
            train_set: Union[dict, str],
            val_set: Optional[Union[dict, str]] = None,
            file_type: str = "h5py",
    ) -> None:
        training_set = DatasetForSAITS(
            train_set, return_labels=False, file_type=file_type
        )
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if val_set is not None:
            if isinstance(val_set, str):
                with h5py.File(val_set, "r") as hf:
                    val_set = {
                        "X": hf["X"][:],
                        "X_intact": hf["X_intact"][:],
                        "indicating_mask": hf["indicating_mask"][:],
                    }
            val_set = BaseDataset(val_set, return_labels=False, file_type=file_type)
            val_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        self._train_model(training_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()
        self._auto_save_model_if_necessary(training_finished=True)

    def predict(
            self,
            test_set: Union[dict, str],
            file_type: str = "h5py",
            diagonal_attention_mask: bool = True,
    ) -> dict:
        self.model.eval()
        test_set = BaseDataset(test_set, return_labels=False, file_type=file_type)
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        imputation_collector = []
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                results = self.model.forward(
                    inputs, diagonal_attention_mask, training=False
                )
                imputed_data = results["imputed_data"]
                imputation_collector.append(imputed_data)
        imputation = torch.cat(imputation_collector).cpu().detach().numpy()
        result_dict = {
            "imputation": imputation,
        }
        return result_dict

    def impute(
            self,
            X: Union[dict, str],
            file_type="h5py",
    ) -> np.ndarray:
        results_dict = self.predict(X, file_type=file_type)
        return results_dict["imputation"]


class _MAITTS(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_steps: int,
            n_features: int,
            d_model: int,
            d_inner: int,
            n_heads: int,
            d_k: int,
            d_v: int,
            dropout: float,
            attn_dropout: float,
            n_mixer_layers,
            diagonal_attention_mask: bool = True,
            ORT_weight: float = 1,
            MIT_weight: float = 1,
            customized_loss_func: Callable = cal_mae,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_steps = n_steps
        actual_n_features = n_features * 2
        self.diagonal_attention_mask = diagonal_attention_mask
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.customized_loss_func = customized_loss_func

        self.timeblock_for_first_block = nn.ModuleList(
            [
                TimesBlock(n_features, n_steps)
                for _ in range(n_mixer_layers)
            ]
        )
        self.layer_stack_for_second_block = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    d_inner,
                    n_heads,
                    d_k,
                    d_v,
                    dropout,
                    attn_dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.position_enc = PositionalEncoding(d_model, n_position=n_steps)
        self.embedding_2 = nn.Linear(actual_n_features, d_model)
        self.reduce_dim_beta = nn.Linear(d_model, n_features)
        self.reduce_dim_gamma = nn.Linear(n_features, n_features)
        self.weight_combine = nn.Linear(n_features + n_steps, n_features)

    def _process(
            self,
            inputs: dict,
    ) -> Tuple[torch.Tensor, list]:
        X, masks = inputs["X"], inputs["missing_mask"]
        input_X_for_first = X
        for timeblcok in self.timeblock_for_first_block:
            input_X_for_first = timeblcok(input_X_for_first)
        X_tilde_1 = input_X_for_first
        X_prime = masks * X + (1 - masks) * input_X_for_first
        input_X_for_second = torch.cat([X_prime, masks], dim=2)
        input_X_for_second = self.embedding_2(input_X_for_second)
        enc_output = self.position_enc(
            input_X_for_second
        )
        attn_weights = None
        for encoder_layer in self.layer_stack_for_second_block:
            enc_output, attn_weights = encoder_layer(enc_output)
        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))
        attn_weights = attn_weights.squeeze(dim=1)
        if len(attn_weights.shape) == 4:
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)
        combining_weights = torch.sigmoid(
            self.weight_combine(torch.cat([masks, attn_weights], dim=2))
        )
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        X_c = masks * X + (1 - masks) * X_tilde_3
        return X_c, [X_tilde_1, X_tilde_2, X_tilde_3]

    def forward(
            self, inputs: dict, diagonal_attention_mask: bool = False, training: bool = True
    ) -> dict:
        X, masks = inputs["X"], inputs["missing_mask"]
        imputed_data, [X_tilde_1, X_tilde_2, X_tilde_3] = self._process(
            inputs
        )
        if not training:
            return {
                "imputed_data": imputed_data,
            }
        ORT_loss = 0
        ORT_loss += self.customized_loss_func(X_tilde_1, X, masks)
        ORT_loss += self.customized_loss_func(X_tilde_2, X, masks)
        ORT_loss += self.customized_loss_func(X_tilde_3, X, masks)
        ORT_loss /= 3
        MIT_loss = self.customized_loss_func(
            X_tilde_3, inputs["X_intact"], inputs["indicating_mask"]
        )
        loss = self.ORT_weight * ORT_loss + self.MIT_weight * MIT_loss
        results = {
            "imputed_data": imputed_data,
            "ORT_loss": ORT_loss,
            "MIT_loss": MIT_loss,
            "loss": loss,
        }
        return results
