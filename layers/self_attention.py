from typing import Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mixer_conv_block import MixerBlock


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        assert temperature > 0, "temperature should be positive"
        assert attn_dropout >= 0, "dropout rate should be non-negative"
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else None

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)
        diag_mask = torch.triu(torch.ones_like(attn), diagonal=1)
        attn = attn.masked_fill(diag_mask == 1, -1e9)
        attn = F.softmax(attn, dim=-1)
        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention_ablatuon(nn.Module):
    def __init__(
            self,
            n_heads: int,
            d_model: int,
            d_k: int,
            d_v: int,
            dropout: float,
            attn_dropout: float,
            period: int
    ):
        super().__init__()
        self.period = period
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, int(n_heads * d_k / period), bias=False)
        self.w_ks = nn.Linear(d_model, int(n_heads * d_k / period), bias=False)
        self.w_vs = nn.Linear(d_model, int(n_heads * d_v / period), bias=False)
        self.attention = ScaledDotProductAttention(d_k ** 0.5, attn_dropout)
        self.fc = nn.Linear(int(n_heads * d_v / period), d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_mixer = MixerBlock(tokens_mlp_dim=400, channels_mlp_dim=d_model, tokens_hidden_dim=256,
                                      channels_hidden_dim=1024)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_steps = q.size(0), q.size(1)
        residual = q
        q = self.w_qs(q).view(batch_size, n_steps, self.n_heads, self.d_k)
        k = self.w_ks(k).view(batch_size, n_steps, self.n_heads, self.d_k)
        v = self.w_vs(v).view(batch_size, n_steps, self.n_heads, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
        v, attn_weights = self.attention(q, k, v, attn_mask)
        v = v.transpose(1, 2).contiguous().view(batch_size, n_steps, self.period, -1)
        v = self.fc(v)
        v = self.dropout(v)
        v += residual
        v = self.layer_norm(v)
        return v, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads: int,
            d_model: int,
            d_k: int,
            d_v: int,
            dropout: float,
            attn_dropout: float,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.attention = ScaledDotProductAttention(d_k ** 0.5, attn_dropout)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_mixer = MixerBlock(tokens_mlp_dim=400, channels_mlp_dim=d_model, tokens_hidden_dim=256,
                                      channels_hidden_dim=1024)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_steps = q.size(0), q.size(1)
        residual = q
        q = self.w_qs(q).view(batch_size, n_steps, self.n_heads, self.d_k)
        k = self.w_ks(k).view(batch_size, n_steps, self.n_heads, self.d_k)
        v = self.w_vs(v).view(batch_size, n_steps, self.n_heads, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
        v, attn_weights = self.attention(q, k, v, attn_mask)
        v = v.transpose(1, 2).contiguous().view(batch_size, n_steps, -1)
        v = self.fc(v)
        v = self.dropout(v)
        v += residual
        v = self.layer_norm(v)
        return v, attn_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in: int, d_hid: int, dropout: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_in, d_hid)
        self.linear_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.layer_mixer = MixerBlock(tokens_mlp_dim=400, channels_mlp_dim=d_in, tokens_hidden_dim=256,
                                      channels_hidden_dim=1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class EncoderLayer_ablation(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_inner: int,
            n_heads: int,
            d_k: int,
            d_v: int,
            dropout: float = 0.1,
            attn_dropout: float = 0.1,
            period: int = 1
    ):
        super().__init__()
        self.slf_attn = MultiHeadAttention_ablatuon(
            n_heads, d_model, d_k, d_v, dropout, attn_dropout, period
        )
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(
            self,
            enc_input: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_output, attn_weights = self.slf_attn(
            enc_input,
            enc_input,
            enc_input,
            attn_mask=src_mask,
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights


class EncoderLayer(nn.Module):

    def __init__(
            self,
            d_model: int,
            d_inner: int,
            n_heads: int,
            d_k: int,
            d_v: int,
            dropout: float = 0.1,
            attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            n_heads, d_model, d_k, d_v, dropout, attn_dropout
        )
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(
            self,
            enc_input: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_output, attn_weights = self.slf_attn(
            enc_input,
            enc_input,
            enc_input,
            attn_mask=src_mask,
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights


class DecoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_inner: int,
            n_heads: int,
            d_k: int,
            d_v: int,
            dropout: float = 0.1,
            attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            n_heads, d_model, d_k, d_v, dropout, attn_dropout
        )
        self.enc_attn = MultiHeadAttention(
            n_heads, d_model, d_k, d_v, dropout, attn_dropout
        )
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(
            self,
            dec_input: torch.Tensor,
            enc_output: torch.Tensor,
            slf_attn_mask: Optional[torch.Tensor] = None,
            dec_enc_attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, attn_mask=slf_attn_mask
        )
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, attn_mask=dec_enc_attn_mask
        )
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


class Encoder(nn.Module):
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
    ):
        super().__init__()
        self.embedding = nn.Linear(n_features, d_model)
        self.dropout = nn.Dropout(dropout)
        self.position_enc = PositionalEncoding(d_model, n_position=n_steps)
        self.enc_layer_stack = nn.ModuleList(
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

    def forward(
            self,
            x: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        x = self.embedding(x)
        enc_output = self.dropout(self.position_enc(x))
        attn_weights_collector = []
        for layer in self.enc_layer_stack:
            enc_output, attn_weights = layer(enc_output, src_mask)
            attn_weights_collector.append(attn_weights)
        if return_attn_weights:
            return enc_output, attn_weights_collector
        return enc_output


class Decoder(nn.Module):
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
    ):
        super().__init__()
        self.embedding = nn.Linear(n_features, d_model)
        self.dropout = nn.Dropout(dropout)
        self.position_enc = PositionalEncoding(d_model, n_position=n_steps)
        self.layer_stack = nn.ModuleList(
            [
                DecoderLayer(
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

    def forward(
            self,
            trg_seq: torch.Tensor,
            enc_output: torch.Tensor,
            trg_mask: Optional[torch.Tensor] = None,
            src_mask: Optional[torch.Tensor] = None,
            return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list, list]]:
        trg_seq = self.embedding(trg_seq)
        dec_output = self.dropout(self.position_enc(trg_seq))
        dec_slf_attn_collector = []
        dec_enc_attn_collector = []
        for layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = layer(
                dec_output,
                enc_output,
                slf_attn_mask=trg_mask,
                dec_enc_attn_mask=src_mask,
            )
            dec_slf_attn_collector.append(dec_slf_attn)
            dec_enc_attn_collector.append(dec_enc_attn)
        if return_attn_weights:
            return dec_output, dec_slf_attn_collector, dec_enc_attn_collector

        return dec_output


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid: int, n_position: int = 200):
        super().__init__()

        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    @staticmethod
    def _get_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_table[:, : x.size(1)].clone().detach()
