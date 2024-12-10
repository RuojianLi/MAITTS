import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def FFT_for_Period(x, k=3):
    
    xf = torch.fft.rfft(x, dim=1)
    
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = np.sort(top_list.detach().cpu().numpy())
    period = x.shape[1] // top_list
    return [400, 200, 100], abs(xf).mean(-1)[:, top_list]

class TimesBlock(nn.Module):
    def __init__(self, n_features, n_steps):
        super(TimesBlock, self).__init__()
        self.seq_len = n_steps
        self.pred_len = 0
        self.k = 3

        self.separable_blocks = nn.ModuleList([Separable_Block(400, n_features),
                                               Separable_Block(200, n_features),
                                               Separable_Block(100, n_features)])

    def forward(self, x):

        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()  
            out = self.separable_blocks[i](out)

            
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)  
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x
        return res


class MlpBlock1(nn.Module):
    def __init__(self, input_dim, mlp_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(0.9)

    def forward(self, x):
        return self.dropout1(self.gelu(self.fc1(x)))


class MlpBlock2(nn.Module):
    def __init__(self, input_dim, mlp_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(0.9)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.dropout2 = nn.Dropout(0.9)

    def forward(self, x):
        return self.dropout2(self.fc2(self.dropout1(self.gelu(self.fc1(x)))))


class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, tokens_hidden_dim, channels_hidden_dim):
        super().__init__()
        self.ln = nn.LayerNorm(tokens_mlp_dim)
        self.tokens_mlp_block = MlpBlock1(tokens_mlp_dim, mlp_dim=tokens_hidden_dim)
        self.channels_mlp_block = MlpBlock2(channels_mlp_dim, mlp_dim=channels_hidden_dim)

    def forward(self, x):
        y = self.ln(x)
        y = y.transpose(1, 2)  
        y = self.channels_mlp_block(y)
        y = y.transpose(1, 2)  
        out = x + y  
        y = self.ln(out)  
        y = out + self.tokens_mlp_block(y)  
        return y


class Separable_Block(nn.Module):
    def __init__(self, tokens_mlp_dims, n_features):
        super().__init__()
        self.channels_mlp_dim = n_features
        self.mixerblock = MixerBlock(tokens_mlp_dims, self.channels_mlp_dim, 1024, 512)
        self.pointwise = nn.Conv2d(self.channels_mlp_dim, self.channels_mlp_dim, kernel_size=1)

    def forward(self, x):
        mixer_outputs = [self.mixerblock(x[:, :, i, :]) for i in range(x.shape[2])]
        mixer_outputs = torch.stack(mixer_outputs, dim=2)
        y = mixer_outputs
        return y
