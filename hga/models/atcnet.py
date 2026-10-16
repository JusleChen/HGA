from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F


def _glorot_(weight: torch.Tensor, fan_in: int, fan_out: int) -> None:
    bound = (6.0 / float(fan_in + fan_out)) ** 0.5
    nn.init.uniform_(weight, -bound, bound)


def _max_norm_(weight: torch.Tensor, dims: tuple[int, ...], limit: float) -> None:
    with torch.no_grad():
        norm = torch.linalg.vector_norm(weight, ord=2, dim=dims, keepdim=True)
        # Exact Keras MaxNorm update, including its epsilon below the limit.
        weight.mul_(norm.clamp(max=limit) / (norm + 1e-7))


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding_left = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=0, bias=True
        )
        nn.init.kaiming_uniform_(self.conv.weight, a=0.0, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.padding_left, 0)))


class TemporalResidualBlock(nn.Module):
    def __init__(self, dilation: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = CausalConv1d(32, 32, kernel_size=4, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(32, eps=1e-3, momentum=0.01)
        self.conv2 = CausalConv1d(32, 32, kernel_size=4, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(32, eps=1e-3, momentum=0.01)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(F.elu(self.bn1(self.conv1(x))))
        x = self.dropout(F.elu(self.bn2(self.conv2(x))))
        return F.elu(x + residual)


class ATCWindowAttention(nn.Module):
    """PyTorch port of the official ATCNet MHA block for the CUDA-only pilot."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(32, eps=1e-6)
        # Official Keras MHA has 2 heads with key_dim=8: Q/K/V width is 16,
        # while the output projection returns the 32-channel residual width.
        self.query = nn.Linear(32, 16, bias=True)
        self.key = nn.Linear(32, 16, bias=True)
        self.value = nn.Linear(32, 16, bias=True)
        self.output = nn.Linear(16, 32, bias=True)
        self.attention_dropout = nn.Dropout(0.5)
        self.residual_dropout = nn.Dropout(0.3)
        for layer in (self.query, self.key, self.value, self.output):
            _glorot_(layer.weight, layer.in_features, layer.out_features)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.shape
        normalized = self.norm(x)
        query = self.query(normalized).reshape(batch, steps, 2, 8).transpose(1, 2)
        key = self.key(normalized).reshape(batch, steps, 2, 8).transpose(1, 2)
        value = self.value(normalized).reshape(batch, steps, 2, 8).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * (8.0 ** -0.5)
        weights = self.attention_dropout(torch.softmax(scores, dim=-1))
        attended = torch.matmul(weights, value).transpose(1, 2).contiguous().reshape(batch, steps, 16)
        return x + self.residual_dropout(self.output(attended))


class ATCWindow(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = ATCWindowAttention()
        self.tcn1 = TemporalResidualBlock(dilation=1)
        self.tcn2 = TemporalResidualBlock(dilation=2)
        self.classifier = nn.Linear(32, 4, bias=True)
        _glorot_(self.classifier.weight, 32, 4)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.tcn1(x.transpose(1, 2))
        x = self.tcn2(x).transpose(1, 2)
        return self.classifier(x[:, -1, :])


class ATCNetTorch(nn.Module):
    """Faithful CUDA PyTorch port of the pinned official ATCNet_ defaults."""

    def __init__(self, n_classes: int = 4, in_chans: int = 22, in_samples: int = 1125):
        super().__init__()
        if (n_classes, in_chans, in_samples) != (4, 22, 1125):
            raise ValueError("SG-HRE ATCNet is locked to the official 4-class, 22x1125 AxxT input")
        # Keras input [N,1,channel,time] is transposed to channels-last
        # [N,time,channel,1] by Permute((3,2,1)). These NCHW layers preserve it.
        self.temporal_conv = nn.Conv2d(1, 16, kernel_size=(64, 1), padding="same", bias=False)
        self.temporal_bn = nn.BatchNorm2d(16, eps=1e-3, momentum=0.01)
        _glorot_(self.temporal_conv.weight, 64, 16 * 64)

        self.depthwise_conv = nn.Conv2d(
            16, 32, kernel_size=(1, 22), groups=16, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(32, eps=1e-3, momentum=0.01)
        # Keras depthwise kernel shape [1,22,16,2] uses fan-in=352, fan-out=44.
        _glorot_(self.depthwise_conv.weight, 22 * 16, 22 * 2)

        self.spatial_conv = nn.Conv2d(32, 32, kernel_size=(16, 1), padding="same", bias=False)
        self.spatial_bn = nn.BatchNorm2d(32, eps=1e-3, momentum=0.01)
        _glorot_(self.spatial_conv.weight, 16 * 32, 16 * 32)
        self.eeg_dropout1 = nn.Dropout(0.3)
        self.eeg_dropout2 = nn.Dropout(0.3)
        self.windows = nn.ModuleList([ATCWindow() for _ in range(5)])

        self._regularized: list[tuple[nn.Parameter, float]] = [
            (self.temporal_conv.weight, 0.009),
            (self.depthwise_conv.weight, 0.009),
            (self.spatial_conv.weight, 0.009),
        ]
        for window in self.windows:
            for tcn in (window.tcn1, window.tcn2):
                for conv in (tcn.conv1, tcn.conv2):
                    self._regularized.append((conv.weight, 0.009))
            self._regularized.append((window.classifier.weight, 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (1, 22, 1125):
            raise ValueError(f"Expected ATCNet input [batch,1,22,1125], got {tuple(x.shape)}")
        x = x.permute(0, 1, 3, 2).contiguous()  # [N,1,time,channel]
        x = self.temporal_bn(self.temporal_conv(x))
        x = self.depthwise_bn(self.depthwise_conv(x))
        x = self.eeg_dropout1(F.avg_pool2d(F.elu(x), kernel_size=(8, 1), stride=(8, 1)))
        x = F.elu(self.spatial_bn(self.spatial_conv(x)))
        x = self.eeg_dropout2(F.avg_pool2d(x, kernel_size=(7, 1), stride=(7, 1)))
        sequence = x.squeeze(-1).transpose(1, 2).contiguous()
        if sequence.shape[1:] != (20, 32):
            raise RuntimeError(f"Official ATCNet convolution block shape changed: {tuple(sequence.shape)}")
        logits = [window(sequence[:, start:start + 16]) for start, window in enumerate(self.windows)]
        return torch.stack(logits, dim=0).mean(dim=0)

    def regularization_loss(self) -> torch.Tensor:
        return sum((coefficient * weight.square().sum() for weight, coefficient in self._regularized))

    def project_constraints_(self) -> None:
        _max_norm_(self.temporal_conv.weight, (1, 2, 3), 0.6)
        depth = self.depthwise_conv.weight.reshape(16, 2, 1, 1, 22)
        _max_norm_(depth, (0, 2, 3, 4), 0.6)
        _max_norm_(self.spatial_conv.weight, (1, 2, 3), 0.6)
        for window in self.windows:
            for tcn in (window.tcn1, window.tcn2):
                for conv in (tcn.conv1, tcn.conv2):
                    _max_norm_(conv.weight, (1, 2), 0.6)


def trainable_cuda_parameters(model: nn.Module, device: torch.device) -> bool:
    parameters: Iterable[nn.Parameter] = (parameter for parameter in model.parameters() if parameter.requires_grad)
    return all(parameter.device == device and parameter.is_cuda for parameter in parameters)
