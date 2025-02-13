import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Callable, Optional, Union
import math
from .blocks import (
    LayerNorm,
    LinearNorm,
    BatchNorm1dTBC,
    CustomSwish,
    MultiheadAttention
)

class EmotionLinear(nn.Linear):
    def __init__(self, emotion_features: int, in_features: int, out_features: int,
                 bias: bool = True, device=None, dtype=None):
        super().__init__(in_features+emotion_features, out_features, bias, device, dtype)
        self.emotion_features = emotion_features
        self.in_features = in_features
        self.out_features = out_features


    def forward(self, emotion: Tensor, input: Tensor) -> Tensor:
        emotion = emotion.view(-1,*[1 for _ in range(len(input.shape[1:-1]))],
                               self.emotion_features).tile((1, *input.shape[1:-1], 1))
        input = torch.concat([emotion,
                              input], -1)
        return super().forward(input)


class EmotionConv1d(nn.Conv1d):
    def __init__(self, emotion_features: int, in_channels: int, out_channels: int, kernel_size: Union[int, tuple],
                 stride: Optional[Union[int, tuple]] = 1, padding: Optional[Union[int, tuple]] = 0,
                 dilation: Optional[Union[int, tuple]] = 1, groups: Optional[int] = 1,
                 bias: bool = True, device=None, dtype=None):
        super().__init__(in_channels+emotion_features, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                         device=device, dtype=dtype)
        self.emotion_features = emotion_features
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, emotion: Tensor, input: Tensor) -> Tensor:
        emotion = emotion.unsqueeze(-1).tile((1,1,input.shape[-1]))
        input = torch.concat([emotion, input], -2)
        return super().forward(input)

class EmotionMish(nn.Module):
    def forward(self,c, x):
        return x * torch.tanh(F.softplus(x))

class EmotionRelu(nn.Module):
    def forward(self,c, x):
        return torch.nn.functional.relu(x)

def same_padding(kernel_size: int, stride: int, dilation: int = 1) -> int:
    """
    Calculate the padding required for a same convolution.

    Args:
    - kernel_size (int): The size of the kernel.
    - stride (int): The stride of the convolution.
    - dilation (int, optional): The dilation rate of the convolution. Defaults to 1.

    Returns:
    - int: The padding required for a same convolution.
    """
    if stride == 1 and dilation == 1:
        # For stride 1 and dilation 1, the padding is (kernel_size - 1) / 2
        padding = (kernel_size - 1) // 2
    else:
        # For stride > 1 or dilation > 1, the padding is calculated based on the formula:
        # padding = ((kernel_size - 1) * dilation - 1) / 2
        padding = ((kernel_size - 1) * dilation - 1) // 2

    return padding
class EmotionConvNorm(nn.Module):
    def __init__(
            self,
            emotion_features,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=None,
            dilation=1,
            bias=True,
            w_init_gain="linear",
    ):
        super(EmotionConvNorm, self).__init__()

        if padding is None:
            padding = same_padding(kernel_size, stride, dilation)

        self.conv = EmotionConv1d(
            emotion_features,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,)

    def forward(self, emotion, signal):
        return self.conv(emotion, signal)

class EmotionTransformerFFNLayer(nn.Module):
    def __init__(self,emotion_features, hidden_size, filter_size, padding="SAME", kernel_size=1, dropout=0., act="gelu"):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.act = act
        if padding == "SAME":
            self.ffn_1 = EmotionConv1d(emotion_features, hidden_size, filter_size, kernel_size, padding=kernel_size // 2)
        elif padding == "LEFT":
            self.padding = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
            self.ffn_1 = EmotionConv1d(emotion_features, hidden_size, filter_size, kernel_size)
        self.ffn_2 = EmotionLinear(emotion_features, filter_size, hidden_size)
        if self.act == "swish":
            self.swish_fn = CustomSwish()

    def forward(self,emotion, x, incremental_state=None):
        # x: T x B x C
        if incremental_state is not None:
            assert incremental_state is None, "Nar-generation does not allow this."
            exit(1)
        if hasattr(self, 'padding'):
            x = self.padding(x)
        x = self.ffn_1(emotion, x.permute(1, 2, 0))
        x = x * self.kernel_size ** -0.5

        if incremental_state is not None:
            x = x[-1:]
        if self.act == "gelu":
            x = F.gelu(x)
        if self.act == "relu":
            x = F.relu(x)
        if self.act == "swish":
            x = self.swish_fn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ffn_2(emotion, x.permute(0, 2, 1)).permute(1,0,2)
        return x


class EmotionEncSALayer(nn.Module):
    def __init__(self,emotion_features, c, num_heads, dropout, attention_dropout=0.1,
                 relu_dropout=0.1, kernel_size=9, padding="SAME", norm="ln", act="gelu"):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.num_heads = num_heads
        if num_heads > 0:
            if norm == "ln":
                self.layer_norm1 = LayerNorm(c)
            elif norm == "bn":
                self.layer_norm1 = BatchNorm1dTBC(c)
            self.self_attn = MultiheadAttention(
                self.c, num_heads, self_attention=True, dropout=attention_dropout, bias=False,
            )
        if norm == "ln":
            self.layer_norm2 = LayerNorm(c)
        elif norm == "bn":
            self.layer_norm2 = BatchNorm1dTBC(c)
        self.ffn = EmotionTransformerFFNLayer(
            emotion_features,
            c, 4 * c, kernel_size=kernel_size, dropout=relu_dropout, padding=padding, act=act)

    def forward(self,emotion, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get("layer_norm_training", None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        if self.num_heads > 0:
            residual = x
            x = self.layer_norm1(x)
            x, _, = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask
            )
            x = F.dropout(x, self.dropout, training=self.training)
            x = residual + x
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]

        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(emotion, x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        return x

class EmotionResidualBlock(nn.Module):
    """ Residual Block """

    def __init__(self, emotion_features, d_encoder, residual_channels, dropout, multi_speaker=True):
        super(EmotionResidualBlock, self).__init__()
        self.multi_speaker = multi_speaker
        self.conv_layer = EmotionConvNorm(
            emotion_features,
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            stride=1,
            padding=int((3 - 1) / 2),
            dilation=1,
        )
        self.diffusion_projection = EmotionLinear(emotion_features, residual_channels, residual_channels)
        if multi_speaker:
            self.speaker_projection = LinearNorm(d_encoder, residual_channels)
        self.conditioner_projection = EmotionConvNorm(
            emotion_features, d_encoder, residual_channels, kernel_size=1
        )
        self.output_projection = EmotionConvNorm(
            emotion_features, residual_channels, 2 * residual_channels, kernel_size=1
        )

    def forward(self, emotion, x, conditioner, diffusion_step, speaker_emb, mask=None):

        diffusion_step = self.diffusion_projection(emotion, diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(emotion, conditioner)
        if self.multi_speaker:
            speaker_emb = self.speaker_projection(speaker_emb).unsqueeze(1).expand(
                -1, conditioner.shape[-1], -1
            ).transpose(1, 2)

        residual = y = x + diffusion_step
        y = self.conv_layer(
            emotion,
            (y + conditioner + speaker_emb) if self.multi_speaker else (y + conditioner)
        )
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(emotion, y)
        x, skip = torch.chunk(y, 2, dim=1)

        return (x + residual) / math.sqrt(2.0), skip

