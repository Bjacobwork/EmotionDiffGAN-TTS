import math

import torch
from .blocks import (
    Embedding,
    SinusoidalPositionalEmbedding,
    LayerNorm,
    LinearNorm,
    ConvNorm,
    BatchNorm1dTBC,
    EncSALayer,
    Mish,
    DiffusionEmbedding,
    ResidualBlock,
)

class Conv2dNorm(torch.nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                         groups, bias, padding_mode, device, dtype)
        torch.nn.init.kaiming_normal_(self.weight)

class DenseLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1,
                 groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__()
        conv_channels = out_channels*2
        self.conv = Conv2dNorm(in_channels, conv_channels, kernel_size, stride, 'same',
                                    dilation, groups, bias, padding_mode, device, dtype)
        layer_channels = out_channels
        self.layer = Conv2dNorm(out_channels, layer_channels, 1, stride, 'same',1, groups,
                                   bias, padding_mode, device, dtype)
    def forward(self, x):
        gate, filter = torch.chunk(self.conv(x), 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        return torch.concat([x, self.layer(y)], dim=1)


class DenseBlock(torch.nn.Sequential):

    def __init__(self, in_channels, growth_size, n_layers, kernel_size, stride=1, dilation=1,
                 groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        layers = []
        for _ in range(n_layers):
            layers.append(DenseLayer(in_channels, growth_size, kernel_size, stride, dilation,
                                     groups, bias, padding_mode, device, dtype))
            in_channels += growth_size
        super().__init__(*layers)


class DenseDenoiser(torch.nn.Module):

    def __init__(self, preprocess_config, model_config):
        super().__init__()
        d_encoder = model_config["transformer"]["encoder_hidden"]
        residual_channels = model_config["denoiser"]["residual_channels"]
        dense_channels = model_config["denoiser"]["dense_channels"]
        dense_layers = model_config["denoiser"]["dense_layers"]
        self.multi_speaker = model_config["multi_speaker"]
        self.conditioner_channels = model_config["denoiser"]["conditioner_channels"]
        self.n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        if self.multi_speaker:
            self.speaker_projection = LinearNorm(d_encoder, self.n_mel_channels)
        self.conditioner_projection = ConvNorm(d_encoder, self.conditioner_channels*self.n_mel_channels)
        self.diffusion_projection = torch.nn.Sequential(
            DiffusionEmbedding(d_encoder),
            LinearNorm(d_encoder, residual_channels * 4),
            Mish(),
            LinearNorm(residual_channels * 4, residual_channels),
            LinearNorm(residual_channels, self.n_mel_channels)
        )
        in_channels = 4 if self.multi_speaker else 3
        in_channels += self.conditioner_channels
        self.dense_projection = torch.nn.Sequential(
            DenseBlock(in_channels, dense_channels, dense_layers, 3),
            Conv2dNorm(in_channels+dense_channels*dense_layers, 1, 1),
        )
        self.output_projection = self.output_projection = torch.nn.Sequential(
            Conv2dNorm(1, 1, 3, padding='same'),
            torch.nn.ReLU(),
            Conv2dNorm(1, 1, 3, padding='same'),
        )
        self.output_projection_1d = torch.nn.Sequential(
            ConvNorm(self.n_mel_channels, self.n_mel_channels),
            torch.nn.ReLU(),
            ConvNorm(self.n_mel_channels, self.n_mel_channels),
        )
        self.position = torch.nn.Parameter(torch.cos(torch.arange(
            self.n_mel_channels)*torch.pi/2).view(1, 1, self.n_mel_channels, 1), False)


    def project_conditioner(self, conditioner):
        return self.conditioner_projection(conditioner).view(
            -1, self.conditioner_channels, self.n_mel_channels, conditioner.shape[-1])

    def project_diffusion_step(self, diffusion_step):
        return self.diffusion_projection(diffusion_step).view(-1, 1, self.n_mel_channels, 1)

    def project_speaker_emb(self, speaker_emb):
        return self.speaker_projection(speaker_emb).view(-1, 1, self.n_mel_channels, 1)

    def forward(self, mel, diffusion_step, conditioner: torch.Tensor, speaker_emb, mask=None):
        """

        :param mel: [B, 1, M, T]
        :param diffusion_step: [B,]
        :param conditioner: [B, M, T]
        :param speaker_emb: [B, M]
        :return: mel_pred: [B, 1, M, T]

        mel torch.Size([4, 1, 80, 189])
        diffusion_step torch.Size([4])
        conditioner torch.Size([4, 256, 189])
        speaker_emb torch.Size([4, 256])

        """

        conditioner = self.project_conditioner(conditioner)
        diffusion_step = self.project_diffusion_step(diffusion_step)
        channels = [mel,
                    self.position.repeat(mel.shape[0], 1, 1, mel.shape[-1]),
                    conditioner,
                    diffusion_step.repeat(1,1,1,mel.shape[-1])]
        if self.multi_speaker:
            channels.append(self.project_speaker_emb(speaker_emb).repeat(1,1,1,mel.shape[-1]))
        mel = torch.concat(channels, dim=1)
        mel = self.dense_projection(mel)
        return self.output_projection(mel)

