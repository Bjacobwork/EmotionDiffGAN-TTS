import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad, dur_to_mel2ph
from utils.pitch_tools import f0_to_coarse, denorm_f0, cwt2f0_norm

from .blocks import (
    Embedding,
    SinusoidalPositionalEmbedding,
    LayerNorm,
    BatchNorm1dTBC,
    DiffusionEmbedding,
)
from .emotion_blocks import (
    EmotionLinear,
    EmotionConv1d,
    EmotionMish,
    EmotionRelu,
    EmotionConvNorm,
    EmotionEncSALayer,
    EmotionResidualBlock,
)
from .modules import (
    LengthRegulator
)
from text.symbols import symbols

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmotionTransformerEncoderLayer(nn.Module):
    def __init__(self, emotion_features, hidden_size, dropout, kernel_size=None, num_heads=2, norm="ln", ffn_padding="SAME", ffn_act="gelu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.op = EmotionEncSALayer(
            emotion_features,
            hidden_size, num_heads, dropout=dropout,
            attention_dropout=0.0, relu_dropout=dropout,
            kernel_size=kernel_size,
            padding=ffn_padding,
            norm=norm, act=ffn_act)

    def forward(self, emotion, x, **kwargs):
        return self.op(emotion, x, **kwargs)


class EmotionFFTBlocks(nn.Module):
    def __init__(self, emotion_features, hidden_size, num_layers, max_seq_len=2000, ffn_kernel_size=9, dropout=None, num_heads=2,
                 use_pos_embed=True, use_last_norm=True, norm="ln", ffn_padding="SAME", ffn_act="gelu", use_pos_embed_alpha=True):
        super().__init__()
        self.num_layers = num_layers
        embed_dim = self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_pos_embed = use_pos_embed
        self.use_last_norm = use_last_norm
        if use_pos_embed:
            self.max_source_positions = max_seq_len
            self.padding_idx = 0
            self.pos_embed_alpha = nn.Parameter(torch.Tensor([1])) if use_pos_embed_alpha else 1
            self.embed_positions = SinusoidalPositionalEmbedding(
                embed_dim, self.padding_idx, init_size=max_seq_len,
            )

        self.layers = nn.ModuleList([
            EmotionTransformerEncoderLayer(emotion_features, self.hidden_size, self.dropout,
                                    kernel_size=ffn_kernel_size, num_heads=num_heads, ffn_padding=ffn_padding, ffn_act=ffn_act)
            for _ in range(self.num_layers)
        ])
        if self.use_last_norm:
            if norm == "ln":
                self.layer_norm = nn.LayerNorm(embed_dim)
            elif norm == "bn":
                self.layer_norm = BatchNorm1dTBC(embed_dim)
        else:
            self.layer_norm = None

    def forward(self,emotion, x, padding_mask=None, attn_mask=None, return_hiddens=False):
        """
        :param x: [B, T, C]
        :param padding_mask: [B, T]
        :return: [B, T, C] or [L, B, T, C]
        """
        padding_mask = x.abs().sum(-1).eq(0).data if padding_mask is None else padding_mask
        nonpadding_mask_TB = 1 - padding_mask.transpose(0, 1).float()[:, :, None]  # [T, B, 1]
        if self.use_pos_embed:
            positions = self.pos_embed_alpha * self.embed_positions(x[..., 0])
            x = x + positions
            x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1) * nonpadding_mask_TB
        hiddens = []
        for layer in self.layers:
            x = layer(emotion, x, encoder_padding_mask=padding_mask, attn_mask=attn_mask) * nonpadding_mask_TB
            hiddens.append(x)
        if self.use_last_norm:
            x = self.layer_norm(x) * nonpadding_mask_TB
        if return_hiddens:
            x = torch.stack(hiddens, 0)  # [L, T, B, C]
            x = x.transpose(1, 2)  # [L, B, T, C]
        else:
            x = x.transpose(0, 1)  # [B, T, C]
        return x

class EmotionVarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, preprocess_config, model_config, train_config):
        super(EmotionVarianceAdaptor, self).__init__()
        emotion_features = model_config['emotion_features']
        self.preprocess_config = preprocess_config

        self.use_pitch_embed = model_config["variance_embedding"]["use_pitch_embed"]
        self.use_energy_embed = model_config["variance_embedding"]["use_energy_embed"]
        self.predictor_grad = model_config["variance_predictor"]["predictor_grad"]

        self.hidden_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.predictor_layers = model_config["variance_predictor"]["predictor_layers"]
        self.dropout = model_config["variance_predictor"]["dropout"]
        self.ffn_padding = model_config["transformer"]["ffn_padding"]
        self.kernel = model_config["variance_predictor"]["predictor_kernel"]
        self.duration_predictor = EmotionDurationPredictor(
            model_config['emotion_features'],
            self.hidden_size,
            n_chans=self.filter_size,
            n_layers=model_config["variance_predictor"]["dur_predictor_layers"],
            dropout_rate=self.dropout, padding=self.ffn_padding,
            kernel_size=model_config["variance_predictor"]["dur_predictor_kernel"],
            dur_loss=train_config["loss"]["dur_loss"])
        self.length_regulator = LengthRegulator()
        if self.use_pitch_embed:
            n_bins = model_config["variance_embedding"]["pitch_n_bins"]
            self.pitch_type = preprocess_config["preprocessing"]["pitch"]["pitch_type"]
            self.use_uv = preprocess_config["preprocessing"]["pitch"]["use_uv"]

            if self.pitch_type == "cwt":
                self.cwt_std_scale = model_config["variance_predictor"]["cwt_std_scale"]
                h = model_config["variance_predictor"]["cwt_hidden_size"]
                cwt_out_dims = 10
                if self.use_uv:
                    cwt_out_dims = cwt_out_dims + 1
                self.cwt_predictor = nn.ModuleList([
                    EmotionLinear(emotion_features, self.hidden_size, h),
                    EmotionPitchPredictor(
                        emotion_features,
                        h,
                        n_chans=self.filter_size,
                        n_layers=self.predictor_layers,
                        dropout_rate=self.dropout, odim=cwt_out_dims,
                        padding=self.ffn_padding, kernel_size=self.kernel)])
                self.cwt_stats_layers = nn.ModuleList([
                    EmotionLinear(emotion_features, self.hidden_size, h), EmotionRelu(),
                    EmotionLinear(emotion_features, h, h), EmotionRelu(),
                    EmotionLinear(emotion_features, h, 2)
                ])
            else:
                self.pitch_predictor = EmotionPitchPredictor(
                    emotion_features,
                    self.hidden_size,
                    n_chans=self.filter_size,
                    n_layers=self.predictor_layers,
                    dropout_rate=self.dropout,
                    odim=2 if self.pitch_type == "frame" else 1,
                    padding=self.ffn_padding, kernel_size=self.kernel)
            self.pitch_embed = Embedding(n_bins, self.hidden_size, padding_idx=0)
        if self.use_energy_embed:
            self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
                "feature"
            ]
            assert self.energy_feature_level in ["phoneme_level", "frame_level"]
            energy_quantization = model_config["variance_embedding"]["energy_quantization"]
            assert energy_quantization in ["linear", "log"]
            n_bins = model_config["variance_embedding"]["energy_n_bins"]
            with open(
                os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
            ) as f:
                stats = json.load(f)
                energy_min, energy_max = stats["energy"][:2]

            self.energy_predictor = EmotionEnergyPredictor(
                emotion_features,
                self.hidden_size,
                n_chans=self.filter_size,
                n_layers=self.predictor_layers,
                dropout_rate=self.dropout, odim=1,
                padding=self.ffn_padding, kernel_size=self.kernel)
            if energy_quantization == "log":
                self.energy_bins = nn.Parameter(
                    torch.exp(
                        torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                    ),
                    requires_grad=False,
                )
            else:
                self.energy_bins = nn.Parameter(
                    torch.linspace(energy_min, energy_max, n_bins - 1),
                    requires_grad=False,
                )
            self.energy_embedding = Embedding(n_bins, self.hidden_size, padding_idx=0)

    def get_pitch_embedding(self, emotion, decoder_inp, f0, uv, mel2ph, control, encoder_out=None):
        pitch_pred = f0_denorm = cwt = f0_mean = f0_std = None
        if self.pitch_type == "ph":
            pitch_pred_inp = encoder_out.detach() + self.predictor_grad * (encoder_out - encoder_out.detach())
            pitch_padding = encoder_out.sum().abs() == 0
            pitch_pred = self.pitch_predictor(emotion, pitch_pred_inp) * control
            if f0 is None:
                f0 = pitch_pred[:, :, 0]
            f0_denorm = denorm_f0(f0, None, self.preprocess_config["preprocessing"]["pitch"], pitch_padding=pitch_padding)
            pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
            pitch = F.pad(pitch, [1, 0])
            pitch = torch.gather(pitch, 1, mel2ph)  # [B, T_mel]
            pitch_embed = self.pitch_embed(pitch)
        else:
            decoder_inp = decoder_inp.detach() + self.predictor_grad * (decoder_inp - decoder_inp.detach())
            pitch_padding = mel2ph == 0

            if self.pitch_type == "cwt":
                pitch_padding = None
                cwt = decoder_inp
                for layer in self.cwt_predictor:
                    cwt = layer(emotion, cwt)
                cwt = cwt_out = cwt * control
                stats_out = encoder_out[:, 0, :]
                for layer in self.cwt_stats_layers:
                    stats_out = layer(emotion, stats_out)
                mean = f0_mean = stats_out[:, 0]
                std = f0_std = stats_out[:, 1]
                cwt_spec = cwt_out[:, :, :10]
                if f0 is None:
                    std = std * self.cwt_std_scale
                    f0 = cwt2f0_norm(
                        cwt_spec, mean, std, mel2ph, self.preprocess_config["preprocessing"]["pitch"],
                    )
                    if self.use_uv:
                        assert cwt_out.shape[-1] == 11
                        uv = cwt_out[:, :, -1] > 0
            elif self.preprocess_config["preprocessing"]["pitch"]["pitch_ar"]:
                pitch_pred = self.pitch_predictor(emotion, decoder_inp, f0 if self.training else None) * control
                if f0 is None:
                    f0 = pitch_pred[:, :, 0]
            else:
                pitch_pred = self.pitch_predictor(emotion, decoder_inp) * control
                if f0 is None:
                    f0 = pitch_pred[:, :, 0]
                if self.use_uv and uv is None:
                    uv = pitch_pred[:, :, 1] > 0

            f0_denorm = denorm_f0(f0, uv, self.preprocess_config["preprocessing"]["pitch"], pitch_padding=pitch_padding)
            if pitch_padding is not None:
                f0[pitch_padding] = 0

            pitch = f0_to_coarse(f0_denorm)  # start from 0
            pitch_embed = self.pitch_embed(pitch)

        pitch_pred = {
            "pitch_pred": pitch_pred,
            "f0_denorm": f0_denorm,
            "cwt": cwt,
            "f0_mean": f0_mean,
            "f0_std": f0_std,
        }

        return pitch_pred, pitch_embed

    def get_energy_embedding(self, emotion, x, target, mask, control):
        x.detach() + self.predictor_grad * (x - x.detach())
        prediction = self.energy_predictor(emotion, x, squeeze=True)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        emotion,
        x,
        src_mask,
        max_src_len,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        mel2ph=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        speaker_emb=None,
    ):
        pitch_prediction = energy_prediction = None

        if speaker_emb is not None:
            x = x + speaker_emb.unsqueeze(1).expand(
            -1, max_src_len, -1
        )

        output_1 = x.clone()
        log_duration_prediction = self.duration_predictor(
            emotion,
            x.detach() + self.predictor_grad * (x - x.detach()), src_mask
        )
        if self.use_energy_embed and self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                emotion,
                x, energy_target, src_mask, e_control
            )
            output_1 += energy_embedding
        x = output_1.clone()

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            mel2ph = dur_to_mel2ph(duration_rounded, src_mask)
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        output_2 = x.clone()
        if self.use_pitch_embed: # and self.pitch_type in ["frame", "cwt"]:
            if pitch_target is not None:
                if self.pitch_type == "cwt":
                    cwt_spec = pitch_target[f"cwt_spec"]
                    f0_mean = pitch_target["f0_mean"]
                    f0_std = pitch_target["f0_std"]
                    pitch_target["f0"] = cwt2f0_norm(
                        cwt_spec, f0_mean, f0_std, mel2ph, self.preprocess_config["preprocessing"]["pitch"],
                    )
                    pitch_target.update({"f0_cwt": pitch_target["f0"]})
                pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                    emotion,
                    x, pitch_target["f0"], pitch_target["uv"], mel2ph, p_control, encoder_out=output_1
                )
            else:
                pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                    emotion,
                    x, None, None, mel2ph, p_control, encoder_out=output_1
                )
            output_2 += pitch_embedding
        if self.use_energy_embed and self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                emotion,
                x, energy_target, mel_mask, e_control
            )
            output_2 += energy_embedding
        x = output_2.clone()

        return (
            x,
            pitch_target,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )

class EmotionPredictorLayer(torch.nn.Module):
    def __init__(self, emotion_features, in_chans, n_chans, kernel_size, dropout_rate=0.1, padding="SAME"):
        super().__init__()
        self.padding = torch.nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                                       if padding == "SAME"
                                       else (kernel_size - 1, 0), 0)
        self.conv = EmotionConv1d(emotion_features, in_chans, n_chans, kernel_size, stride=1, padding=0)
        self.layer = torch.nn.Sequential(
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )

    def forward(self, emotion, x):
        return self.layer(self.conv(emotion, self.padding(x)))
class EmotionDurationPredictor(torch.nn.Module):
    """Duration predictor module.
    This is a module of duration predictor described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain from the hidden embeddings of encoder.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    Note:
        The outputs are calculated in log domain.
    """

    def __init__(self, emotion_features, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0, padding="SAME", dur_loss="mse"):
        """Initilize duration predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.
        """
        super(EmotionDurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        self.dur_loss = dur_loss
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv.append(EmotionPredictorLayer(emotion_features, in_chans, n_chans, kernel_size, dropout_rate, padding))
        if self.dur_loss in ["mse", "huber"]:
            odims = 1
        elif self.dur_loss == "mog":
            odims = 15
        elif self.dur_loss == "crf":
            odims = 32
            from torchcrf import CRF
            self.crf = CRF(odims, batch_first=True)
        self.linear = EmotionLinear(emotion_features, n_chans, odims)

    def forward(self,emotion, xs, x_masks=None):
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(emotion, xs)  # (B, C, Tmax)
            if x_masks is not None:
                xs = xs * (1 - x_masks.float())[:, None, :]

        xs = self.linear(emotion, xs.transpose(1, -1))  # [B, T, C]
        xs = xs * (1 - x_masks.float())[:, :, None]  # (B, T, C)
        if self.dur_loss in ["mse"]:
            xs = xs.squeeze(-1)  # (B, Tmax)
        return xs


class EmotionPitchPredictor(torch.nn.Module):
    def __init__(self,emotion_features, idim, n_layers=5, n_chans=384, odim=2, kernel_size=5,
                 dropout_rate=0.1, padding="SAME"):
        """Initilize pitch predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super(EmotionPitchPredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv.append(EmotionPredictorLayer(emotion_features, in_chans, n_chans, kernel_size, dropout_rate, padding))
        self.linear = EmotionLinear(emotion_features, n_chans, odim)
        self.embed_positions = SinusoidalPositionalEmbedding(idim, 0, init_size=4096)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))

    def forward(self,emotion, xs, squeeze=False):
        """

        :param xs: [B, T, H]
        :return: [B, T, H]
        """
        positions = self.pos_embed_alpha * self.embed_positions(xs[..., 0])
        xs = xs + positions
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(emotion, xs)  # (B, C, Tmax)
        # NOTE: calculate in log domain
        xs = self.linear(emotion, xs.transpose(1, -1))  # (B, Tmax, H)
        return xs.squeeze(-1) if squeeze else xs


class EmotionEnergyPredictor(EmotionPitchPredictor):
    pass


class EmotionDenoiser(nn.Module):
    """ Conditional Diffusion Denoiser """

    def __init__(self, preprocess_config, model_config):
        super(EmotionDenoiser, self).__init__()
        emotion_features = model_config['emotion_features']
        n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        d_encoder = model_config["transformer"]["encoder_hidden"]
        residual_channels = model_config["denoiser"]["residual_channels"]
        residual_layers = model_config["denoiser"]["residual_layers"]
        dropout = model_config["denoiser"]["denoiser_dropout"]
        multi_speaker = model_config["multi_speaker"]

        self.input_projection = EmotionConvNorm(emotion_features, n_mel_channels, residual_channels, kernel_size=1)

        self.diffusion_embedding = DiffusionEmbedding(residual_channels)
        self.mlp = nn.ModuleList([
            EmotionLinear(emotion_features, residual_channels, residual_channels * 4),
            EmotionMish(),
            EmotionLinear(emotion_features, residual_channels * 4, residual_channels)]
        )
        self.residual_layers = nn.ModuleList(
            [
                EmotionResidualBlock(
                    emotion_features, d_encoder, residual_channels, dropout=dropout, multi_speaker=multi_speaker
                )
                for _ in range(residual_layers)
            ]
        )
        self.skip_projection = EmotionConvNorm(
            emotion_features,
            residual_channels, residual_channels, kernel_size=1
        )
        self.output_projection = EmotionConvNorm(
            emotion_features,
            residual_channels, n_mel_channels, kernel_size=1
        )

    def forward(self, emotion, mel, diffusion_step, conditioner, speaker_emb, mask=None):
        """

        :param mel: [B, 1, M, T]
        :param diffusion_step: [B,]
        :param conditioner: [B, M, T]
        :param speaker_emb: [B, M]
        :return:
        """
        x = mel[:, 0]
        x = self.input_projection(emotion, x)  # x [B, residual_channel, T]
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        for layer in self.mlp:
            diffusion_step = layer(emotion, diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(emotion, x, conditioner, diffusion_step, speaker_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(emotion, x)
        x = F.relu(x)
        x = self.output_projection(emotion, x)  # [B, 80, T]

        return x[:, None, :, :]
