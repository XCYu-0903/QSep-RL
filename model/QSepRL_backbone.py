#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import copy

import loralib as lora
import torch
from torchlibrosa import ISTFT, STFT
from torchlibrosa.stft import magphase


def set_module(model, submodule_key, module):
    tokens = submodule_key.split(".")
    cur_mod = model
    for token in tokens[:-1]:
        cur_mod = getattr(cur_mod, token)
    setattr(cur_mod, tokens[-1], module)


def process_model(model, rank):
    for name, module in model.named_modules():
        if "WindowAttention" in str(type(module)):
            for sub_name, layer in module.named_modules():
                if isinstance(layer, torch.nn.Linear):
                    lora_layer = lora.Linear(
                        layer.in_features,
                        layer.out_features,
                        r=rank,
                        bias=hasattr(layer, "bias"),
                        merge_weights=False,
                    )
                    lora_layer.weight = layer.weight
                    if hasattr(layer, "bias"):
                        lora_layer.bias = layer.bias
                    set_module(model, name + "." + sub_name, lora_layer)
    return model


class QSepRLBackbone(torch.nn.Module):
    """Inference-only QSep-RL backbone with checkpoint-compatible module names."""

    def __init__(
        self,
        clap_model,
        decoder_model,
        use_lora=False,
        rank=8,
        nfft=1024,
        **_,
    ):
        super().__init__()
        self.phase = decoder_model.phase
        self.clap_model = clap_model
        for param in self.clap_model.parameters():
            param.requires_grad = False

        self.audio_branch = copy.deepcopy(self.clap_model.model.audio_branch)
        if use_lora:
            process_model(self.audio_branch, rank)
            lora.mark_only_lora_as_trainable(self.audio_branch, bias="lora_only")

        self.decoder_model = decoder_model
        self.stft = STFT(
            n_fft=nfft,
            hop_length=320,
            win_length=nfft,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.istft = ISTFT(
            n_fft=nfft,
            hop_length=320,
            win_length=nfft,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.features = self.install_forward_hooks()

    @torch.no_grad()
    def separate(self, mixed, mixed_resample, query, negative_query=None):
        self.eval()
        real, imag = self.stft(mixed)
        mag, cos, sin = magphase(real, imag)

        embed_pos = self.clap_model.get_text_embedding(list(query), use_tensor=True).to(mixed.device)
        if negative_query is None:
            embed_neg = torch.zeros_like(embed_pos)
        else:
            embed_neg = self.clap_model.get_text_embedding(list(negative_query), use_tensor=True).to(mixed.device)

        pred, _ = self.separate_from_features(
            embed_pos,
            embed_neg,
            mag,
            cos,
            sin,
            length=mixed.size(-1),
            mixed_resample=mixed_resample,
        )
        return pred

    def separate_from_features(
        self,
        embed_p,
        embed_n,
        mag,
        cos,
        sin,
        length,
        features=None,
        mixed_resample=None,
    ):
        features = self.features if features is None else features
        if mixed_resample is not None:
            features.clear()
            features.append(mag)
            self.audio_branch({"waveform": mixed_resample})

        embed = torch.nn.functional.normalize(torch.cat([embed_p, embed_n], dim=-1), dim=-1)
        mask = self.decoder_model(
            hidden_state=features[-1],
            skip_features=features[:-1],
            embed=embed,
        )
        pred = self.wav_reconstruct(mask, mag, cos, sin, length=length)
        features.clear()
        return pred, mask

    def wav_reconstruct(self, mask, mag_x, cos_x, sin_x, length):
        if self.phase:
            mag_y = torch.nn.functional.relu_(mag_x * mask[0])
            _, mask_cos, mask_sin = magphase(mask[1], mask[2])
            cos_y = cos_x * mask_cos - sin_x * mask_sin
            sin_y = sin_x * mask_cos + cos_x * mask_sin
        else:
            mag_y = torch.nn.functional.relu_(mag_x * mask)
            cos_y = cos_x
            sin_y = sin_x
        return self.istft(mag_y * cos_y, mag_y * sin_y, length=length)

    def install_forward_hooks(self):
        features = []

        def get_features_list(_, __, output):
            features.append(output)

        def get_features_list_basic_layer(_, __, output):
            features.append(output[0])

        def spectrogram_padding(_, __, out):
            return torch.nn.functional.pad(out, (0, 0, 0, 1024 - out.size(2)))

        self.audio_branch.spectrogram_extractor.register_forward_hook(spectrogram_padding)
        self.audio_branch.patch_embed.register_forward_hook(get_features_list)
        for module in self.audio_branch.layers:
            module.register_forward_hook(get_features_list_basic_layer)
        return features


# Backward-compatible class name for inference/evaluation imports.
QSepRL = QSepRLBackbone
