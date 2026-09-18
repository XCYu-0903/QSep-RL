#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Waveformer-main
@File    ：dataset_online.py
@IDE     ：PyCharm
@Author  ：Aisaka/Hao Ma @SDU
@Date    ：2023/11/1 下午6:47
'''
import os
import random
import json

import torch
import torchaudio
import torchaudio.transforms as AT
import numpy as np
import librosa

from data_utils.audioset_ontology import has_label_conflict, load_ancestor_map, split_mids


class AudioCapMix(torch.utils.data.Dataset):  # type: ignore

    def __init__(self, input_dir=None, dset='', sr=None,
                 resample_rate=None, metadata_json=None,
                 ontology_path=None, use_ontology_filter=False,
                 max_sample_attempts=200):
        assert dset in ['train', 'val'], \
            "`dset` must be one of ['train', 'val']"
        self.dset = dset
        self.use_ontology_filter = use_ontology_filter
        self.max_sample_attempts = max_sample_attempts
        self.ancestor_map = load_ancestor_map(ontology_path) if use_ontology_filter and ontology_path else None
        self.data_path = os.path.join(input_dir, dset) if input_dir is not None else None
        self.data_meta = dict()
        self.data_paths = dict()
        self.data_mids = dict()
        self.data_class_names = dict()
        metadata_json = metadata_json or os.path.join('./metadata', f'audiocaps_{dset}.json')
        with open(metadata_json, encoding='utf-8') as f:
            rows = json.load(f)
        for row in rows:
            file_name = row['file_name']
            self.data_meta[file_name] = row['caption']
            if row.get('path'):
                self.data_paths[file_name] = row['path']
            self.data_mids[file_name] = split_mids(row.get('mids', ''))
            self.data_class_names[file_name] = row.get('class_names', '')
        self.augmentation = torchaudio.transforms.SpeedPerturbation(48000, [0.9, 1.1])

        self.data_names = list(self.data_meta.keys())
        if dset == 'val':
            self.noise_names = []
            for name in self.data_names:
                while True:
                    noise_name = random.sample(self.data_names, 1)[0]
                    if noise_name != name:
                        break
                self.noise_names.append(noise_name)

        if resample_rate is not None:
            self.resampler = AT.Resample(sr, resample_rate)
            self.sr = sr
            self.resample_rate = resample_rate
        else:
            self.sr = sr

    def __len__(self):
        return len(self.data_names)

    def load_wav(self, path):
        max_length = self.sr * 10
        wav = librosa.core.load(path, sr=self.sr)[0]
        if len(wav) > max_length:
            wav = wav[0:max_length]

        # pad audio to max length, 10s for AudioCaps
        if len(wav) < max_length:
            wav = np.pad(wav, (0, max_length - len(wav)), 'constant')
        return wav

    def get_audio_path(self, file_name):
        if file_name in self.data_paths:
            return self.data_paths[file_name]
        if self.data_path is None:
            raise ValueError("input_dir is required when metadata_csv does not provide absolute paths")
        return os.path.join(self.data_path, file_name)

    def is_compatible(self, name, other_names):
        if not self.use_ontology_filter or self.ancestor_map is None:
            return True
        mids = self.data_mids.get(name, [])
        if not mids:
            return True
        for other_name in other_names:
            other_mids = self.data_mids.get(other_name, [])
            if has_label_conflict(mids, other_mids, self.ancestor_map):
                return False
        return True

    def sample_compatible_name(self, excluded_names):
        excluded = set(excluded_names)
        fallback = None
        for _ in range(self.max_sample_attempts):
            name = random.sample(self.data_names, 1)[0]
            if name in excluded:
                continue
            if fallback is None:
                fallback = name
            if self.is_compatible(name, excluded):
                return name
        if fallback is not None:
            return fallback
        raise RuntimeError("Unable to sample a different AudioCaps item")

    def __getitem__(self, idx):

        tgt_name = self.data_names[idx]
        if self.dset =='train':
            noise_name = self.sample_compatible_name([tgt_name])
        else:
            noise_name = self.noise_names[idx]

        snr = torch.zeros((1,))
        # snr = (torch.rand((1,)) * 10 - 5) if self.dset == 'train' else torch.zeros((1,))
        tgt = torch.tensor(self.load_wav(self.get_audio_path(tgt_name))).unsqueeze(0)
        noise = torch.tensor(self.load_wav(self.get_audio_path(noise_name))).unsqueeze(0)
        mixed = torchaudio.functional.add_noise(tgt, noise, snr=snr)
        neg_sample, _ = self.augmentation(self.resampler(noise.squeeze()))

        max_value = torch.max(torch.abs(mixed))
        if max_value > 1:
            tgt *= 0.9 / max_value
            mixed *= 0.9 / max_value

        tgt = tgt.squeeze()
        mixed = mixed.squeeze()
        tgt_cap = self.data_meta[tgt_name]
        neg_cap = self.data_meta[noise_name]
        pos_sample, _ = self.augmentation(self.resampler(tgt))

        mixed_resample = self.resampler(mixed)

        return mixed, mixed_resample, tgt_cap, neg_cap, tgt,\
               self.pad_or_trim(pos_sample, 48000*10), self.pad_or_trim(neg_sample, 48000*10)

    def pad_or_trim(self, wav_in, target_len):
        if wav_in.size(0) < target_len:
            wav_in = torch.nn.functional.pad(wav_in, (0, target_len-wav_in.size(0)))
        elif wav_in.size(0) > target_len:
            wav_in = wav_in[:target_len]
        max_value = torch.max(torch.abs(wav_in))
        if max_value > 1:
            wav_in *= 0.9 / max_value
        return wav_in


# if __name__ == "__main__":
#     dataset = FSDSoundScapesDataset(input_dir="/home/user/202212661/clapsep/Waveformer-main/data/audiocap",
#         dset="test",
#         sr=32000,
#         resample_rate=48000,
#         return_neg=True)
#     mixed, tgt_cap, tgt = dataset.__getitem__(1)
#     print()
