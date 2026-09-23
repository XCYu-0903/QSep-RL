import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as AT
from torchlibrosa.stft import magphase
from torchmetrics.audio.sdr import (
    scale_invariant_signal_distortion_ratio as si_sdr,
    signal_distortion_ratio as sdr,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from helpers import utils as local_utils


HF_MODEL_NAME_TO_CACHE_DIR = {
    "bert-base-uncased": "models--bert-base-uncased",
    "roberta-base": "models--roberta-base",
    "facebook/bart-base": "models--facebook--bart-base",
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: true/false.")


def find_hf_snapshot(model_name, hf_hub_dir):
    model_cache_dir = HF_MODEL_NAME_TO_CACHE_DIR[model_name]
    snapshots_dir = os.path.join(os.path.expanduser(hf_hub_dir), model_cache_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None
    snapshots = [
        os.path.join(snapshots_dir, name)
        for name in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, name))
    ]
    return sorted(snapshots)[-1] if snapshots else None


def patch_transformers_local_loading(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from transformers import BartModel, BartTokenizer
    from transformers import BertModel, BertTokenizer
    from transformers import RobertaModel, RobertaTokenizer

    class_to_model_name = {
        BertTokenizer: "bert-base-uncased",
        BertModel: "bert-base-uncased",
        RobertaTokenizer: "roberta-base",
        RobertaModel: "roberta-base",
        BartTokenizer: "facebook/bart-base",
        BartModel: "facebook/bart-base",
    }
    for cls, model_name in class_to_model_name.items():
        snapshot = find_hf_snapshot(model_name, args.hf_hub_dir)
        if snapshot is None:
            continue
        original_from_pretrained = cls.from_pretrained

        def from_pretrained_local(
            pretrained_model_name_or_path,
            *model_args,
            _model_name=model_name,
            _snapshot=snapshot,
            _original=original_from_pretrained,
            **kwargs,
        ):
            if pretrained_model_name_or_path == _model_name:
                pretrained_model_name_or_path = _snapshot
            kwargs.setdefault("local_files_only", True)
            return _original(pretrained_model_name_or_path, *model_args, **kwargs)

        cls.from_pretrained = from_pretrained_local


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if any(key.startswith("clap_model.") for key in state_dict):
        raise ValueError(
            f"{checkpoint_path} contains clap_model.* keys. "
            "QSep-RL checkpoints must not include the original CLAP weights."
        )
    result = model.load_state_dict(state_dict, strict=False)
    real_missing = [key for key in result.missing_keys if not key.startswith("clap_model.")]
    real_unexpected = [key for key in result.unexpected_keys if "." in key]
    if real_missing or real_unexpected:
        raise RuntimeError(
            "Failed to load QSep-RL checkpoint: "
            f"missing non-CLAP keys={real_missing}, unexpected keys={real_unexpected}"
        )
    logging.info("Loaded QSep-RL model checkpoint from %s.", checkpoint_path)


def build_model(args, device):
    patch_transformers_local_loading(args)

    import laion_clap

    from model.CLAPSep_decoder import HTSAT_Decoder
    from model.QSepRL_backbone import QSepRLBackbone

    if not os.path.exists(args.clap_path):
        raise FileNotFoundError(f"CLAP checkpoint not found: {args.clap_path}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"QSep-RL checkpoint not found: {args.checkpoint_path}")

    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=str(device))
    clap_model.load_ckpt(args.clap_path, verbose=args.verbose_clap)
    decoder = HTSAT_Decoder(**args.model)
    model = QSepRLBackbone(
        clap_model,
        decoder,
        use_lora=args.lora,
        rank=args.lora_rank,
        nfft=args.nfft,
    )
    load_checkpoint(model, args.checkpoint_path)
    model.to(device)
    model.eval()
    return model


def load_wav(path, sr, duration):
    max_length = sr * duration
    wav = librosa.core.load(path, sr=sr, mono=True)[0]
    if len(wav) > max_length:
        wav = wav[:max_length]
    if len(wav) < max_length:
        wav = np.pad(wav, (0, max_length - len(wav)), "constant")
    return torch.from_numpy(wav).float()


def pad_or_trim(wav, target_len):
    if wav.size(-1) < target_len:
        wav = torch.nn.functional.pad(wav, (0, target_len - wav.size(-1)))
    elif wav.size(-1) > target_len:
        wav = wav[..., :target_len]
    return wav


def resolve_path(row, key, root):
    path_key = f"{key}_path"
    if row.get(path_key):
        return row[path_key]
    file_key = f"{key}_file"
    if file_key not in row:
        raise KeyError(f"Metadata row must contain {file_key} or {path_key}.")
    return str(Path(root) / row[file_key])


class CLAPSepEvalDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_csv, audio_root, sr=32000, resample_rate=48000, duration=10):
        self.metadata_csv = metadata_csv
        self.audio_root = audio_root
        self.sr = sr
        self.resample_rate = resample_rate
        self.duration = duration
        self.resampler = AT.Resample(sr, resample_rate)
        with open(metadata_csv, encoding="utf-8", newline="") as f:
            self.rows = list(csv.DictReader(f, skipinitialspace=True))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        source_path = resolve_path(row, "source", self.audio_root)
        noise_path = resolve_path(row, "noise", self.audio_root)
        target = load_wav(source_path, self.sr, self.duration).unsqueeze(0)
        noise = load_wav(noise_path, self.sr, self.duration).unsqueeze(0)
        mixed = torchaudio.functional.add_noise(target, noise, snr=torch.zeros(1))
        max_value = torch.max(torch.abs(mixed))
        if max_value > 1:
            target = target * (0.9 / max_value)
            mixed = mixed * (0.9 / max_value)
        mixed = mixed.squeeze(0)
        target = target.squeeze(0)
        return {
            "idx": idx,
            "mixed": mixed,
            "mixed_resample": self.resampler(mixed),
            "target": target,
            "positive_caption": row["positive_caption"],
            "negative_caption": row["negative_caption"],
            "source_file": row.get("source_file", os.path.basename(source_path)),
            "noise_file": row.get("noise_file", os.path.basename(noise_path)),
        }


@torch.no_grad()
def separate_batch(model, mixed, mixed_resample, pos_caps, neg_caps, query_mode):
    real, imag = model.stft(mixed)
    mag, cos, sin = magphase(real, imag)

    if query_mode == "negative":
        embed_neg = model.clap_model.get_text_embedding(list(neg_caps), use_tensor=True).to(mixed.device)
        embed_pos = torch.zeros_like(embed_neg)
    else:
        embed_pos = model.clap_model.get_text_embedding(list(pos_caps), use_tensor=True).to(mixed.device)
        if query_mode == "positive":
            embed_neg = torch.zeros_like(embed_pos)
        else:
            embed_neg = model.clap_model.get_text_embedding(list(neg_caps), use_tensor=True).to(mixed.device)

    model.features.clear()
    model.features.append(mag)
    embed = torch.cat([embed_pos, embed_neg], dim=-1)
    model.audio_branch({"waveform": mixed_resample})
    mask = model.decoder_model(
        hidden_state=model.features[-1],
        skip_features=model.features[:-1],
        embed=embed,
    )
    pred = model.wav_reconstruct(mask, mag, cos, sin, length=mixed.size(-1))
    model.features.clear()
    return pred


@torch.no_grad()
def clap_score(model, pred, captions, resampler, clap_len):
    pred_for_clap = pad_or_trim(resampler(pred), clap_len)
    audio_embed = model.clap_model.get_audio_embedding_from_data(pred_for_clap, use_tensor=True)
    text_embed = model.clap_model.get_text_embedding(list(captions), use_tensor=True)
    audio_embed = torch.nn.functional.normalize(audio_embed.to(pred.device), dim=-1)
    text_embed = torch.nn.functional.normalize(text_embed.to(pred.device), dim=-1)
    return torch.sum(audio_embed * text_embed, dim=-1)


@torch.no_grad()
def clap_score_tri(model, pred, pos_caps, neg_caps, resampler, clap_len):
    pred_for_clap = pad_or_trim(resampler(pred), clap_len)
    audio_embed = model.clap_model.get_audio_embedding_from_data(pred_for_clap, use_tensor=True)
    pos_embed = model.clap_model.get_text_embedding(list(pos_caps), use_tensor=True)
    neg_embed = model.clap_model.get_text_embedding(list(neg_caps), use_tensor=True)
    audio_embed = torch.nn.functional.normalize(audio_embed.to(pred.device), dim=-1)
    pos_embed = torch.nn.functional.normalize(pos_embed.to(pred.device), dim=-1)
    neg_embed = torch.nn.functional.normalize(neg_embed.to(pred.device), dim=-1)
    return torch.sum(audio_embed * pos_embed, dim=-1) - torch.sum(audio_embed * neg_embed, dim=-1)


def summarize(values):
    tensor = torch.tensor(values, dtype=torch.float32)
    return {"mean": tensor.mean().item(), "std": tensor.std(unbiased=False).item()}


def evaluate(args):
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    params = local_utils.Params(os.path.join(args.exp_dir, "config.json"))
    for key, value in params.__dict__.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            vars(args)[key] = value
    args.sr = args.sr or args.train_data.get("sr") or args.val_data.get("sr")
    args.resample_rate = args.resample_rate or args.train_data.get("resample_rate") or args.val_data.get("resample_rate")

    model = build_model(args, device)
    dataset = CLAPSepEvalDataset(
        args.metadata_csv,
        args.audio_root,
        sr=args.sr,
        resample_rate=args.resample_rate,
        duration=args.duration,
    )
    if args.limit is not None:
        dataset.rows = dataset.rows[:args.limit]
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    clap_resampler = AT.Resample(args.sr, args.resample_rate).to(device)
    clap_len = args.resample_rate * args.duration

    sdri_vals = []
    si_sdri_vals = []
    clapscore_vals = []
    clapscore_tri_vals = []
    for batch_idx, batch in enumerate(loader, start=1):
        mixed = batch["mixed"].to(device, non_blocking=True)
        mixed_resample = batch["mixed_resample"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        pred = separate_batch(
            model,
            mixed,
            mixed_resample,
            batch["positive_caption"],
            batch["negative_caption"],
            args.query_mode,
        )
        if args.eval_sdri:
            sdri_vals.extend((sdr(pred, target) - sdr(mixed, target)).detach().cpu().tolist())
        if args.eval_si_sdri:
            si_sdri_vals.extend((si_sdr(pred, target) - si_sdr(mixed, target)).detach().cpu().tolist())
        if args.eval_clapscore:
            clapscore_vals.extend(
                clap_score(model, pred, batch["positive_caption"], clap_resampler, clap_len).detach().cpu().tolist()
            )
        if args.eval_clapscore_tri:
            clapscore_tri_vals.extend(
                clap_score_tri(
                    model,
                    pred,
                    batch["positive_caption"],
                    batch["negative_caption"],
                    clap_resampler,
                    clap_len,
                ).detach().cpu().tolist()
            )
        if batch_idx % args.log_interval == 0 or batch_idx == len(loader):
            logging.info("batch %d/%d", batch_idx, len(loader))

    summary = {"num_samples": len(dataset), "query_mode": args.query_mode}
    if args.eval_sdri:
        summary["sdri"] = summarize(sdri_vals)
    if args.eval_si_sdri:
        summary["si_sdri"] = summarize(si_sdri_vals)
    if args.eval_clapscore:
        summary["clapscore"] = summarize(clapscore_vals)
    if args.eval_clapscore_tri:
        summary["clapscore_tri"] = summarize(clapscore_tri_vals)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info(json.dumps(summary, indent=2))
    logging.info("Wrote summary to %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QSep-RL on CLAPSep-style source/noise metadata.")
    parser.add_argument("--exp_dir", default="./experiments/QSep-RL")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--audio_root", required=True)
    parser.add_argument("--clap_path", default=None)
    parser.add_argument("--output_dir", default="eval_results")
    parser.add_argument("--output_name", default="summary.json")
    parser.add_argument("--query_mode", choices=["positive", "negative", "positive_negative"], default="positive_negative")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--sr", type=int, default=None)
    parser.add_argument("--resample_rate", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval_sdri", type=str2bool, default=True)
    parser.add_argument("--eval_si_sdri", type=str2bool, default=True)
    parser.add_argument("--eval_clapscore", type=str2bool, default=True)
    parser.add_argument("--eval_clapscore_tri", type=str2bool, default=True)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--verbose_clap", action="store_true", default=False)
    parser.add_argument("--hf_hub_dir", default=os.path.expanduser("~/.cache/huggingface/hub"))
    cli_args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    evaluate(cli_args)
