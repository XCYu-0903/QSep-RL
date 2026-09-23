import argparse
import logging
import os
import sys

import torch
import torchaudio
import torchaudio.transforms as AT
from torchlibrosa.stft import magphase

from helpers import utils as local_utils


HF_MODEL_NAME_TO_CACHE_DIR = {
    "bert-base-uncased": "models--bert-base-uncased",
    "roberta-base": "models--roberta-base",
    "facebook/bart-base": "models--facebook--bart-base",
}


def load_audio_mono(path):
    wav, input_sr = torchaudio.load(path)
    return wav.float().mean(dim=0), input_sr


def resample_if_needed(wav, source_sr, target_sr):
    if source_sr == target_sr:
        return wav
    return AT.Resample(source_sr, target_sr)(wav)


def normalize_if_needed(wav):
    max_value = torch.max(torch.abs(wav))
    if max_value > 1:
        wav = wav * (0.9 / max_value)
    return wav


def split_audio(wav, chunk_size):
    chunks = []
    lengths = []
    for start in range(0, wav.numel(), chunk_size):
        chunk = wav[start:start + chunk_size]
        lengths.append(chunk.numel())
        if chunk.numel() < chunk_size:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_size - chunk.numel()))
        chunks.append(chunk)
    if not chunks:
        chunks = [torch.zeros(chunk_size, dtype=wav.dtype)]
        lengths = [0]
    return chunks, lengths


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
            logging.warning("No local HuggingFace snapshot found for %s under %s", model_name, args.hf_hub_dir)
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


@torch.no_grad()
def separate_chunk(model, mixed, mixed_resample, query, negative_query, device):
    mixed = mixed.unsqueeze(0).to(device)
    mixed_resample = mixed_resample.unsqueeze(0).to(device)

    real, imag = model.stft(mixed)
    mag, cos, sin = magphase(real, imag)

    embed_pos = model.clap_model.get_text_embedding([query], use_tensor=True).to(device)
    if negative_query:
        embed_neg = model.clap_model.get_text_embedding([negative_query], use_tensor=True).to(device)
    else:
        embed_neg = torch.zeros_like(embed_pos)

    model.features.clear()
    model.features.append(mag)
    model.audio_branch({"waveform": mixed_resample})
    pred, _ = model.separate_from_features(
        embed_pos,
        embed_neg,
        mag,
        cos,
        sin,
        length=mixed.size(-1),
        features=model.features,
    )
    model.features.clear()
    return pred.squeeze(0).detach().cpu()


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


def infer(args):
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    model = build_model(args, device)
    model_sr = args.sr or args.train_data.get("sr") or args.val_data.get("sr")
    resample_rate = (
        args.resample_rate
        or args.train_data.get("resample_rate")
        or args.val_data.get("resample_rate")
    )
    if model_sr is None or resample_rate is None:
        raise ValueError("sr and resample_rate must be provided by config.json or command-line args.")

    wav, input_sr = load_audio_mono(args.audio_path)
    input_num_samples = wav.numel()
    wav = normalize_if_needed(resample_if_needed(wav, input_sr, model_sr))
    if input_sr != model_sr:
        logging.info("Resampled input from %d Hz to model sample rate %d Hz", input_sr, model_sr)

    chunk_size = model_sr * args.duration
    resampler = AT.Resample(model_sr, resample_rate)
    chunks, lengths = split_audio(wav, chunk_size)

    outputs = []
    for idx, (chunk, valid_len) in enumerate(zip(chunks, lengths), start=1):
        logging.info("Separating chunk %d/%d", idx, len(chunks))
        mixed_resample = resampler(chunk)
        pred = separate_chunk(
            model,
            chunk,
            mixed_resample,
            args.query,
            args.negative_query,
            device,
        )
        outputs.append(pred[:valid_len] if valid_len > 0 else pred[:0])

    separated = torch.cat(outputs, dim=0)
    separated = resample_if_needed(separated, model_sr, input_sr)
    separated = separated[:input_num_samples]
    if separated.numel() < input_num_samples:
        separated = torch.nn.functional.pad(separated, (0, input_num_samples - separated.numel()))
    separated = torch.clamp(separated, -1.0, 1.0).unsqueeze(0)
    torchaudio.save(args.output_path, separated, input_sr)
    logging.info("Saved separated audio to %s at %d Hz", args.output_path, input_sr)


def parse_args():
    parser = argparse.ArgumentParser(description="QSep-RL text-query target sound extraction.")
    parser.add_argument("--audio_path", required=True, help="Path to the mixed input audio.")
    parser.add_argument(
        "--query",
        required=True,
        help="Text query describing the target sound."
    )
    parser.add_argument(
        "--negative_query",
        default=None,
        help="Optional text query describing sound to suppress. Defaults to a zero negative embedding.",
    )
    parser.add_argument("--output_path", default="separated_target.wav", help="Path to save the extracted target.")
    parser.add_argument("--exp_dir", default="./experiments/QSep-RL", help="Directory containing config.json.")
    parser.add_argument("--checkpoint_path", default="./cp/best.pt", help="Path to a QSep-RL model checkpoint.")
    parser.add_argument(
        "--clap_path",
        default="path/to/music_audioset_epoch_15_esc_90.14.pt",
        help="Path to local LAION-CLAP checkpoint.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--duration", type=int, default=10, help="Chunk length in seconds. The model was trained on 10s.")
    parser.add_argument("--sr", type=int, default=None, help="Model sample rate. Defaults to train_data.sr.")
    parser.add_argument(
        "--resample_rate",
        type=int,
        default=None,
        help="Sample rate used by CLAP audio branch. Defaults to train_data.resample_rate.",
    )
    parser.add_argument("--verbose_clap", action="store_true", default=False)
    parser.add_argument(
        "--hf_hub_dir",
        default=os.path.expanduser("~/.cache/huggingface/hub"),
        help="Local HuggingFace hub cache directory used for CLAP tokenizers/text encoders.",
    )

    cli_args = parser.parse_args()
    provided_options = {
        token.split("=", 1)[0]
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    option_to_dest = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    provided_dests = {option_to_dest[option] for option in provided_options if option in option_to_dest}

    params = local_utils.Params(os.path.join(cli_args.exp_dir, "config.json"))
    for key, value in params.__dict__.items():
        if key not in provided_dests:
            vars(cli_args)[key] = value
    return cli_args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    infer(parse_args())
