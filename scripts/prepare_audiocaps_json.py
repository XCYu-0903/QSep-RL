import argparse
import csv
import json
from pathlib import Path


def read_table(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, skipinitialspace=True))


def norm_id(value):
    return str(value or "").strip()


def build_audio_index(audio_root, extensions):
    index = {}
    audio_root = Path(audio_root)
    for ext in extensions:
        for path in audio_root.rglob(f"*{ext}"):
            stem = path.stem
            keys = {path.name, stem}
            if stem.startswith("Y"):
                keys.add(stem[1:])
            parts = stem.split("_")
            if parts:
                keys.add(parts[0])
                if parts[0].startswith("Y"):
                    keys.add(parts[0][1:])
            for key in keys:
                index.setdefault(key, str(path.resolve()))
    return index


def candidate_keys(row):
    keys = []
    for field in ("path", "file_name", "filename", "wav", "audio"):
        value = norm_id(row.get(field))
        if value:
            path = Path(value)
            keys.extend([value, path.name, path.stem])
    youtube_id = norm_id(row.get("youtube_id") or row.get("ytid") or row.get("youtube"))
    start_time = norm_id(row.get("start_time") or row.get("start_seconds") or row.get("start"))
    if youtube_id:
        keys.append(youtube_id)
        keys.append(f"Y{youtube_id}")
        if start_time:
            start_clean = start_time.replace(".", "_")
            keys.extend(
                [
                    f"{youtube_id}_{start_time}",
                    f"Y{youtube_id}_{start_time}",
                    f"{youtube_id}_{start_clean}",
                    f"Y{youtube_id}_{start_clean}",
                ]
            )
    return keys


def get_caption(row):
    for key in ("caption", "caption_1", "text", "sentence"):
        value = norm_id(row.get(key))
        if value:
            return value
    raise ValueError(f"Cannot find caption column in row: {row}")


def resolve_audio(row, audio_index):
    direct_path = norm_id(row.get("path"))
    if direct_path and Path(direct_path).is_file():
        return str(Path(direct_path).resolve())
    for key in candidate_keys(row):
        if key in audio_index:
            return audio_index[key]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audiocaps_csv", required=True, help="AudioCaps train/val CSV with captions.")
    parser.add_argument("--audio_root", required=True, help="Root directory containing downloaded AudioSet/AudioCaps audio.")
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".wav", ".flac", ".mp3", ".m4a", ".ogg"],
        help="Audio extensions to index recursively.",
    )
    parser.add_argument(
        "--missing_json",
        default=None,
        help="Optional path to save rows whose audio could not be matched.",
    )
    args = parser.parse_args()

    rows = read_table(args.audiocaps_csv)
    audio_index = build_audio_index(args.audio_root, args.extensions)
    output = []
    missing = []
    for row in rows:
        path = resolve_audio(row, audio_index)
        if path is None:
            missing.append(row)
            continue
        file_name = norm_id(row.get("file_name") or row.get("filename") or Path(path).name)
        output.append(
            {
                "file_name": file_name,
                "path": path,
                "caption": get_caption(row),
                "mids": norm_id(row.get("mids")),
                "class_names": norm_id(row.get("class_names")),
            }
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if args.missing_json:
        missing_path = Path(args.missing_json)
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        with open(missing_path, "w", encoding="utf-8") as f:
            json.dump(missing, f, ensure_ascii=False, indent=2)

    print(f"matched={len(output)} missing={len(missing)} output={output_path}")


if __name__ == "__main__":
    main()
