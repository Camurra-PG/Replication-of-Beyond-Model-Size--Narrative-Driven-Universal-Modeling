import argparse
import gzip
import io
import json
import os
import pickle
import random
import statistics
from pathlib import Path

import numpy as np
import transformers
import zstandard as zstd
from tqdm.auto import tqdm


def load_pickle(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        return pickle.load(f)


def find_existing_file(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    raise FileNotFoundError("None of these files exist:\n" + "\n".join(str(p) for p in paths))


def is_failure(txt: str) -> bool:
    if not isinstance(txt, str):
        return True
    return txt.lstrip().startswith(("- OOM", "- ERR", "- Portrait generation failed"))


def augment_text(text: str, drop_prob: float, seed: int) -> str:
    random.seed(int(seed))
    lines = [line for line in text.splitlines() if line.strip()]
    keep_markers = ("##", "[CLIENT_", "[END]", "— FIN —", "[PROFILE]")

    kept = [
        line for line in lines
        if any(marker in line for marker in keep_markers) or random.random() > drop_prob
    ]

    if len(kept) < 2:
        kept = lines[:2]

    return "\n".join(kept)


def strip_to_max(text: str, n: int = 12_000) -> str:
    return text[-n:] if len(text) > n else text


def build_fallback_rich_text(cid, status="fallback"):
    return f"""[PROFILE]
## OVERVIEW ##
[CLIENT_{cid}]
User Type: inactive
Status: {status}

## CHURN_PROPENSITY ##
CHURN_RISK: Unknown

## TARGET_WINDOW_14D ##
No activity in last 14 days

## TEMPORAL ##
Inactive: No recorded activity

## CUSTOM ##
DEFAULT_USER: Fallback profile
Generated for completeness

[END]"""


def merge_features_and_portraits(features, portraits, expected_clients=None):
    valid_portraits = {
        int(cid): portrait
        for cid, portrait in portraits.items()
        if not is_failure(portrait)
    }

    print(f"Features loaded: {len(features):,}")
    print(f"Portraits loaded: {len(portraits):,}")
    print(f"Valid portraits: {len(valid_portraits):,}")

    final_data = {}
    no_json_count = 0
    no_portrait_count = 0
    skipped_non_dict = 0

    for cid, res in tqdm(features.items(), desc="Merging features + portraits"):
        cid = int(cid)

        if not isinstance(res, dict):
            skipped_non_dict += 1
            rich = build_fallback_rich_text(cid, "non_dict")
            final_data[cid] = {"profile": {"client_id": cid, "fallback": True}, "rich_text": rich}
            continue

        rich = ""

        if res.get("status") == "success" and "json_str" in res:
            try:
                jd = json.loads(res["json_str"])
                rich = jd.get("rich_text", "") or res.get("rich_text", "")
                profile = jd.get("profile", res.get("profile", {"client_id": cid}))
            except Exception:
                rich = res.get("rich_text", "")
                profile = res.get("profile", {"client_id": cid})
                no_json_count += 1
        else:
            rich = res.get("rich_text", "")
            profile = res.get("profile", {"client_id": cid})
            no_json_count += 1

        if not rich:
            rich = build_fallback_rich_text(cid, res.get("status", "missing_rich_text"))
            profile = {"client_id": cid, "fallback": True}

        if cid in valid_portraits:
            portrait = valid_portraits[cid].strip()
            insert = f"\n## PORTRAIT ##\n{portrait}\n"
            if "## PORTRAIT ##" not in rich:
                if "[END]" in rich:
                    rich = rich.replace("[END]", insert + "[END]")
                else:
                    rich += insert
        else:
            no_portrait_count += 1

        if "client_id" not in profile:
            profile["client_id"] = cid

        final_data[cid] = {
            "profile": profile,
            "rich_text": rich,
        }

    if expected_clients is not None and len(final_data) != expected_clients:
        print(f"Warning: final_data has {len(final_data):,}, expected {expected_clients:,}")

    print("\nMerge summary:")
    print(f"  final clients: {len(final_data):,}")
    print(f"  no_json/fallback-ish: {no_json_count:,}")
    print(f"  no valid portrait: {no_portrait_count:,}")
    print(f"  skipped non-dict: {skipped_non_dict:,}")
    print(f"  with portrait marker: {sum('## PORTRAIT ##' in d['rich_text'] for d in final_data.values()):,}")

    return final_data


def save_example_client(final_data, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    with_portrait = [cid for cid, d in final_data.items() if "## PORTRAIT ##" in d["rich_text"]]
    if with_portrait:
        cid = with_portrait[0]
    else:
        cid = next(iter(final_data.keys()))

    out = output_dir / f"example_client_{cid}.txt"
    out.write_text(final_data[cid]["rich_text"], encoding="utf-8")

    print(f"\nExample client saved:")
    print(f"  {out}")
    print(f"  chars: {len(final_data[cid]['rich_text']):,}")
    print(f"  has portrait: {'## PORTRAIT ##' in final_data[cid]['rich_text']}")


def save_complete_texts(final_data, output_dir: Path, drop_prob_aug1: float, max_prompt_chars: int):
    output_file = output_dir / "complete_texts_1M.jsonl.zst"
    print(f"\nSaving text dataset: {output_file}")

    cctx = zstd.ZstdCompressor(level=3, threads=-1)
    with_portrait_count = 0

    with open(output_file, "wb") as f_out:
        with cctx.stream_writer(f_out) as compressor:
            with io.TextIOWrapper(compressor, encoding="utf-8") as writer:
                for cid, data in tqdm(final_data.items(), desc="Writing complete_texts_1M"):
                    text = augment_text(data["rich_text"], drop_prob=drop_prob_aug1, seed=cid)
                    text = strip_to_max(text, max_prompt_chars)

                    record = {
                        "id": int(cid),
                        "text": text,
                    }
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")

                    if "## PORTRAIT ##" in text:
                        with_portrait_count += 1

    print("Text dataset saved:")
    print(f"  file: {output_file}")
    print(f"  size: {output_file.stat().st_size / 1024**2:.2f} MB")
    print(f"  records: {len(final_data):,}")
    print(f"  with portrait: {with_portrait_count:,}")


def tokenize_and_write(
    final_data,
    output_dir: Path,
    tokenizer_name: str,
    max_tokens: int,
    max_prompt_chars: int,
    drop_prob_aug1: float,
    use_aug2: bool,
    drop_prob_aug2: float,
    batch_clients: int,
):
    output_file = output_dir / "complete_dataset_1M.jsonl.zst"
    print(f"\nTokenizing and writing: {output_file}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Max tokens: {max_tokens}")
    print(f"Clients: {len(final_data):,}")
    print(f"Augmentations: {'aug1 + aug2' if use_aug2 else 'aug1 only'}")

    try:
        import orjson
        dumps = orjson.dumps
    except ImportError:
        dumps = lambda obj: json.dumps(obj, separators=(",", ":")).encode("utf-8")
        print("orjson not installed, using json fallback.")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        token=True,
    )

    cctx = zstd.ZstdCompressor(level=2, threads=-1)

    records_written = 0
    token_lens_sample = []
    ids = list(final_data.keys())

    with open(output_file, "wb") as fout:
        with cctx.stream_writer(fout) as zst:
            for start in tqdm(range(0, len(ids), batch_clients), desc="Tokenizing"):
                chunk_ids = ids[start:start + batch_clients]

                texts = []
                for cid in chunk_ids:
                    base = strip_to_max(final_data[cid]["rich_text"], max_prompt_chars)
                    aug1 = augment_text(base, drop_prob_aug1, seed=cid)

                    texts.append(base)
                    texts.append(aug1)

                    if use_aug2:
                        aug2 = augment_text(base, drop_prob_aug2, seed=cid + 1000)
                        texts.append(aug2)

                encoded = tokenizer(
                    texts,
                    truncation=True,
                    padding=False,
                    max_length=max_tokens,
                    return_attention_mask=False,
                )["input_ids"]

                pos = 0
                for cid in chunk_ids:
                    input_ids = encoded[pos]
                    input_ids_aug1 = encoded[pos + 1]
                    pos += 2

                    rec = {
                        "client_id": int(cid),
                        "input_ids": input_ids,
                        "input_ids_aug1": input_ids_aug1,
                    }

                    if use_aug2:
                        rec["input_ids_aug2"] = encoded[pos]
                        pos += 1

                    token_lens_sample.append(len(input_ids))
                    zst.write(dumps(rec) + b"\n")
                    records_written += 1

    avg_len = statistics.mean(token_lens_sample) if token_lens_sample else 0
    med_len = statistics.median(token_lens_sample) if token_lens_sample else 0

    print("\nTokenized dataset created:")
    print(f"  records written: {records_written:,}")
    print(f"  avg tokens: {avg_len:.0f}")
    print(f"  median tokens: {med_len:.0f}")
    print(f"  file: {output_file}")
    print(f"  size: {output_file.stat().st_size / 1024**2:.1f} MB")

    return output_file


def verify_dataset(path: Path, expected_records: int, sample_records: int = 5):
    print(f"\nVerifying dataset: {path}")

    dctx = zstd.ZstdDecompressor()
    total = 0
    unique_sample = set()
    token_lengths = []

    with open(path, "rb") as f:
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                if not line.strip():
                    continue

                rec = json.loads(line)
                total += 1

                if total <= sample_records:
                    unique_sample.add(rec["client_id"])
                    token_lengths.append(len(rec["input_ids"]))
                    print(f"  sample {total}: client_id={rec['client_id']}, tokens={len(rec['input_ids'])}, aug1={len(rec['input_ids_aug1'])}")

                elif total % 100000 == 0:
                    print(f"  checked {total:,} records...")

    print("\nVerification result:")
    print(f"  total records: {total:,}")
    print(f"  expected: {expected_records:,}")
    print(f"  status: {'OK' if total == expected_records else 'NOT OK'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output_features/gemma1b")
    parser.add_argument("--features", default=None)
    parser.add_argument("--portraits", default=None)
    parser.add_argument("--tokenizer", default="google/gemma-3-1b-it")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-prompt-chars", type=int, default=12000)
    parser.add_argument("--drop-prob-aug1", type=float, default=0.35)
    parser.add_argument("--drop-prob-aug2", type=float, default=0.50)
    parser.add_argument("--use-aug2", action="store_true")
    parser.add_argument("--batch-clients", type=int, default=2048)
    parser.add_argument("--skip-texts", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = Path(args.features) if args.features else find_existing_file([
        output_dir / "complete_features_1000000_clients.pkl",
        output_dir / "complete_features_1,000,000_clients.pkl",
    ])

    portraits_path = Path(args.portraits) if args.portraits else find_existing_file([
        output_dir / "portraits_1000000.pkl.gz",
        output_dir / "portraits_1,000,000.pkl.gz",
    ])

    print("Input files:")
    print(f"  features:  {features_path}")
    print(f"  portraits: {portraits_path}")

    print("\nLoading features...")
    features = load_pickle(features_path)

    print("Loading portraits...")
    portraits = load_pickle(portraits_path)

    final_data = merge_features_and_portraits(
        features=features,
        portraits=portraits,
        expected_clients=len(features),
    )

    save_example_client(final_data, output_dir)

    if not args.skip_texts:
        save_complete_texts(
            final_data=final_data,
            output_dir=output_dir,
            drop_prob_aug1=args.drop_prob_aug1,
            max_prompt_chars=args.max_prompt_chars,
        )

    dataset_path = tokenize_and_write(
        final_data=final_data,
        output_dir=output_dir,
        tokenizer_name=args.tokenizer,
        max_tokens=args.max_tokens,
        max_prompt_chars=args.max_prompt_chars,
        drop_prob_aug1=args.drop_prob_aug1,
        use_aug2=args.use_aug2,
        drop_prob_aug2=args.drop_prob_aug2,
        batch_clients=args.batch_clients,
    )

    if not args.skip_verify:
        verify_dataset(dataset_path, expected_records=len(final_data))

    print("\nDONE. Dataset is ready for training.")


if __name__ == "__main__":
    main()
import gzip
import io
import json
import pickle
import random
from pathlib import Path

import transformers
import zstandard as zstd
from tqdm.auto import tqdm


OUTPUT_DIR = Path("output_features/gemma1b")
FEATURES_PATH = OUTPUT_DIR / "complete_features_1000000_clients.pkl"
PORTRAITS_PATH = OUTPUT_DIR / "portraits_1000000.pkl.gz"

TOKENIZER_NAME = "google/gemma-3-1b-it"
MAX_TOKENS = 2048
MAX_PROMPT_CHARS = 12000
DROP_PROB_AUG1 = 0.35
BATCH_CLIENTS = 2048


def load_pickle(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        return pickle.load(f)


def is_failure(txt):
    if not isinstance(txt, str):
        return True

    s = str(txt).strip()
    lines = [l.strip() for l in s.splitlines() if l.strip()]
    content_lines = [l for l in lines if l != "— FIN —"]

    if not s:
        return True

    if s.lstrip().startswith(("- OOM", "- ERR", "- Portrait generation failed")):
        return True

    no_lines = sum(l in {"- No.", "No.", "- No"} for l in content_lines)

    if len(content_lines) == 0:
        return True

    if len(content_lines) <= 2 and len(s) < 80:
        return True

    if len(content_lines) >= 5 and no_lines / max(1, len(content_lines)) > 0.8:
        return True

    if len(s) < 120:
        return True

    return False


def augment_text(text, drop_prob, seed):
    random.seed(int(seed))
    lines = [line for line in text.splitlines() if line.strip()]
    keep_markers = ("##", "[CLIENT_", "[END]", "— FIN —", "[PROFILE]")

    kept = []
    for line in lines:
        if any(marker in line for marker in keep_markers) or random.random() > drop_prob:
            kept.append(line)

    if len(kept) < 2:
        kept = lines[:2]

    return "\n".join(kept)


def strip_to_max(text, n=MAX_PROMPT_CHARS):
    return text[-n:] if len(text) > n else text


def fallback_text(cid, status="fallback"):
    return f"""[PROFILE]
## OVERVIEW ##
[CLIENT_{cid}]
User Type: inactive
Status: {status}

## CHURN_PROPENSITY ##
CHURN_RISK: Unknown

## TARGET_WINDOW_14D ##
No activity in last 14 days

## TEMPORAL ##
Inactive: No recorded activity

## CUSTOM ##
DEFAULT_USER: Fallback profile

[END]"""


def get_rich_and_profile(cid, res):
    if not isinstance(res, dict):
        return fallback_text(cid, "non_dict"), {"client_id": cid, "fallback": True}

    rich = ""
    profile = res.get("profile", {"client_id": cid})

    if res.get("status") == "success" and "json_str" in res:
        try:
            jd = json.loads(res["json_str"])
            rich = jd.get("rich_text", "") or res.get("rich_text", "")
            profile = jd.get("profile", profile)
        except Exception:
            rich = res.get("rich_text", "")
    else:
        rich = res.get("rich_text", "")

    if not rich:
        rich = fallback_text(cid, res.get("status", "missing_rich_text"))
        profile = {"client_id": cid, "fallback": True}

    if "client_id" not in profile:
        profile["client_id"] = cid

    return rich, profile


def main():
    print("Loading features...")
    features = load_pickle(FEATURES_PATH)
    print(f"Features: {len(features):,}")

    print("Loading portraits...")
    portraits_raw = load_pickle(PORTRAITS_PATH)
    portraits = {int(k): v for k, v in portraits_raw.items() if not is_failure(v)}
    print(f"Portraits raw: {len(portraits_raw):,}")
    print(f"Portraits valid: {len(portraits):,}")

    print("Building final texts...")
    final_texts = {}
    with_portrait = 0

    for cid, res in tqdm(features.items(), desc="Merging"):
        cid = int(cid)
        rich, profile = get_rich_and_profile(cid, res)

        if cid in portraits:
            portrait = portraits[cid].strip()
            insert = f"\n## PORTRAIT ##\n{portrait}\n"
            if "## PORTRAIT ##" not in rich:
                if "[END]" in rich:
                    rich = rich.replace("[END]", insert + "[END]")
                else:
                    rich += insert
            with_portrait += 1

        final_texts[cid] = strip_to_max(rich)

    print(f"Final clients: {len(final_texts):,}")
    print(f"With portrait: {with_portrait:,}")

    example_cid = next((cid for cid, txt in final_texts.items() if "## PORTRAIT ##" in txt), next(iter(final_texts)))
    example_path = OUTPUT_DIR / f"example_client_{example_cid}.txt"
    example_path.write_text(final_texts[example_cid], encoding="utf-8")
    print(f"Example saved: {example_path}")

    texts_path = OUTPUT_DIR / "complete_texts_1M.jsonl.zst"
    print(f"Writing texts: {texts_path}")
    cctx = zstd.ZstdCompressor(level=3, threads=-1)

    with open(texts_path, "wb") as fout:
        with cctx.stream_writer(fout) as zst:
            with io.TextIOWrapper(zst, encoding="utf-8") as writer:
                for cid, text in tqdm(final_texts.items(), desc="Writing texts"):
                    writer.write(json.dumps({"id": cid, "text": text}, ensure_ascii=False) + "\n")

    print("Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        use_fast=True,
        token=True,
    )

    dataset_path = OUTPUT_DIR / "complete_dataset_1M.jsonl.zst"
    print(f"Writing tokenized dataset: {dataset_path}")
    cctx = zstd.ZstdCompressor(level=2, threads=-1)

    ids = list(final_texts.keys())
    written = 0

    try:
        import orjson
        dumps = orjson.dumps
    except Exception:
        dumps = lambda obj: json.dumps(obj, separators=(",", ":")).encode("utf-8")

    with open(dataset_path, "wb") as fout:
        with cctx.stream_writer(fout) as zst:
            for start in tqdm(range(0, len(ids), BATCH_CLIENTS), desc="Tokenizing"):
                chunk_ids = ids[start:start + BATCH_CLIENTS]

                texts = []
                for cid in chunk_ids:
                    base = final_texts[cid]
                    aug1 = augment_text(base, DROP_PROB_AUG1, cid)
                    texts.append(base)
                    texts.append(aug1)

                enc = tokenizer(
                    texts,
                    truncation=True,
                    padding=False,
                    max_length=MAX_TOKENS,
                    return_attention_mask=False,
                )["input_ids"]

                pos = 0
                for cid in chunk_ids:
                    rec = {
                        "client_id": int(cid),
                        "input_ids": enc[pos],
                        "input_ids_aug1": enc[pos + 1],
                    }
                    pos += 2
                    zst.write(dumps(rec) + b"\n")
                    written += 1

    print(f"Written records: {written:,}")
    print(f"Dataset size: {dataset_path.stat().st_size / 1024**3:.2f} GiB")

    print("Verifying first records...")
    dctx = zstd.ZstdDecompressor()
    with open(dataset_path, "rb") as f:
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for i, line in enumerate(text_stream):
                rec = json.loads(line)
                print(f"sample {i+1}: client_id={rec['client_id']}, tokens={len(rec['input_ids'])}, aug1={len(rec['input_ids_aug1'])}")
                if i >= 4:
                    break

    print("DONE. Dataset is ready for training.")


if __name__ == "__main__":
    main()
