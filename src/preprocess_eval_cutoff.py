import unsloth
import os
import sys
import json
import time
import pickle
import gzip
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import zstandard as zstd
import io
from tqdm.auto import tqdm

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_MAX_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["POLARS_MAX_THREADS"] = "4"
os.environ["SKIP_URL_GRAPH"] = "1"

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from ubm.text_representation_v3 import AdvancedUBMGenerator

# ====== KONFIGURATION: an deinen finalen target_days=75 Lauf angepasst ======
DATA_DIR = "retailrocket_data"
EVAL_DIR = "retailrocket_eval_sweep_75"          # dein finaler, gewaehlter Eval-Datensatz
CACHE_DIR = "retailrocket_data/cache_eval_cutoff75"  # NEUER, dedizierter Cache (nicht der alte!)
OUTPUT_DIR = "output_features/retailrocket_eval_cutoff75"
OBSERVATION_CUTOFF = datetime(2015, 7, 5, 2, 59, 47, 788000)  # exakt dein Sweep-75 Cutoff
CHUNK_SIZE = 500  # nur fuer Fortschrittsanzeige, keine Parallelisierung noetig

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"EVAL_DIR   : {EVAL_DIR}")
print(f"CACHE_DIR  : {CACHE_DIR}")
print(f"OUTPUT_DIR : {OUTPUT_DIR}")
print(f"CUTOFF     : {OBSERVATION_CUTOFF}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Nur 5 Clients zum Testen")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    client_ids = np.load(f"{EVAL_DIR}/input/relevant_clients.npy").astype(int)
    print(f"Relevant clients total: {len(client_ids):,}")

    if args.debug:
        client_ids = client_ids[:5]
        print(f"DEBUG MODE: {client_ids.tolist()}")
    elif args.limit and args.limit > 0:
        client_ids = client_ids[: args.limit]
        print(f"LIMIT MODE: {len(client_ids)} clients")

    # ------------------------------------------------------------
    # 1) EINMALIG: Daten laden MIT Cutoff, globale Stats berechnen/cachen
    # ------------------------------------------------------------
    print("\n=== Lade Daten mit Observation-Cutoff (einmalig) ===")
    t0 = time.time()
    gen = AdvancedUBMGenerator(DATA_DIR, CACHE_DIR, debug_mode=False)
    gen.load_data(
        use_cache=True,
        relevant_client_ids=None,          # globale Stats ueber die GESAMTE (cutoff-begrenzte) Historie
        observation_end=OBSERVATION_CUTOFF,
    )
    print(f"Laden abgeschlossen in {time.time()-t0:.1f}s")
    print(f"Reference time (muss dem Cutoff entsprechen): {gen.reference_time}")
    assert gen.reference_time == OBSERVATION_CUTOFF, "Cutoff stimmt nicht ueberein!"

    # ------------------------------------------------------------
    # 2) Profile fuer alle relevanten Clients generieren (in Chunks, wegen Fortschrittsanzeige)
    # ------------------------------------------------------------
    print(f"\n=== Generiere Profile fuer {len(client_ids):,} Clients ===")
    all_results = {}
    chunks = [client_ids[i : i + CHUNK_SIZE].tolist() for i in range(0, len(client_ids), CHUNK_SIZE)]

    t0 = time.time()
    for chunk in tqdm(chunks, desc="Profile generieren"):
        reps = gen.generate_representations(chunk, max_length=4096)
        for cid, json_str in reps.items():
            all_results[cid] = json.loads(json_str)

    elapsed = time.time() - t0
    print(f"Profile generiert in {elapsed/60:.1f} min ({len(all_results)/elapsed:.1f} clients/s)")

    features_path = f"{OUTPUT_DIR}/complete_features_{len(client_ids)}_clients.pkl"
    with open(features_path, "wb") as f:
        pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Gespeichert: {features_path}")

    # ------------------------------------------------------------
    # 3) Texte fuer Portrait-LLM vorbereiten
    # ------------------------------------------------------------
    texts_for_portraits = {}
    for cid, data in all_results.items():
        rich_text = data.get("rich_text", "")
        if rich_text:
            texts_for_portraits[cid] = rich_text

    print(f"\nTexte fuer Portraits: {len(texts_for_portraits):,}")
    texts_path = f"{OUTPUT_DIR}/texts_for_portraits_{len(texts_for_portraits)}.pkl"
    with open(texts_path, "wb") as f:
        pickle.dump(texts_for_portraits, f)
    print(f"Gespeichert: {texts_path}")

    del gen
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # 4) LLM-Portraits generieren (nutzt bereits vorhandene Multi-GPU-Logik)
    # ------------------------------------------------------------
    print("\n=== Generiere LLM-Portraits ===")
    from ubm.portrait_generator import generate_portraits

    t0 = time.time()
    portraits = generate_portraits(texts_for_portraits)
    print(f"Portraits generiert in {(time.time()-t0)/60:.1f} min: {len(portraits):,}")

    portraits_path = f"{OUTPUT_DIR}/portraits_{len(client_ids)}.pkl.gz"
    with gzip.open(portraits_path, "wb") as f:
        pickle.dump(portraits, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Gespeichert: {portraits_path}")

    # ------------------------------------------------------------
    # 5) Portraits mit rich_text zusammenfuehren
    # ------------------------------------------------------------
    def is_failure(txt: str) -> bool:
        return txt.lstrip().startswith(("- OOM", "- ERR", "Error on"))

    valid_portraits = {cid: p for cid, p in portraits.items() if not is_failure(p)}
    print(f"\nGueltige Portraits: {len(valid_portraits):,} / {len(portraits):,}")

    final_data = {}
    with_portrait = 0
    for cid, rich_text in texts_for_portraits.items():
        rich = rich_text
        if cid in valid_portraits:
            portrait = valid_portraits[cid].strip()
            insert = f"\n## PORTRAIT ##\n{portrait}\n"
            if "[END]" in rich:
                rich = rich.replace("[END]", insert + "[END]")
            else:
                rich += insert
            with_portrait += 1
        final_data[cid] = rich

    print(f"Clients mit Portrait: {with_portrait:,} / {len(final_data):,}")

    # ------------------------------------------------------------
    # 6) Finales complete_texts_*.jsonl.zst schreiben (Input fuer Qwen/Stella-Extraktion)
    # ------------------------------------------------------------
    MAX_PROMPT_CHARS = 12_000

    def strip_to_max(text, n=MAX_PROMPT_CHARS):
        return text[-n:] if len(text) > n else text

    output_file = Path(OUTPUT_DIR) / f"complete_texts_{len(final_data)}.jsonl.zst"
    print(f"\n=== Schreibe {output_file} ===")

    zstd_compressor = zstd.ZstdCompressor(level=3)
    with open(output_file, "wb") as f_out:
        with zstd_compressor.stream_writer(f_out) as compressor:
            with io.TextIOWrapper(compressor, encoding="utf-8") as writer:
                for cid, text in tqdm(final_data.items(), desc="Schreiben"):
                    final_text = strip_to_max(text)
                    record = {"id": int(cid), "text": final_text}
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nFertig! Datei: {output_file}")
    print(f"Groesse: {output_file.stat().st_size/1024/1024:.1f} MB")
    print(f"Records: {len(final_data):,}")


if __name__ == "__main__":
    main()