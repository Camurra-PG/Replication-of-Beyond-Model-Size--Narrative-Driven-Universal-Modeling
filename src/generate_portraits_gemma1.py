#!/usr/bin/env python3
import os
import sys
import gc
import gzip
import time
import pickle
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def setup_logging(output_dir: Path, run_id: str) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / f"portraits_gemma1_main_{run_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)],
    )

def build_worker_code() -> str:
    return r'''
import os
import sys
import gc
import time
import pickle
from pathlib import Path
from datetime import datetime

gpu_id = int(sys.argv[1])
input_file = sys.argv[2]
output_file = sys.argv[3]
checkpoint_file = sys.argv[4]
batch_size = int(sys.argv[5])
checkpoint_every = int(sys.argv[6])

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.getcwd() + "/src"))
from ubm.portrait_generator import PortraitGenerator

print(f"[GPU {gpu_id}] Start {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
print(f"[GPU {gpu_id}] Loading input: {input_file}", flush=True)

with open(input_file, "rb") as f:
    items = pickle.load(f)

results = {}
start_index = 0

ckpt = Path(checkpoint_file)
if ckpt.exists():
    print(f"[GPU {gpu_id}] Resuming from checkpoint: {ckpt}", flush=True)
    with ckpt.open("rb") as f:
        data = pickle.load(f)
    results = data.get("results", {})
    start_index = data.get("last_index", 0)
    print(f"[GPU {gpu_id}] Loaded {len(results)} existing portraits, start_index={start_index}", flush=True)

gen = PortraitGenerator("cuda:0")

items_remaining = items[start_index:]
total_batches = (len(items_remaining) + batch_size - 1) // batch_size

pbar = tqdm(
    range(0, len(items_remaining), batch_size),
    total=total_batches,
    desc=f"GPU {gpu_id}",
)

recent_times = []

def save_checkpoint(local_i: int):
    absolute_index = start_index + local_i
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tmp_ckpt = ckpt.with_suffix(".tmp")
    with tmp_ckpt.open("wb") as f:
        pickle.dump(
            {
                "results": results,
                "last_index": absolute_index,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    tmp_ckpt.replace(ckpt)
    print(f"[GPU {gpu_id}] Checkpoint saved: {len(results)} portraits, last_index={absolute_index}", flush=True)

for batch_no, local_i in enumerate(pbar, start=1):
    t0 = time.time()
    batch = items_remaining[local_i:local_i + batch_size]

    try:
        portraits = gen.generate_batch(batch)

    except torch.cuda.OutOfMemoryError:
        print(f"[GPU {gpu_id}] OOM at batch size {len(batch)}; retrying half batches", flush=True)
        torch.cuda.empty_cache()
        gc.collect()

        portraits = {}
        step = max(1, len(batch) // 2)
        for j in range(0, len(batch), step):
            sub = batch[j:j + step]
            try:
                portraits.update(gen.generate_batch(sub))
            except Exception as e:
                print(f"[GPU {gpu_id}] Failed subbatch: {type(e).__name__}: {e}", flush=True)
                for cid, _ in sub:
                    portraits[cid] = f"- ERR: {type(e).__name__}: {str(e)[:120]}\n— FIN —"

    except Exception as e:
        print(f"[GPU {gpu_id}] Failed batch: {type(e).__name__}: {e}", flush=True)
        portraits = {}
        for cid, _ in batch:
            portraits[cid] = f"- ERR: {type(e).__name__}: {str(e)[:120]}\n— FIN —"

    results.update(portraits)

    dt = time.time() - t0
    recent_times.append(dt)
    if len(recent_times) > 20:
        recent_times.pop(0)

    avg = sum(recent_times) / len(recent_times)
    remaining_batches = total_batches - batch_no
    eta_min = remaining_batches * avg / 60 if avg > 0 else 0

    pbar.set_postfix(
        batch_s=f"{dt:.1f}",
        total=len(results),
        eta_min=f"{eta_min:.1f}",
    )

    if batch_no % checkpoint_every == 0:
        save_checkpoint(local_i + len(batch))

    if batch_no % 25 == 0:
        torch.cuda.empty_cache()
        gc.collect()

with open(output_file, "wb") as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

if ckpt.exists():
    ckpt.unlink()

print(f"[GPU {gpu_id}] Done. Wrote {len(results)} portraits to {output_file}", flush=True)
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--texts", default="output_features/gemma1b/texts_for_portraits_1000000.pkl")
    parser.add_argument("--out", default="output_features/gemma1b/portraits_1000000.pkl.gz")
    parser.add_argument("--batch-size", type=int, default=180)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is not set. Run: export HF_TOKEN='your_token'")

    project_root = Path.cwd()
    output_dir = Path("output_features/gemma1b")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(output_dir, run_id)

    texts_path = Path(args.texts)
    if not texts_path.exists():
        raise FileNotFoundError(f"Missing texts file: {texts_path}")

    logging.info("Loading texts from %s", texts_path)
    with texts_path.open("rb") as f:
        texts = pickle.load(f)

    if args.limit is not None:
        texts = dict(list(texts.items())[:args.limit])

    items = list(texts.items())
    del texts
    gc.collect()

    logging.info("Texts loaded: %s", f"{len(items):,}")

    import torch
    n_gpus = torch.cuda.device_count()
    if n_gpus <= 0:
        raise RuntimeError("No CUDA GPUs detected.")

    logging.info("GPUs detected: %d", n_gpus)
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        logging.info("GPU %d: %s", i, props.name)

    temp_dir = output_dir / "portrait_tmp_gemma1"
    ckpt_dir = output_dir / "portrait_checkpoints_gemma1"
    logs_dir = output_dir / "logs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    per_gpu, remainder = divmod(len(items), n_gpus)
    assignments = []

    start = 0
    for gpu_id in range(n_gpus):
        count = per_gpu + (1 if gpu_id < remainder else 0)
        end = start + count
        shard = items[start:end]

        input_file = temp_dir / f"gpu_{gpu_id}_input.pkl"
        output_file = temp_dir / f"gpu_{gpu_id}_output.pkl"
        checkpoint_file = ckpt_dir / f"gpu_{gpu_id}_checkpoint.pkl"

        with input_file.open("wb") as f:
            pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)

        assignments.append(
            {
                "gpu_id": gpu_id,
                "input": input_file,
                "output": output_file,
                "checkpoint": checkpoint_file,
                "count": count,
            }
        )

        logging.info("GPU %d assigned %s clients", gpu_id, f"{count:,}")
        start = end

    worker_code = build_worker_code()

    procs = []
    for a in assignments:
        log_file = logs_dir / f"portraits_gemma1_gpu_{a['gpu_id']}_{run_id}.log"
        cmd = [
            sys.executable,
            "-c",
            worker_code,
            str(a["gpu_id"]),
            str(a["input"]),
            str(a["output"]),
            str(a["checkpoint"]),
            str(args.batch_size),
            str(args.checkpoint_every),
        ]

        log_handle = log_file.open("w")
        p = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, cwd=str(project_root))
        procs.append((p, a, log_handle, log_file))

        logging.info("Started GPU %d pid=%d log=%s", a["gpu_id"], p.pid, log_file)
        time.sleep(2)

    failed = False
    while True:
        running = 0
        for p, a, log_handle, log_file in procs:
            ret = p.poll()
            if ret is None:
                running += 1
            elif ret != 0:
                failed = True

        if running == 0:
            break

        logging.info("Still running: %d/%d GPU workers", running, len(procs))
        time.sleep(60)

    for p, a, log_handle, log_file in procs:
        log_handle.close()
        ret = p.returncode
        if ret != 0:
            logging.error("GPU %d failed with code %s. Check log: %s", a["gpu_id"], ret, log_file)

    if failed:
        raise RuntimeError("At least one GPU worker failed. Check output_features/gemma1b/logs/")

    logging.info("Merging outputs...")
    portraits = {}
    stats = defaultdict(int)

    for a in assignments:
        if not a["output"].exists():
            raise FileNotFoundError(f"Missing GPU output: {a['output']}")

        with a["output"].open("rb") as f:
            part = pickle.load(f)

        portraits.update(part)
        stats[a["gpu_id"]] = len(part)
        logging.info("GPU %d output: %s portraits", a["gpu_id"], f"{len(part):,}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Saving final portraits to %s", out_path)
    with gzip.open(out_path, "wb") as f:
        pickle.dump(portraits, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Compatibility filename used by some old code
    compat_path = output_dir / f"portraits_{len(portraits):,}.pkl.gz"
    if compat_path != out_path:
        logging.info("Also saving compatibility copy to %s", compat_path)
        with gzip.open(compat_path, "wb") as f:
            pickle.dump(portraits, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info("Done. Total portraits: %s", f"{len(portraits):,}")
    logging.info("Final file size: %.2f GiB", out_path.stat().st_size / 1024**3)

    for a in assignments:
        a["input"].unlink(missing_ok=True)
        a["output"].unlink(missing_ok=True)

if __name__ == "__main__":
    main()
