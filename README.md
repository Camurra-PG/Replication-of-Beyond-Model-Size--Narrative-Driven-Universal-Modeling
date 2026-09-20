# Narrative-Driven Universal Behavioral Modeling — Retailrocket Adaption

Code for the master's thesis *"Replicating a Narrative-Driven Universal Behavior Modeling Approach"* (Mutlu Orhan, University of Klagenfurt, supervised by Prof. Dietmar Jannach).

This repository is a **fork of the RecSys Challenge 2025 submission by team *teitlax*** ([Rousseau & Veyssiere, 2025](https://doi.org/10.1145/3758126.3758128)). It contains the profile-generation, fine-tuning and embedding-extraction pipeline, in two states:

1. the original pipeline, re-run on the Challenge dataset to check whether the published results can be obtained by someone other than their authors;
2. the same pipeline adapted to the [Retailrocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset), which it was never designed for.

The evaluation harness lives in a **separate repository**: [`Camurra-PG/recsys2025`](https://github.com/Camurra-PG/recsys2025).

---

## What is original and what is mine

The commit history separates the two. The initial commit is the unmodified upstream state; everything after it was written for this thesis.

| File | Status | What changed |
|---|---|---|
| `src/ubm/text_representation_v3.py` | **heavily modified** | Feature extractors for Retailrocket; `observation_end` handling; cache validation |
| `src/ubm/portrait_generator.py` | **modified** | New `RETAILROCKET_BRIEF` prompt; `spawn` process pool; token limits raised |
| `preprocess_eval_cutoff.py` | **new** | Leakage-free profile generation with an explicit, asserted observation cutoff |
| `extract_embeddings_gemma1.py` | **modified** | `TokenizedDataset` fallback for raw-text records |
| `train_gemma1.py` | unchanged logic | Only hyperparameters and step count differ for Retailrocket |
| `extract_embeddings_qwen.py`, `extract_embeddings_stella.py` | unchanged | Used as-is |
| `ensemble.py` | unchanged | Not used in the thesis (see *Deviations* below) |

---

## Project structure

```
.
├── Dockerfile
├── Makefile                       # pipeline targets: data, features, finetune, extract, ensemble, clean
├── requirements.txt
├── download_data.sh
├── src/
│   └── ubm/
│       ├── text_representation_v3.py   # feature extractors + rich text profile generation
│       └── portrait_generator.py       # narrative distillation (COMPETITION_BRIEF / RETAILROCKET_BRIEF)
├── preprocess_eval_cutoff.py      # NEW: leakage-free profile generation for Retailrocket
├── preprocessing_gemma1.py        # original preprocessing (Challenge dataset)
├── preprocessing_gemma12.py
├── train_gemma1.py                # contrastive fine-tuning (InfoNCE + LoRA + projection head)
├── train_gemma12.py
├── extract_embeddings_gemma1.py
├── extract_embeddings_qwen.py
├── extract_embeddings_stella.py
└── ensemble.py                    # weighted ensemble optimizer (not used in the thesis)
```

---

## Where each part is described in the thesis

| Thesis section | Code |
|---|---|
| 3.2.2 — Global statistics and rich text profile generation | `src/ubm/text_representation_v3.py` |
| 3.2.3 — Narrative distillation | `src/ubm/portrait_generator.py` (`COMPETITION_BRIEF`) |
| 3.2.4 — Embedding extraction | `train_gemma1.py`, `extract_embeddings_*.py` |
| 3.2.5 — Ensemble optimization | `ensemble.py` |
| 3.2.6 — Development environment, hyperparameters | `Makefile`, `Dockerfile` |
| 3.3.2 — Adapting the feature extractors | `src/ubm/text_representation_v3.py` |
| 3.3.3 — Adapting the distillation prompt | `src/ubm/portrait_generator.py` (`RETAILROCKET_BRIEF`) |
| 3.3.5 — Temporal data leakage and its correction | `preprocess_eval_cutoff.py` |

---

## The eleven feature extractors

Each section of a user profile is produced by one extractor. Five were removed for Retailrocket because the underlying data does not exist there, and two were added.

| Extractor | Contributes | On Retailrocket |
|---|---|---|
| `temporal` | Peak hours, active weekdays, recency | retained |
| `sequence` | Common event-type transitions | retained |
| `graph` | Co-occurrence and centrality indicators | retained |
| `intent` | Funnel stage, cart-abandonment signals | partly retained (no search intent) |
| `price` | Explored price-bucket range | **removed** — no validated price field |
| `social` | Popularity affinity relative to population | retained |
| `name_embedding` | Quantized product-name embeddings | **removed** — no product names |
| `churn_propensity` | Purchase recency, churn indicators | retained |
| `top_sku` | Ranked per-client SKU propensity | **removed** |
| `top_category` | Ranked per-client category propensity | **removed** |
| `custom_behavior` | Cart conversion rate, history span | retained |
| `availability` | In-stock vs. out-of-stock interactions | **added for Retailrocket** |
| `global_popularity` | Overlap with dataset-wide popular items | **added for Retailrocket** |

---

## Setup

### 1. Clone

```bash
git clone https://github.com/Camurra-PG/Replication-of-Beyond-Model-Size--Narrative-Driven-Universal-Modeling.git
cd Replication-of-Beyond-Model-Size--Narrative-Driven-Universal-Modeling
```

### 2. Build the container

```bash
docker build -t ndum-env .
```

### 3. Hugging Face token

Required to download the Gemma checkpoints.

```bash
export HF_TOKEN="your_token"
```

### 4. Data

The datasets are **not** included here, for licence reasons.

- **RecSys 2025 Challenge dataset** — https://recsys.synerise.com/data-set (CC BY-NC 4.0)
- **Retailrocket dataset** — https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

```bash
./download_data.sh        # Challenge dataset
```

Place the Retailrocket CSV files (`events.csv`, `item_properties_part1.csv`, `item_properties_part2.csv`, `category_tree.csv`) under `ubc_data/retailrocket/`.

---

## Running the pipeline

### Challenge dataset (full run)

```bash
docker run --rm -it --gpus all \
  --shm-size=2g \
  --env HF_TOKEN=${HF_TOKEN} \
  --env TORCH_COMPILE_DISABLE=1 \
  -v "$(pwd)":/app \
  ndum-env \
  make all DEBUG=False
```

### Debug mode (five clients, minutes instead of hours)

Always run this first. It restricts every stage to five fixed clients and fails fast on a wrong path or data format.

```bash
docker run --rm -it --gpus all \
  --shm-size=2g \
  --env HF_TOKEN=${HF_TOKEN} \
  --env TORCH_COMPILE_DISABLE=1 \
  -v "$(pwd)":/app \
  ndum-env \
  make all DEBUG=True
```

### Retailrocket (leakage-free profiles)

**Do not use `preprocessing_gemma1.py` for Retailrocket.** It falls back to `reference_time = dataset_end`, which lets every recency and propensity feature see into the target window. Use the dedicated script instead, which takes an explicit cutoff and asserts that the resulting `reference_time` matches it before generating anything:

```bash
python preprocess_eval_cutoff.py \
  --data-dir ubc_data/retailrocket \
  --observation-end 2015-07-05 \
  --output-dir output_features/retailrocket_cutoff75
```

Then fine-tune and extract:

```bash
make finetune DATASET=retailrocket STEPS=1100
make extract  DATASET=retailrocket
```

### Output layout

```
.
├── ubc_data/               # raw and preprocessed event files
├── output_features/        # rich and enriched text profiles
├── models/                 # fine-tuning checkpoints
├── embeddings/             # extracted embedding arrays (client_ids.npy, embeddings.npy)
└── unsloth_compiled_cache/
```

---

## Fine-tuning hyperparameters

Fixed in the `Makefile`; the original paper does not state them.

| Setting | Challenge | Retailrocket |
|---|---|---|
| Base checkpoint | `unsloth/gemma-3-1b-it-unsloth-bnb-4bit` | same |
| Quantization | 4-bit (QLoRA) | same |
| LoRA rank *r* / α | 16 / 32 | same |
| Projection dimension | 2048 | same |
| InfoNCE temperature | 0.07 | same |
| Batch size / grad. accumulation | 24 / 4 | same |
| **Max steps** | **10,000** | **1,100** |
| Max sequence length | 2048 tokens | same |
| Learning rate | 2e-5 | same |
| Warmup steps / weight decay | 500 / 0.01 | same |
| Seed | 42 | same |

**Note on checkpoint selection.** The contrastive objective saturates within roughly the first 700 steps (accuracy > 0.99, loss < 0.002). Embeddings extracted at step 1100 outperform those from the completed 10,000-step run by 0.08–0.10 AUROC on every task. The `Makefile` references `checkpoint-1100`, but `--save_total_limit 5` deletes it long before a full run ends — a full run leaves only checkpoints 9600–10000. If you want the early checkpoint, stop training at step 1100 deliberately.

---

## Deviations from the published pipeline

Both were made for compute reasons and are discussed in Section 3.2.10 of the thesis.

- **Gemma-3-12B was not fine-tuned.** Only Gemma-3-1B was. A 12B run was estimated at several hundred GPU-hours.
- **No weighted ensemble was fitted.** All embedding sources are evaluated as standalone comparison points, so `ensemble.py` is unused here. Results are therefore comparable to the individual-model rows of the original paper, **not** to its ensemble row.

A third point is an addition rather than an omission: **BGE** (`BGE-large-en-v1.5` on the Challenge dataset, `BAAI/bge-multilingual-gemma2` on Retailrocket) was added as a diagnostic after `stella_en_400M_v5` scored near chance level. It is never part of an ensemble.

---

## Hardware

Developed and run on rented RunPod instances.

- **Feature and profile generation** — 4 × A100
- **Fine-tuning, extraction, evaluation** — 1 × H100 80 GB

An 80 GB accelerator is required: Qwen3-Embedding-8B alone needs roughly 16 GB for its weights in half precision, before activations. Fine-tuning took about 36 hours; extracting Qwen embeddings for one million profiles took about 13 hours at ~21 profiles/second.

Run long jobs inside `tmux` — a dropped SSH connection otherwise kills them.

---

## Citation

```bibtex
@mastersthesis{orhan2026narrative,
  author = {Orhan, Mutlu},
  title  = {Replicating a Narrative-Driven Universal Behavior Modeling Approach},
  school = {University of Klagenfurt},
  year   = {2026}
}
```

Original approach:

```bibtex
@inproceedings{rousseau2025beyond,
  author    = {Rousseau, Alexandre and Veyssiere, Yann},
  title     = {Beyond Model Size: Narrative Driven Universal Modeling},
  booktitle = {Proceedings of the Recommender Systems Challenge 2025},
  pages     = {12--15},
  year      = {2025},
  doi       = {10.1145/3758126.3758128}
}
```

## Licence

The upstream licence applies to the inherited code. Both datasets keep their own terms and are not redistributed here.
