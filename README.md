# Narrative-Driven Universal Behavioral Modeling

Builds Universal Behavioral Profiles from raw e-commerce event logs: each user's interaction history is rewritten as a structured text document, which is then encoded into a dense vector by a language model. One profile per user serves several downstream prediction tasks — churn, category propensity, SKU propensity.

Fork of the RecSys Challenge 2025 submission by team *teitlax* ([paper](https://doi.org/10.1145/3758126.3758128)), extended to run on the [Retailrocket dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) in addition to the original Challenge data.

Scoring lives in a separate repository: [recsys2025](https://github.com/Camurra-PG/recsys2025).

## Structure

```
.
├── Dockerfile
├── Makefile                          # targets: data, features, finetune, extract, ensemble, clean
├── requirements.txt
├── download_data.sh
├── src/ubm/
│   ├── text_representation_v3.py     # feature extractors → rich text profiles
│   └── portrait_generator.py         # narrative distillation (COMPETITION_BRIEF / RETAILROCKET_BRIEF)
├── preprocess_eval_cutoff.py         # profile generation with an explicit observation cutoff
├── preprocessing_gemma1.py           # profile generation, Challenge dataset
├── train_gemma1.py                   # contrastive fine-tuning (InfoNCE + LoRA + projection head)
├── extract_embeddings_gemma1.py
├── extract_embeddings_qwen.py
├── extract_embeddings_stella.py
└── ensemble.py                       # weighted ensemble optimizer
```

## Setup

```bash
git clone https://github.com/Camurra-PG/Replication-of-Beyond-Model-Size--Narrative-Driven-Universal-Modeling.git
cd Replication-of-Beyond-Model-Size--Narrative-Driven-Universal-Modeling
docker build -t ndum-env .
export HF_TOKEN="your_token"        # needed for the Gemma checkpoints
```

Datasets are not included. Run `./download_data.sh` for the Challenge data; place the Retailrocket CSVs (`events.csv`, `item_properties_part*.csv`, `category_tree.csv`) under `ubc_data/retailrocket/`.

An 80 GB GPU is required — Qwen3-Embedding-8B alone needs ~16 GB for weights in half precision.

## Usage

Debug mode first. It restricts every stage to five clients and fails fast on a wrong path or data format.

```bash
docker run --rm -it --gpus all --shm-size=2g \
  --env HF_TOKEN=${HF_TOKEN} --env TORCH_COMPILE_DISABLE=1 \
  -v "$(pwd)":/app ndum-env \
  make all DEBUG=True
```

Full run on the Challenge dataset:

```bash
docker run --rm -it --gpus all --shm-size=2g \
  --env HF_TOKEN=${HF_TOKEN} --env TORCH_COMPILE_DISABLE=1 \
  -v "$(pwd)":/app ndum-env \
  make all DEBUG=False
```

Retailrocket:

```bash
python preprocess_eval_cutoff.py \
  --data-dir ubc_data/retailrocket \
  --observation-end 2015-07-05 \
  --output-dir output_features/retailrocket_cutoff75

make finetune DATASET=retailrocket STEPS=1100
make extract  DATASET=retailrocket
```

Output goes to `output_features/` (profiles), `models/` (checkpoints) and `embeddings/` (`client_ids.npy`, `embeddings.npy`).

## Two things worth knowing

**Use `preprocess_eval_cutoff.py` for Retailrocket, not `preprocessing_gemma1.py`.** The latter falls back to `reference_time = dataset_end`, so every recency and propensity feature sees into the window you are trying to predict. The cutoff script takes the boundary explicitly and asserts it before generating anything.

**The contrastive objective saturates early.** Accuracy passes 0.99 within ~700 of the configured 10,000 steps. Embeddings extracted at step 1100 outperform those from the finished run by 0.08–0.10 AUROC. The `Makefile` points at `checkpoint-1100`, but `--save_total_limit 5` deletes it long before a full run ends — stop training there deliberately if you want it.

## Hyperparameters

Set in the `Makefile`.

| | Challenge | Retailrocket |
|---|---|---|
| Base | `unsloth/gemma-3-1b-it-unsloth-bnb-4bit` | same |
| LoRA r / α | 16 / 32 | same |
| Projection dim | 2048 | same |
| InfoNCE temperature | 0.07 | same |
| Batch / grad accum | 24 / 4 | same |
| Max steps | 10,000 | 1,100 |
| Max length | 2048 | same |
| LR / warmup / decay | 2e-5 / 500 / 0.01 | same |
| Seed | 42 | same |

## Licence

Upstream licence applies to the inherited code. Datasets keep their own terms and are not redistributed here.
