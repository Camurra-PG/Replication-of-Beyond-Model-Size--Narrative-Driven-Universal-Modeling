#!/usr/bin/env python3
"""
Embedding extraction with BAAI/bge-multilingual-gemma2.

This script is intentionally very close to the Qwen/Stella extractors:
- reads local JSONL.ZST with fields: id, text
- cleans client identifiers
- applies an instruction prompt
- uses sentence-transformers encode()
- saves client_ids.npy, embeddings.npy and submission.zip

Recommended default model:
    BAAI/bge-multilingual-gemma2

Why this model:
    It is a LLM-based multilingual embedding model based on Gemma2-9B,
    so it is closer to Qwen3-Embedding-8B than small encoder models like Stella.
"""

import argparse
import gc
import io
import json
import logging
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import zstandard as zstd
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class BGEMultilingualGemmaEmbeddingExtractor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_path = self.args.dataset_path

        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=== CONFIGURATION ===")
        logger.info(f"Mode         : {'DEBUG' if args.debug else 'PRODUCTION'}")
        logger.info(f"Model        : {self.args.model_name}")
        logger.info(f"Device       : {self.device}")
        logger.info(f"Batch size   : {self.args.batch_size}")
        logger.info(f"Max length   : {self.args.max_length}")
        logger.info(f"Embedding dim: {self.args.embedding_dim}")
        logger.info(f"Input file   : {self.input_path}")
        logger.info(f"Output dir   : {self.output_dir.resolve()}/")

    def load_model(self):
        """Load embedding model through sentence-transformers."""
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading model {self.args.model_name} with sentence-transformers...")

        hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
            logger.info("HuggingFace authentication configured")

        model_kwargs = {
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if torch.cuda.is_available():
            model_kwargs["device_map"] = {"": 0}

        tokenizer_kwargs = {"padding_side": "left"}

        self.sentence_model = SentenceTransformer(
            self.args.model_name,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )

        # Use the max_length argument. In your old scripts it was logged but not actually applied.
        try:
            self.sentence_model.max_seq_length = self.args.max_length
            logger.info(f"Set sentence_model.max_seq_length = {self.args.max_length}")
        except Exception as exc:
            logger.warning(f"Could not set max_seq_length: {exc}")

        self.model = self.sentence_model[0].auto_model
        self.tokenizer = self.sentence_model[0].tokenizer

        logger.info("Model loaded successfully!")

    def read_local_jsonl_zst(self, limit=None):
        """Read a local compressed JSONL.ZST file."""
        logger.info(f"Opening local file {self.input_path}...")
        with open(self.input_path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                count = 0

                for line in text_stream:
                    if not line.strip():
                        continue

                    yield json.loads(line)
                    count += 1

                    if limit and count >= limit:
                        break

    def clean_text(self, text, client_id=None):
        """Remove client identifiers and normalize whitespace."""
        patterns = [
            r"\[CLIENT_\d+\]",
            r"CLIENT_\d+",
            r"client_id:\s*\d+",
            r'"client_id":\s*\d+,?',
            r"'client_id':\s*\d+,?",
        ]

        if client_id is not None:
            patterns += [
                f"\\[CLIENT_{client_id}\\]",
                f"CLIENT_{client_id}\\b",
                f"\\b{client_id}\\b",
            ]

        cleaned = str(text)

        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def get_detailed_instruct(self, task_desc, query):
        """Instruction format kept identical to your Qwen/Stella scripts."""
        return f"Instruct: {task_desc}\nQuery:{query}"

    def get_embeddings_batch(self, texts):
        """Generate one batch of embeddings."""
        task = "Generate a dense behavioral representation for this user profile"
        instructed_texts = [self.get_detailed_instruct(task, text) for text in texts]

        try:
            embeddings = self.sentence_model.encode(
                instructed_texts,
                batch_size=self.args.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                device=self.device,
            )

            embeddings = embeddings.astype(np.float16)
            current_dim = embeddings.shape[1]

            if current_dim != self.args.embedding_dim:
                if current_dim < self.args.embedding_dim:
                    pad = np.zeros(
                        (embeddings.shape[0], self.args.embedding_dim - current_dim),
                        dtype=np.float16,
                    )
                    embeddings = np.concatenate([embeddings, pad], axis=1)
                    logger.warning(
                        f"Padded embeddings from {current_dim} to {self.args.embedding_dim}"
                    )
                else:
                    embeddings = embeddings[:, : self.args.embedding_dim]
                    logger.warning(
                        f"Truncated embeddings from {current_dim} to {self.args.embedding_dim}"
                    )

            # Normalize again after padding/truncation.
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

            return embeddings.astype(np.float16)

        except Exception as exc:
            # Do NOT silently return random embeddings in production.
            # Random embeddings make evaluation results meaningless and hide real errors.
            logger.exception(f"Error in get_embeddings_batch: {exc}")
            if self.args.allow_random_fallback:
                logger.warning("Using random fallback embeddings because --allow_random_fallback is set")
                fallback = np.random.randn(len(texts), self.args.embedding_dim).astype(np.float16)
                norms = np.linalg.norm(fallback, axis=1, keepdims=True)
                return fallback / (norms + 1e-8)
            raise

    def process_dataset(self):
        """Main processing loop."""
        self.load_model()
        limit = 5 if self.args.debug else None

        logger.info("Counting records...")
        total = sum(1 for _ in self.read_local_jsonl_zst(limit=limit))
        logger.info(f"Will process {total:,} records")

        all_ids = []
        all_embeddings = []
        batch_texts = []
        batch_ids = []
        processed = 0
        lengths = []

        pbar = tqdm(
            self.read_local_jsonl_zst(limit=limit),
            total=total,
            desc="Extracting embeddings",
        )

        for record in pbar:
            client_id = record.get("id")
            text = record.get("text", "")

            if client_id is None or not text:
                continue

            cleaned = self.clean_text(text, client_id)
            if not cleaned:
                continue

            lengths.append(len(cleaned))
            batch_texts.append(cleaned)
            batch_ids.append(client_id)

            if len(batch_texts) >= self.args.batch_size:
                embeddings = self.get_embeddings_batch(batch_texts)

                all_ids.extend(batch_ids)
                all_embeddings.append(embeddings)
                processed += len(batch_texts)

                pbar.set_postfix(
                    {
                        "processed": processed,
                        "batch_size": len(batch_texts),
                        "gpu_mem": (
                            f"{torch.cuda.memory_allocated() / 1e9:.1f}GB"
                            if torch.cuda.is_available()
                            else "N/A"
                        ),
                    }
                )

                batch_texts = []
                batch_ids = []

                if processed % 10000 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        if batch_texts:
            embeddings = self.get_embeddings_batch(batch_texts)
            all_ids.extend(batch_ids)
            all_embeddings.append(embeddings)
            processed += len(batch_texts)

        client_ids = np.array(all_ids, dtype=np.int64)
        embeddings_array = np.vstack(all_embeddings).astype(np.float16)

        logger.info("=== EXTRACTION STATS ===")
        logger.info(f"Total processed : {processed:,}")
        logger.info(f"IDs shape       : {client_ids.shape}, dtype={client_ids.dtype}")
        logger.info(
            f"Embeddings shape: {embeddings_array.shape}, dtype={embeddings_array.dtype}"
        )

        if lengths:
            logger.info(
                f"Text length     : mean={np.mean(lengths):.0f}, std={np.std(lengths):.0f}"
            )

        self.save_results(client_ids, embeddings_array)

    def save_results(self, client_ids, embeddings):
        """Save output files."""
        logger.info("=== SAVING RESULTS ===")

        ids_path = self.output_dir / "client_ids.npy"
        embeddings_path = self.output_dir / "embeddings.npy"

        np.save(ids_path, client_ids)
        np.save(embeddings_path, embeddings)

        logger.info(f"Saved: {ids_path}")
        logger.info(f"Saved: {embeddings_path}")

        zip_path = self.output_dir / "submission.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(ids_path, "client_ids.npy")
            zf.write(embeddings_path, "embeddings.npy")

        logger.info(f"Created archive: {zip_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings with BAAI/bge-multilingual-gemma2"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-multilingual-gemma2",
        help="HuggingFace model name",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--embedding_dim", type=int, default=2048)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", type=str, default="embeddings/bge-multilingual-gemma2/")
    parser.add_argument("--debug", action="store_true", help="Process only 5 records")
    parser.add_argument(
        "--allow_random_fallback",
        action="store_true",
        help="Return random embeddings if model encoding fails. Not recommended.",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("BGE MULTILINGUAL GEMMA2 EMBEDDING EXTRACTION")
    logger.info("=" * 60)

    extractor = BGEMultilingualGemmaEmbeddingExtractor(args)
    extractor.process_dataset()


if __name__ == "__main__":
    main()
