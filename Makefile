# Makefile for RetailRocket NDUM / RecSys 2025 project

.PHONY: all data features finetune extract ensemble clean \
        train_gemma1_only extract_gemma1_only \
        qwen_only stella_only bge_only show_config

DEBUG_FLAG :=
ifeq ($(DEBUG),true)
	DEBUG_FLAG := --debug
endif

PYTHON := python3

# RetailRocket data directory used by preprocessing_gemma1.py / preprocessing_gemma12.py
DATA_DIR := retailrocket_data

MODELS_DIR := models
FEATURES_DIR := output_features
EMBEDDINGS_DIR := embeddings

# These must match the hardcoded OUTPUT_DIR values inside preprocessing_gemma1.py and preprocessing_gemma12.py
GEMMA1_FEATURE_DIR := $(FEATURES_DIR)/retailrocket_gemma1b
GEMMA12_FEATURE_DIR := $(FEATURES_DIR)/retailrocket_gemma12b

GEMMA1_MODEL_ID := unsloth/gemma-3-1b-it-unsloth-bnb-4bit

GEMMA1_MODEL_DIR := $(MODELS_DIR)/contrastive_1M
GEMMA1_MAX_STEPS := 1100
CHECKPOINT_DIR := $(GEMMA1_MODEL_DIR)/checkpoint-$(GEMMA1_MAX_STEPS)

# Keep this as embeddings/gemma1b if ensemble.py should work unchanged
GEMMA1_EMBEDDINGS_DIR := $(EMBEDDINGS_DIR)/gemma1b

all: ensemble

show_config:
	@echo "DATA_DIR              = $(DATA_DIR)"
	@echo "FEATURES_DIR          = $(FEATURES_DIR)"
	@echo "GEMMA1_FEATURE_DIR    = $(GEMMA1_FEATURE_DIR)"
	@echo "GEMMA12_FEATURE_DIR   = $(GEMMA12_FEATURE_DIR)"
	@echo "MODELS_DIR            = $(MODELS_DIR)"
	@echo "GEMMA1_MODEL_DIR      = $(GEMMA1_MODEL_DIR)"
	@echo "GEMMA1_MAX_STEPS      = $(GEMMA1_MAX_STEPS)"
	@echo "CHECKPOINT_DIR        = $(CHECKPOINT_DIR)"
	@echo "EMBEDDINGS_DIR        = $(EMBEDDINGS_DIR)"
	@echo "GEMMA1_EMBEDDINGS_DIR = $(GEMMA1_EMBEDDINGS_DIR)"
	@echo "DEBUG_FLAG            = $(DEBUG_FLAG)"

# Step 1: Download RetailRocket data
data:
	@echo "--- 1. Downloading RetailRocket data ---"
	@bash src/download_data.sh $(DATA_DIR)

# Step 2: Feature Generation
features: data
	@echo "--- 2. Generating RetailRocket Features ---"
	@mkdir -p $(GEMMA1_FEATURE_DIR)
	@mkdir -p $(GEMMA12_FEATURE_DIR)
	$(PYTHON) src/preprocessing_gemma1.py $(DEBUG_FLAG)
	$(PYTHON) src/preprocessing_gemma12.py $(DEBUG_FLAG)

# Step 3: Fine-tuning Gemma1B
finetune: train_gemma1_only

train_gemma1_only:
	@echo "--- Training Gemma1B contrastive model ---"
	@echo "Feature dir: $(GEMMA1_FEATURE_DIR)"
	@echo "Output dir : $(GEMMA1_MODEL_DIR)"
	@echo "Max steps  : $(GEMMA1_MAX_STEPS)"
	@mkdir -p $(GEMMA1_MODEL_DIR)
	@DATASET=$$(ls $(GEMMA1_FEATURE_DIR)/complete_dataset_*.jsonl.zst | head -n 1); \
	echo "Using dataset: $$DATASET"; \
	accelerate launch src/train_gemma1.py \
		--model_id $(GEMMA1_MODEL_ID) \
		--dataset_path "$$DATASET" \
		--output_dir $(GEMMA1_MODEL_DIR) \
		--load_in_4bit \
		--lora_r 16 \
		--lora_alpha 32 \
		--projection_dim 2048 \
		--temperature 0.07 \
		--batch_size 24 \
		--gradient_accumulation 4 \
		--max_steps $(GEMMA1_MAX_STEPS) \
		--max_length 2048 \
		--learning_rate 2e-5 \
		--seed 42 \
		--logging_steps 25 \
		--save_steps 100 \
		--save_total_limit 5 \
		--warmup_steps 500 \
		--weight_decay 0.01

# Step 4: Embedding Extraction
extract: qwen_only stella_only extract_gemma1_only

extract_gemma1_only:
	@echo "--- Extracting Gemma1B embeddings ---"
	@echo "Checkpoint : $(CHECKPOINT_DIR)"
	@echo "Output dir : $(GEMMA1_EMBEDDINGS_DIR)"
	@mkdir -p $(GEMMA1_EMBEDDINGS_DIR)
	@DATASET=$$(ls $(GEMMA1_FEATURE_DIR)/complete_dataset_*.jsonl.zst | head -n 1); \
	echo "Using dataset: $$DATASET"; \
	accelerate launch src/extract_embeddings_gemma1.py \
		--model_id $(GEMMA1_MODEL_ID) \
		--projection_dim 2048 \
		--temperature 0.07 \
		--checkpoint_dir $(CHECKPOINT_DIR) \
		--dataset_path "$$DATASET" \
		--output_dir $(GEMMA1_EMBEDDINGS_DIR) \
		--max_length 2048 \
		--batch_size 32 \
		--load_in_4bit \
		--create_submission \
		$(DEBUG_FLAG)

qwen_only:
	@echo "--- Extracting Qwen embeddings ---"
	@mkdir -p $(EMBEDDINGS_DIR)/qwen3-8b
	@TEXTS=$$(ls $(GEMMA1_FEATURE_DIR)/complete_texts_*.jsonl.zst | head -n 1); \
	echo "Using texts: $$TEXTS"; \
	$(PYTHON) src/extract_embeddings_qwen.py \
		--dataset_path "$$TEXTS" \
		$(DEBUG_FLAG)

stella_only:
	@echo "--- Extracting Stella embeddings ---"
	@mkdir -p $(EMBEDDINGS_DIR)/stella
	@TEXTS=$$(ls $(GEMMA1_FEATURE_DIR)/complete_texts_*.jsonl.zst | head -n 1); \
	echo "Using texts: $$TEXTS"; \
	$(PYTHON) src/extract_embeddings_stella.py \
		--dataset_path "$$TEXTS" \
		$(DEBUG_FLAG)

bge_only:
	@echo "--- Extracting BGE Large EN embeddings ---"
	@mkdir -p $(EMBEDDINGS_DIR)/bge-large-en-v1.5
	@TEXTS=$$(ls $(GEMMA1_FEATURE_DIR)/complete_texts_*.jsonl.zst | head -n 1); \
	echo "Using texts: $$TEXTS"; \
	$(PYTHON) src/extract_embeddings_qwen.py \
		--model_name BAAI/bge-large-en-v1.5 \
		--dataset_path "$$TEXTS" \
		--batch_size 256 \
		--max_length 512 \
		--embedding_dim 2048 \
		$(DEBUG_FLAG)

# Step 5: Final Ensemble
ensemble: extract
	@echo "--- Creating Final Ensemble ---"
	@mkdir -p $(EMBEDDINGS_DIR)/ensemble
	$(PYTHON) src/ensemble.py

clean:
	@echo "--- Cleaning generated directories ---"
	rm -rf $(FEATURES_DIR) $(EMBEDDINGS_DIR)
	rm -rf unsloth_compiled_cache/ __pycache__/ src/__pycache__/ src/ubm/__pycache__/
	@echo "✅ Clean complete."