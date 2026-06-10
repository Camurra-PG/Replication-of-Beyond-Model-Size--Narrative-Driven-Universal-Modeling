from __future__ import annotations
import json
import logging
from pydoc import text
import torch
import textwrap
import os
import re
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

# Configure logging
torch.backends.cudnn.benchmark = True
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("PORTRAIT_LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# Constants
MODEL_NAME = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
#MODEL_NAME = "google/gemma-3-1b-it"
CTX_LIMIT = 3048
MAX_PROMPT_TOKENS = int(os.getenv("PORTRAIT_MAX_PROMPT_TOKENS", "1800"))  # Réduit de 512 pour éviter OOM
GEN_TOKENS = 192
BATCH_SIZE = int(os.getenv("PORTRAIT_BATCH", "180")) 
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("La variable d'environnement HF_TOKEN n'est pas définie.")

RETAILROCKET_BRIEF = textwrap.dedent("""
[CONTEXT – Retailrocket E-Commerce Recommendation Evaluation]

You receive a structured Retailrocket e-commerce user profile derived from observed
behavioural events.

The downstream evaluation uses these user representations for recommendation-related
prediction tasks such as:
- churn / inactivity risk
- category affinity / category propensity
- SKU affinity / SKU propensity
- new-SKU or exploration tendency
- general recommendation relevance from observed behavioural patterns

Observed event types:
- page_visit
- add_to_cart
- product_buy

Available signals may include:
- user type and behavioural segments
- churn, purchase recency, activity gap, and return-risk indicators
- recent activity over the last 14 days
- temporal habits such as time of day, weekday/weekend behaviour, session rhythm
- browsing, cart, and purchase funnel behaviour
- SKU_PROPENSITY signals for personal product affinity
- CAT_PROPENSITY signals for personal category affinity
- GLOBAL_TOP_SKU / GLOBAL_TOP_CAT overlap with globally popular items/categories
- availability interactions such as IN_STOCK or OUT_OF_STOCK exposure
- category/product co-occurrence and sparse graph-centrality indicators
- RAW_SEQUENCE containing chronological Retailrocket events

Important dataset limitations:
- There is no validated price data in this Retailrocket pipeline.
- There is no validated search-query behaviour in this Retailrocket pipeline.
- Do not infer price sensitivity, discount responsiveness, search intent, demographics,
  brands, income, gender, age, or causal explanations.
- Do not invent SKU or category identifiers.
- Preserve relevant SKU_... and CAT_... identifiers exactly as provided.
- If the user has only page_visit events, describe them as browsing-only.
- If the user has no purchases or carts, explicitly say there is no observed conversion signal.
- Never answer that the behavioural signal is too sparse if at least one event exists.
- Even for sparse browsing-only users, produce useful behavioural bullets from recency,
  category/SKU exposure, availability, popularity overlap, and churn/return indicators.

Overall goal:
Produce a professional, concise behavioural portrait for recommender-system modelling.
The portrait should expose signals useful for the evaluation tasks, not marketing advice.

### OUTPUT FORMAT
- Do not copy raw feature lines verbatim.
- Do not output section names such as [CHURN], [SEQ], [SKU], [CAT], or ## CHURN_PROPENSITY ##.
- Do not output raw tags such as [OUT_OF_STOCK], [IN_STOCK], [REJECTED], [SUSPENDED], [INVALIDATED], or [END].
- Do not output feature names alone, such as CHURN_RISK:HIGH or PURCHASE_RECENCY:106d.
- Convert structured signals into natural behavioural interpretation.
- Output 4 to 8 bullet points.
- Each bullet must be one line.
- Each line must start exactly with "- "
- Plain English only.
- No introduction.
- No "Okay".
- No "Here is".
- No markdown headings.
- No code.
- No explanations outside the bullets.
- End with exactly: — FIN —

### GOOD EXAMPLE
- Buyer profile with strong historical conversion behaviour but currently high inactivity risk due to 106 days since last purchase and no activity in the last 14 days.
- Behaviour shows broad multi-category exploration, with strong affinity toward CAT_1051, CAT_959, and CAT_808.
- SKU affinity is concentrated around SKU_119736, SKU_198209, and SKU_37254, while browsing remains diverse across many products.
- Funnel behaviour includes many page visits, meaningful add-to-cart activity, and a high cart-to-purchase conversion ratio.
- Interactions are evening-dominant and often occur in deep sessions, suggesting intensive browsing when active.
- Availability signals show mostly in-stock interactions, with some out-of-stock exposure during browsing.
- Global-popularity overlap is stronger for categories than individual SKUs, so category-level recommendation signals may be more reliable.
— FIN —

### BAD EXAMPLE
Okay, here is a breakdown of the user behaviour:
The user seems interesting and might like some products.

### REMINDER
You are an expert behavioural analyst.
Write only the bullet list.
Do not include introductions, summaries, headings, or code fences.
Terminate with — FIN —.
""")

def _device_list() -> list[str]:
    """Liste des devices disponibles"""
    import torch
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


# PortraitGenerator class reste identique
class PortraitGenerator:
    """Utilise la chat-template officielle Gemma-3."""
    def __init__(self, device):
        import torch, logging, os
        from unsloth import FastModel

        self.device = device
        logging.getLogger(__name__).info("Loading %s on %s …", MODEL_NAME, device)

        self.model, tokenizer = FastModel.from_pretrained(
            MODEL_NAME,
            max_seq_length=CTX_LIMIT,
            load_in_4bit=True,          # ← off
            #torch_dtype=torch.float16,    # ← on
            device_map={"": device},
        )
        self.model.eval()
        # self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)  # disabled for stable multi-GPU inference
        self.tpl = get_chat_template(tokenizer, "gemma-3")
        #self.base_tok = self.tpl.tokenizer
        self.base_tok = getattr(self.tpl, "tokenizer", self.tpl)

        self.system_header = (
    "You are an expert behavioural analyst for recommender-system user modelling.\n"
    "Return ONLY plain bullet points.\n"
    "Every output line except the final terminator must start with '- '.\n"
    "Do NOT write introductions such as 'Okay', 'Here is', or 'Below is'.\n"
    "Do NOT generate code, markdown headings, tables, explanations, or code fences.\n"
    "Terminate with exactly '— FIN —'.\n"
)
    def _strip_rich_text(
    self,
    rt: str,
    keep_raw: bool = False,
    last_n: int = 30,
) -> str:
        """
        Prepare a compact Retailrocket profile for portrait generation.
    
        Raw event sequences and recent-history event listings are removed because
        the portrait should be based on the compact behavioural summaries,
        propensities, and popularity signals.
        """
        rt = re.sub(
            r"##\s*PORTRAIT\s*##.*?(?=##|$|\[END\])",
            "",
            rt,
            flags=re.DOTALL,
        )
    
        if not keep_raw:
            rt = re.sub(
                r"##\s*RAW_SEQUENCE\s*##.*?(?=##|$|\[END\])",
                "",
                rt,
                flags=re.DOTALL,
            )
    
        rt = rt.replace("[END]", "").strip()
    
        if len(rt) > 8000:
            rt = rt[:8000] + "\n[TRUNCATED FOR MEMORY]"
    
        return rt

    def _encode_batch(self, items):
        """Encode un batch de conversations"""
        conv_strings, cids = [], []
        for cid, rich in items:
            profile_txt = self._strip_rich_text(rich, keep_raw=False)
    
            token_ids = self.base_tok.encode(profile_txt, add_special_tokens=False)[:MAX_PROMPT_TOKENS]
            profile_txt = self.base_tok.decode(token_ids, skip_special_tokens=True)
    
            messages = [
                {
                    "role": "system",
                    "content": (
                        self.system_header.strip() + "\n\n" + RETAILROCKET_BRIEF.strip()
                    ),
                },
                {
                    "role": "user",
                    "content": f"[CLIENT_ID={cid}]\n{profile_txt}",
                },
            ]
    
            prompt_str = self.base_tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            conv_strings.append(prompt_str)
            cids.append(cid)
    
        batch = self.base_tok(
            conv_strings,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CTX_LIMIT - GEN_TOKENS,
        ).to(self.device)
    
        return batch, cids

    def _clean_portrait(self, raw_text: str) -> str:
        """Nettoie le portrait généré"""
        text = re.sub(r'```[\s\S]*?```', '', raw_text)
        text = text.split("— FIN —", 1)[0]
        
        bad_starts = (
    "- okay",
    "- here is",
    "- here's",
    "- below is",
    "- certainly",
    "- sure",
)

        forbidden_contains = (
            "[CHURN]",
            "[RECENT_HISTORY]",
            "[TIME]",
            "[SEQ]",
            "[AVAIL]",
            "[SOCIAL]",
            "[TOP]",
            "[SKU]",
            "[CAT]",
            "[STATS]",
            "[MISC]",
            "[END]",
            "## ",
            "[SUSPENDED]",
            "[REJECTED]",
            "[CANCELLED]",
            "[INVALIDATED]",
            "[LOST]",
            "[UNAVAILABLE]",
        )

        raw_feature_prefixes = (
            "- CHURN_RISK:",
            "- PURCHASE_RECENCY:",
            "- AVG_PURCHASE_INTERVAL:",
            "- POST_PURCHASE:",
            "- POST_PURCHASE_EVENTS:",
            "- SKU_PROPENSITY",
            "- CAT_PROPENSITY",
            "- GLOBAL_TOP_",
            "- BURST:",
            "- H_cat:",
            "- TOD_VAR:",
            "- CENT_",
            "- REC_RANK:",
        )

        bullets = []
        for ln in text.splitlines():
            line = ln.strip()
            if not line.startswith("- "):
                continue
            
            low = line.lower()
            if low.startswith(bad_starts):
                continue
            
            if any(x in line for x in forbidden_contains):
                continue
            
            if line.startswith(raw_feature_prefixes):
                continue
            
            line = line.replace("**", "").strip()

            # Keep only meaningful natural-language bullets.
            if len(line) < 25:
                continue
            
            bullets.append(line)
        
        if not bullets:
            cleaned = re.sub(r"\s+", " ", text).strip()
            if cleaned and cleaned != "— FIN —":
                bullets = [f"- {cleaned[:240]}"]
            else:
                bullets = [
                    "- Sparse browsing-only profile with limited explicit conversion signal.",
                    "- Behaviour is mainly represented through observed page visits, category/SKU exposure, recency, and availability interactions.",
                    "- No reliable purchase, price, demographic, or search-intent conclusions should be inferred."
                ]
        
        return "\n".join(bullets) + "\n— FIN —"

    @torch.inference_mode()
    def generate_batch(self, items):
        batch, cids = self._encode_batch(items)
    
        eos_id = self.base_tok.eos_token_id or self.base_tok.encode("— FIN —", add_special_tokens=False)[0]
    
        generated = self.model.generate(
            **batch,
            max_new_tokens=GEN_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=eos_id,
        )
    
        prompt_lens = batch["attention_mask"].sum(dim=1)
    
        portraits = {}
        for idx, cid in enumerate(cids):
            gen_part = generated[idx, int(prompt_lens[idx]):]
            text = self.base_tok.decode(gen_part, skip_special_tokens=True)
            portraits[cid] = self._clean_portrait(text)
    
        return portraits


# ========== NOUVEAU : Worker fonction pour multiprocessing ==========
def _gpu_worker(args):
    """Worker qui s'exécute sur un GPU spécifique"""
    device, items = args
    
    # Set CUDA device pour ce processus
    if 'cuda' in device:
        gpu_id = device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        actual_device = 'cuda:0'  # Dans le processus, c'est toujours cuda:0
    else:
        actual_device = device
    
    logger.info(f"GPU Worker starting on {device} (actual: {actual_device}) for {len(items)} items")
    
    try:
        gen = PortraitGenerator(actual_device)
        results = gen.generate_batch(items)
        
        # Cleanup
        del gen.model
        del gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info(f"GPU Worker on {device} completed {len(results)} portraits")
        return results
        
    except Exception as e:
        logger.error(f"GPU Worker on {device} failed: {e}", exc_info=True)
        return {cid: f"Error on {device}: {str(e)}\n— FIN —" for cid, _ in items}


# ========== NOUVELLE version parallèle de generate_portraits ==========
def generate_portraits(rich_texts: dict[int, str],
                      batch_size: int = BATCH_SIZE) -> dict[int, str]:
    """Version VRAIMENT parallèle qui utilise tous les GPUs simultanément"""
    
    items = list(rich_texts.items())
    devices = _device_list()
    
    if not devices:
        logger.error("No devices found (CPU/GPU). Exiting.")
        return {}
    if not items:
        logger.info("No items to process.")
        return {}
    
    logger.info(f"Starting parallel portrait generation: {len(items)} items on {len(devices)} devices")
    
    # Répartir les items entre les GPUs
    chunks_per_gpu = []
    items_per_gpu = len(items) // len(devices)
    remainder = len(items) % len(devices)
    
    start_idx = 0
    for i, device in enumerate(devices):
        # Distribuer équitablement + reste
        chunk_size = items_per_gpu + (1 if i < remainder else 0)
        end_idx = start_idx + chunk_size
        
        if chunk_size > 0:
            # Diviser en batches pour ce GPU
            gpu_chunks = []
            for j in range(start_idx, end_idx, batch_size):
                batch = items[j:min(j + batch_size, end_idx)]
                if batch:
                    gpu_chunks.append(batch)
            
            if gpu_chunks:
                chunks_per_gpu.append((device, gpu_chunks))
                logger.info(f"GPU {device}: {chunk_size} items in {len(gpu_chunks)} batches")
        
        start_idx = end_idx
    
    # Lancer les processus en parallèle
    portraits = {}
    
    with ProcessPoolExecutor(max_workers=len(devices)) as executor:
        # Préparer tous les jobs
        future_to_device = {}
        
        for device, chunks in chunks_per_gpu:
            # Concatener tous les chunks pour ce GPU
            all_items_for_gpu = []
            for chunk in chunks:
                all_items_for_gpu.extend(chunk)
            
            future = executor.submit(_gpu_worker, (device, all_items_for_gpu))
            future_to_device[future] = device
        
        # Collecter les résultats
        for future in as_completed(future_to_device):
            device = future_to_device[future]
            try:
                gpu_results = future.result()
                portraits.update(gpu_results)
                logger.info(f"Collected {len(gpu_results)} portraits from {device}")
            except Exception as e:
                logger.error(f"Failed to get results from {device}: {e}")
    
    logger.info(f"Portrait generation completed: {len(portraits)} portraits generated")
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Final CUDA cache cleared")
    
    return portraits