# ubm/text_representation_v3.py

from __future__ import annotations
# import unsloth
import os, multiprocessing
# 1) On détecte automatiquement  le nombre de vCPU (sur a2-highgpu-1g → 12)
n_threads = multiprocessing.cpu_count()
print(f"n_threads:{n_threads}")
# For BLAS / OpenMP back-ends
os.environ["OPENBLAS_NUM_THREADS"]  = "4"
os.environ["MKL_NUM_THREADS"]       = "4"
os.environ["NUMEXPR_MAX_THREADS"]   = "4"
os.environ["OMP_NUM_THREADS"]       = "4"
os.environ["POLARS_MAX_THREADS"]    = "4"
#import unsloth
# NetworKit – needs an explicit call
import networkit as nk
nk.setNumberOfThreads(4)
import pyarrow.parquet as pq  
import math             #  ← NEW
import statistics        #  ← NEW
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re
import time
import scipy.sparse as sp
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set, Optional, Union, Any
import logging
from scipy.stats import entropy
from collections import Counter, defaultdict
import os
from tqdm import tqdm
from pathlib import Path
import gc
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import polars as pl
import json, pickle, gc, os, logging, networkx as nx
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm
from itertools import combinations          # <-- IMPORT supplémentaire en haut de fichier
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
from math import log2
import networkit as nk
# from networkit import embedding as nk_embed      # NetworKit’s fast Node2Vec
#from .portrait_generator import PortraitGenerator, generate_portraits
print("Polars pool size:", pl.threadpool_size())
from math import isfinite
pl.enable_string_cache()
# --- Setup Logging ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ---------------------------------------------------------------------------
# >>> GLOBAL CONSTANTS (already present in the original file – duplicated   <<<
# >>> here for self‑containment; keep them in sync with the main module)   <<<
# ---------------------------------------------------------------------------
MAX_RICH_TOKENS          : int = 4096  # set to 2048 if you want shorter texts
TOP_FEATURES_PER_SECTION : int = 10    # soft‑cap per section before trimming
IMPLICIT_WEIGHT_REPEAT   : int = 2     # repeat high‑weight tokens N times
TOP_RETAILROCKET_SKUS: int = 100
TOP_RETAILROCKET_CATEGORIES: int = 100
SECTIONS_ORDER: List[str] = [
    "OVERVIEW",
    "CHURN_PROPENSITY",
    "RECENT_HISTORY_14D",
    "TEMPORAL",
    "SEQUENCE",
    "PRICE",
    "AVAILABILITY",
    "SOCIAL",
    "GLOBAL_POPULARITY",
    "SKU_PROPENSITY",
    "CAT_PROPENSITY",
    "PROP_SUBSET_STATS",
    "CUSTOM",
]

SECTION_MARKERS = {
    "OVERVIEW": "[PROFILE]",
    "CHURN_PROPENSITY": "[CHURN]",
    "RECENT_HISTORY_14D": "[RECENT_HISTORY]",
    "TEMPORAL": "[TIME]",
    "SEQUENCE": "[SEQ]",
    "PRICE": "[PRICE]",
    "AVAILABILITY": "[AVAIL]",
    "SOCIAL": "[SOCIAL]",
    "GLOBAL_POPULARITY": "[TOP]",
    "SKU_PROPENSITY": "[SKU]",
    "CAT_PROPENSITY": "[CAT]",
    "PROP_SUBSET_STATS": "[STATS]",
    "CUSTOM": "[MISC]",
}

# ---------------------------------------------------------------------------
#                          UTILITY HELPERS                                   #
# ---------------------------------------------------------------------------
# --- Base Class ---
class FeatureExtractorBase:
    """Base class for feature extractors"""
    def __init__(self, parent):
        self.parent = parent # Reference to parent AdvancedUBMGenerator instance
        self.logger = logging.getLogger(self.__class__.__name__)
        # Access shared data via self.parent, e.g., self.parent.sku_properties_dict
        if self.parent.debug_mode: self.logger.setLevel(logging.DEBUG)

    def extract_features(self, client_id: int, events: pl.DataFrame, now: datetime) -> List[str]:
        """Extract features for a client - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement extract_features")


def compute_sparse_pagerank(src: np.ndarray,
                            dst: np.ndarray,
                            weights: np.ndarray,
                            alpha: float = 0.85,
                            tol: float = 1e-6,
                            max_iter: int = 1000) -> dict[int, float]:
    """
    Parallel PageRank using NetworKit.  ~10-20× faster than NetworkX
    on million-edge graphs.
    """
    g, id2orig = _nk_graph_from_edges(src, dst, weights, directed=True)
    pr = nk.centrality.PageRank(
        g, damp=alpha, tol=tol, maxIterations=max_iter, normalized=False
    )
    pr.run()
    scores = pr.scores()          # list[float] aligned with 0…n-1 ids
    return {int(id2orig[i]): s for i, s in enumerate(scores)}

# ------------------------------------------------------------------
# NetworKit helpers
# ------------------------------------------------------------------
# --- BEGIN PATCH: helpers -----------------------------------------------------
import numpy as np

def _nk_graph_from_edges(src: np.ndarray,
                         dst: np.ndarray,
                         w:   np.ndarray,
                         directed: bool = True) -> tuple[nk.Graph, np.ndarray]:
    """
    Build a NetworKit graph + id→label array in one pass.
    """
    nodes, inverse = np.unique(np.concatenate([src, dst]), return_inverse=True)
    g = nk.Graph(len(nodes), weighted=True, directed=directed)
    half = len(src)
    for u, v, weight in zip(inverse[:half], inverse[half:], w):
        if u == v:
            continue
        eid = g.addEdge(u, v, w=float(weight))
        if eid == -1:                          # multi-edge → accumulate
            eid = g.edgeId(u, v)
            g.setWeight(u, v, g.weight(u, v) + float(weight))
    return g, nodes                            # nodes[i] = original id/label


def compute_sparse_pagerank(src: np.ndarray,
                            dst: np.ndarray,
                            weights: np.ndarray,
                            alpha: float = 0.85,
                            tol: float = 1e-6,
                            max_iter: int = 1_000) -> dict[int, float]:
    g, id2orig = _nk_graph_from_edges(src, dst, weights, directed=True)

    pr = nk.centrality.PageRank(g, damp=alpha, tol=tol, normalized=False)

    # older NetworKit (< 10) does *not* accept maxIterations as __init__ kw
    if hasattr(pr, "setMaxIterations"):
        pr.setMaxIterations(max_iter)

    pr.run()
    scores = pr.scores()                       # list[float] aligned with 0…n-1 ids
    return {int(id2orig[i]): s for i, s in enumerate(scores)}



def _shannon_entropy(counter: Counter) -> float:
    """Shannon entropy in bits from a Counter of counts."""
    n = sum(counter.values())
    if n == 0:
        return 0.0
    return -sum((c / n) * log2(c / n) for c in counter.values())

def _approx_token_len(text: str) -> int:
    """Very cheap proxy for token count (≈ whitespace split)."""
    return len(text.split())


def _truncate_to_max_tokens(lines: List[str], limit: int) -> List[str]:
    """Greedy keep‑from‑start strategy (safer for ordered / weighted chunks)."""
    kept: List[str] = []
    for ln in lines:
        if _approx_token_len("\n".join(kept + [ln])) > limit:
            break
        kept.append(ln)
    return kept


def _top_k_features(features: List[str], k: int) -> Tuple[List[str], List[str]]:
    """Return `(top_k, overflow)` lists – *overflow* can be shuffled/dropped."""
    if len(features) <= k:
        return features, []
    return features[:k], features[k:]


def _repeat_for_weight(lines: List[str], repeat: int) -> List[str]:
    """Naïve implicit weight: duplicate *every* line *repeat* times."""
    if repeat <= 1:
        return lines
    out: List[str] = []
    for ln in lines:
        out.extend([ln] * repeat)
    return out
def bucketize_days(delta_days: int) -> str:
    if delta_days <= 3:
        return "R_0-3d"
    if delta_days <= 7:
        return "R_3-7d"
    if delta_days <= 30:
        return "R_7-30d"
    return "R_30+d"

POP_QUANT_EDGES: list = []   # global mutable (sera rempli une fois)

def pop_bin(score: float) -> str:
    """Retourne la quantile de popularité (Q0 à Q4). Q0 = inconnu / score manquant."""
    if score is None or math.isnan(score):
        return "Q0"
    # Les bords sont calculés et stockés dans AdvancedUBMGenerator._compute_product_popularities
    for i, edge in enumerate(POP_QUANT_EDGES, start=1):
        if score <= edge:
            return f"Q{i}"
    return "Q4"

# ────────────────────────────────────────────────────────────────────────────
# Helper : transforme le dictionnaire section→liste en texte final
# ────────────────────────────────────────────────────────────────────────────
def _build_rich_text(
    section_map: dict[str, list[str]],
    max_tokens: int = 3500,
    implicit_repeat: int = 2,
    top_per_section: int = 10,
    shuffle_seed: int | None = None,
    use_markers: bool = True,  # NOUVEAU paramètre
) -> str:
    """
    Version optimisée avec markers et déduplication améliorée
    """
    import random
    import itertools
    from collections import OrderedDict

    rnd = random.Random(shuffle_seed)
    
    # Déduplication globale des features
    seen_features = set()
    deduped_sections = {}
    
    for section, items in section_map.items():
        deduped_items = []
        for item in items:
            # Normaliser pour la déduplication
            normalized = item.strip().lower()
            if normalized not in seen_features:
                seen_features.add(normalized)
                deduped_items.append(item)
        deduped_sections[section] = deduped_items

    def _truncate(tokens: list[str], limit: int) -> list[str]:
        total = 0
        out = []
        for tok in tokens:
            total += len(tok.split())
            if total > limit:
                break
            out.append(tok)
        return out

    lines: list[str] = []

    for section in SECTIONS_ORDER:
        items = deduped_sections.get(section, [])
        if not items:
            continue

        # 1. Limiter au top K
        items = items[:top_per_section]

        # 2. Répétition implicite pour les items importants
        repeated = []
        for item in items:
            # Les features de churn/propensity sont toujours répétées
            if "CHURN_" in item or "PROPENSITY" in item or "**" in item:
                repeated.extend([item] * implicit_repeat)
            else:
                repeated.append(item)

        ordered_sections = {
            "GLOBAL_POPULARITY",
            "SKU_PROPENSITY",
            "CAT_PROPENSITY",
        }

        if len(repeated) > 3 and section not in ordered_sections:
            first_items = repeated[:2]
            rest_items = repeated[2:]
            rnd.shuffle(rest_items)
            repeated = first_items + rest_items

        # 4. Ajouter le marqueur de section et le contenu
        if use_markers and section in SECTION_MARKERS:
            lines.append(f"{SECTION_MARKERS[section]}")
        lines.append(f"## {section} ##")
        lines.extend(repeated)

    # 5. Coupe globale au nombre de tokens demandé
    lines = _truncate(lines, max_tokens)

    # 6. Ajouter un marqueur de fin
    if use_markers:
        lines.append("[END]")

    return "\n".join(lines)






class TemporalFeatureExtractor(FeatureExtractorBase):
    """Extract temporal patterns from user behavior, enhanced with recency and inactivity."""

    def extract_features(self, client_id: int, events: pl.DataFrame, now: datetime) -> List[str]:
        if events.height == 0: return ["No activity data for temporal analysis"]
        features = []
        try:
            # Use a single timestamp access for efficiency
            timestamps_sorted = events.sort("timestamp")['timestamp']
            last_ts = timestamps_sorted.max()
            first_ts = timestamps_sorted.min()

            # Appeler les sous-méthodes
            self._extract_daily_patterns(events, features)
            self._extract_weekly_patterns(events, features)
            self._extract_session_patterns(events, features, timestamps_sorted) # Passer les timestamps triés
            self._extract_recency_and_frequency(events, features, now, last_ts) # Enhanced recency
            self._extract_inactivity_gaps(features, timestamps_sorted, now, last_ts) # New inactivity analysis

        except Exception as e:
            self.logger.error(f"Err temporal client {client_id}: {e}", exc_info=self.parent.debug_mode)
            features.append("Err temporal")
        return features

    def _extract_daily_patterns(self, events: pl.DataFrame, features: List[str]) -> None:
            try:
                # ✅ FIX: Utiliser pl.count() au lieu de pl.col().count()
                hour_counts = (
                    events
                    .with_columns(pl.col('timestamp').dt.hour().alias('hour_of_day'))
                    .group_by('hour_of_day')
                    .agg(pl.count().alias("count"))  # ← Changé ici
                    .sort('hour_of_day')
                )
                
                if hour_counts.height > 0:
                    max_count_row = hour_counts.sort("count", descending=True).row(0, named=True)
                    if max_count_row:
                        max_count = max_count_row['count']
                        peak_hours = hour_counts.filter(pl.col('count') >= 0.8 * max_count)['hour_of_day'].drop_nulls().to_list()
                        if peak_hours: 
                            features.append(f"Peak hours: {', '.join([f'{h}:00' for h in sorted(peak_hours)])}")
                        
                        # Segments de la journée
                        morning = [h for h in peak_hours if 5 <= h < 12]
                        afternoon = [h for h in peak_hours if 12 <= h < 18]
                        evening = [h for h in peak_hours if h >= 18 or h < 5]
                        
                        time_segments = [(len(morning), "Morning"), (len(afternoon), "Afternoon"), (len(evening), "Evening")]
                        dominant = max([s for s in time_segments if s[0] > 0], key=lambda x: x[0], default=(0, None))
                        if dominant[1]: 
                            features.append(f"{dominant[1]}-dominant")
            except Exception as e:
                self.logger.debug(f"Err daily: {e}")
                features.append("Err daily patterns")

    # ── TemporalFeatureExtractor._extract_weekly_patterns ──
    def _extract_weekly_patterns(self, events: pl.DataFrame, features: List[str]) -> None:
        try:
            # ✅ FIX: Utiliser pl.count() et créer la colonne weekday d'abord
            day_counts = (
                events
                .with_columns(pl.col('timestamp').dt.weekday().alias('day_of_week'))
                .group_by('day_of_week')
                .agg(pl.count().alias('count'))  # ← Changé ici
                .sort('day_of_week')
            )

            day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu',
                         4: 'Fri', 5: 'Sat', 6: 'Sun', 7: 'Sun'}

            if day_counts.height == 0:
                return

            max_count = day_counts['count'].max()
            
            if max_count is None:
                return

            # Récupérer les jours de pointe
            peak_days = (
                day_counts
                .filter(pl.col('count') >= 0.8 * max_count)['day_of_week']
                .to_list()
            )
            
            # Normaliser les jours (au cas où il y aurait des valeurs > 6)
            peak_days = [(d % 7) for d in peak_days]

            if peak_days:
                features.append(
                    "Active days: " + ", ".join(day_names[d] for d in sorted(peak_days))
                )

                weekday = any(d < 5 for d in peak_days)
                weekend = any(d >= 5 for d in peak_days)
                if weekday and not weekend:
                    features.append("Weekday-dominant")
                elif weekend and not weekday:
                    features.append("Weekend-dominant")

        except Exception as e:
            self.logger.debug(f"Err weekly: {e}")
            features.append("Err weekly patterns")

            
    # ── dans la même classe ──────────────────────────────────────────────────────
    def _split_sessions(self, timestamps_sorted: pl.Series, gap: int = 30) -> pl.Series:
        """
        Renvoie un id de session (0,1,2,…) pour chaque événement.
        Nouveau numéro si écart > gap minutes.
        """
        time_diff = timestamps_sorted.diff().dt.total_seconds() / 60
        is_new    = time_diff.is_null() | (time_diff > gap)
        return is_new.cum_sum()          



    
    def _extract_session_patterns(self,
                                  events: pl.DataFrame,
                                  features: List[str],
                                  timestamps_sorted: pl.Series) -> None:
        try:
            if events.height <= 1:
                return
            # 1. scinder en sessions
            sess_ids = self._split_sessions(timestamps_sorted)        # <-- NEW
            sess_df = pl.DataFrame({'sid': sess_ids, 'ts': timestamps_sorted})

            # 2. stats par session (durée & #events)
            sess_stats = (
                sess_df.group_by('sid')
                       .agg(pl.min('ts').alias('start'),
                            pl.max('ts').alias('end'),
                            pl.count().alias('cnt'))
                       .with_columns(
                           ((pl.col('end') - pl.col('start'))
                            .dt.total_seconds() / 60).alias('dur'))   # minutes
            )
            n_sessions = sess_stats.height         
            if n_sessions == 0:
                return
            # mix jour/nuit des débuts de sessions
            starts = sess_stats['start']
            night  = (starts.dt.hour() >= 18) | (starts.dt.hour() < 5)
            ratio  = night.sum() / n_sessions
            if ratio > 0.7:
                features.append("Mostly evening sessions")
            elif ratio < 0.3:
                features.append("Mostly daytime sessions")
            


            # 3. moyennes
            avg_dur = sess_stats['dur'].mean()        # durée moyenne en minutes
            avg_cnt = sess_stats['cnt'].mean()        # évènements / session

            features.append(f"Sessions: {n_sessions}")
            if avg_dur < 5:
                features.append(f"Typically very short sessions (~{avg_dur:.1f} m)")
            elif avg_dur > 60:
                features.append(f"Typically long sessions (~{avg_dur:.1f} m)")

            if avg_cnt < 3:
                features.append(f"Typically shallow sessions (~{avg_cnt:.1f} evt)")
            elif avg_cnt > 15:
                features.append(f"Typically deep sessions (~{avg_cnt:.1f} evt)")

        except Exception as e:
            self.logger.debug(f"Err session: {e}")
            features.append("Err session patterns")


    def _extract_recency_and_frequency(self, events: pl.DataFrame, features: List[str], now: datetime, last_ts: Optional[datetime]) -> None:
        """Enhanced recency and recent frequency calculations."""
        try:
            if events.height == 0 or last_ts is None:
                 features.append("Recency: No Activity")
                 return

            days_since_last = (now - last_ts).days
            features.append(f"Days Since Last Activity: {days_since_last}")
            if days_since_last <= 7: features.append("Activity Status: Very Recent")
            elif days_since_last <= 30: features.append("Activity Status: Moderately Recent")
            elif days_since_last <= 90: features.append("Activity Status: Lapsed")
            else: features.append("Activity Status: Very Lapsed/Inactive")

            # Recency of specific important actions
            last_purchase_ts = events.filter(pl.col('event_type') == pl.lit('product_buy', dtype=pl.Categorical))['timestamp'].max()
            last_cart_add_ts = events.filter(pl.col('event_type') == pl.lit('add_to_cart', dtype=pl.Categorical))['timestamp'].max()

            if last_purchase_ts: features.append(f"Days Since Last Purchase: {(now - last_purchase_ts).days}")
            else: features.append("Days Since Last Purchase: Never")
            if last_cart_add_ts: features.append(f"Days Since Last Cart Add: {(now - last_cart_add_ts).days}")
            else: features.append("Days Since Last Cart Add: Never")

            # Recent Frequency (last 30 days)
            cutoff_30d = now - timedelta(days=30)
            recent_events = events.filter(pl.col("timestamp") >= cutoff_30d)

            if recent_events.height > 0:
                event_count_30d   = recent_events.height

                # quick-win : l’utilisateur était absent >90 j et revient dans les 30 derniers jours
                if days_since_last > 90:
                    features.append("Reactivation after long dormancy")

                active_days_30d   = recent_events['timestamp'].dt.date().n_unique()
                first_ts_in_30d   = recent_events['timestamp'].min()
                observed_days     = max(1, (last_ts - first_ts_in_30d).days + 1)  # au moins 1 jour
                active_ratio      = active_days_30d / min(30, observed_days)

                features.append(
                    f"Activity (Last 30d): {event_count_30d} events over "
                    f"{active_days_30d} days (Active Ratio: {active_ratio:.1%})"
                )
            else:
                features.append("Activity (Last 30d): None")


        except Exception as e:
            self.logger.debug(f"Err recency/frequency: {e}")
            features.append("Err recency/frequency")
            
        # Fraîcheur globale
        fresh14 = events.filter(pl.col('timestamp') >= now - timedelta(days=14)).height / events.height
        if fresh14 > 0.5:
            features.append("Recent-heavy activity (<14d)")
        

    def _extract_inactivity_gaps(self, features: List[str], timestamps_sorted: pl.Series, now: datetime, last_ts: Optional[datetime]) -> None:
        """Analyze inactivity gaps between events."""
        try:
            if timestamps_sorted.len() <= 1 or last_ts is None: return # Need at least 2 events

            timestamps_list = timestamps_sorted.to_list()
            gaps_days = [(timestamps_list[i+1] - timestamps_list[i]).total_seconds() / (3600 * 24) for i in range(len(timestamps_list)-1)]

            if gaps_days:
                 max_gap = np.max(gaps_days)
                 features.append(f"Max Inactivity Gap: {max_gap:.1f} days")
                 gaps_gt_7d = sum(1 for g in gaps_days if g > 7)
                 gaps_gt_30d = sum(1 for g in gaps_days if g > 30)
                 if gaps_gt_30d > 0: features.append(f"Notable Gaps (>30d): {gaps_gt_30d}")
                 elif gaps_gt_7d > 0: features.append(f"Notable Gaps (>7d): {gaps_gt_7d}")

            # Check recent gap (since last event)
            days_since_last = (now - last_ts).days
            if days_since_last > 30:
                 features.append("Recent Status: Currently Inactive (>30 days)")
            elif days_since_last > 7:
                 features.append("Recent Status: Currently Lapsing (>7 days)")

        except Exception as e:
            self.logger.debug(f"Err inactivity gaps: {e}")
            features.append("Err inactivity gaps")

    def _count_sessions_from_timestamps(self, timestamps_sorted: pl.Series, session_gap_minutes: int = 30) -> int:
        """Helper to count sessions just from a sorted timestamp series."""
        if timestamps_sorted.len() <= 1: return timestamps_sorted.len()
        try:
            time_diffs_minutes = timestamps_sorted.diff().dt.total_seconds() / 60
            # Un diff() donne null pour le premier -> is_null() est vrai -> début de session
            # Ensuite, vérifier si diff > gap
            session_starts = time_diffs_minutes.is_null() | (time_diffs_minutes > session_gap_minutes)
            num_sessions = session_starts.sum()
            return int(num_sessions)
        except Exception as e:
            self.logger.error(f"Error counting sessions from timestamps: {e}")
            return 1 # Fallback


class SequenceFeatureExtractor(FeatureExtractorBase):
    """Extract sequential behavior patterns"""
    def extract_features(self, client_id: int, events: pl.DataFrame, now: datetime) -> List[str]:
        if events.height < 2:
            return ["Very limited activity"]
    
        features: List[str] = []
        try:
            # Polars → Python list (no Pandas round-trip)
            event_types = (
                events.sort("timestamp")
                      ["event_type"]
                      .to_list()
            )
    
            if event_types.count("product_buy") == 1:
                features.append("Single-purchase buyer")
    
            self._extract_event_sequences(event_types, features)
            self._extract_purchase_funnel(events, features)
    
            if 'category_id' in events.columns:
                self._extract_Browse_sequences(events, features)
    
        except Exception as e:
            self.logger.error(f"Error extracting sequence features for client {client_id}: {e}",
                              exc_info=self.parent.debug_mode)
            features.append("Error during sequence feature extraction.")
        return features


    def _extract_event_sequences(self, event_types: list, features: List[str]) -> None:
        try:
            if len(event_types) < 3: return
            trigrams = self._create_ngrams(event_types, 3)
            if trigrams:
                trigram_counts = Counter(trigrams); total_trigrams = len(trigrams)
                for trigram, count in trigram_counts.most_common(2):
                    if count >= 2:
                        sequence_str = " -> ".join(trigram); freq_pct = (count / total_trigrams) * 100
                        features.append(f"Common sequence: {sequence_str} ({count} times, {freq_pct:.1f}%)")
            fourgrams = self._create_ngrams(event_types, 4)
            if fourgrams:
                fourgram_counts = Counter(fourgrams); total_fourgrams = len(fourgrams)
                for fourgram, count in fourgram_counts.most_common(1):
                    if count >= 2:
                        sequence_str = " -> ".join(fourgram); freq_pct = (count / total_fourgrams) * 100
                        features.append(f"Common extended path: {sequence_str} ({count} times, {freq_pct:.1f}%)")
            # ---------- Ping‑pong A‑B‑A‑B detection ----------
            if len(event_types) >= 4:
                pp_count = 0
                for j in range(len(event_types) - 3):
                    a, b, c, d = event_types[j:j+4]
                    if a == c and b == d and a != b:
                        pp_count += 1
                if pp_count >= 2:
                    features.append(f"Ping‑pong navigation pattern ({pp_count} times)")             
        except Exception as e:
            self.logger.debug(f"Error extracting event sequences: {e}")
            features.append("Error extracting event sequences")

    def _extract_purchase_funnel(
        self,
        events: pl.DataFrame,
        features: List[str],
    ) -> None:
        """
        Summarize observed Retailrocket funnel events.

        Retailrocket contains product views, cart additions and transactions,
        but no search-query events. Ratios are descriptive event ratios and
        should not be interpreted as a fully observed conversion path.
        """
        try:
            tmp = (
                events.group_by("event_type")
                .agg(pl.len().alias("count"))
            )

            event_counts = {
                row["event_type"]: row["count"]
                for row in tmp.iter_rows(named=True)
            }

            views = event_counts.get("page_visit", 0)
            cart_adds = event_counts.get("add_to_cart", 0)
            purchases = event_counts.get("product_buy", 0)

            if views == 0 and cart_adds == 0 and purchases == 0:
                return

            funnel_stages = ["Observed funnel events:"]

            if views > 0:
                funnel_stages.append(f"  Product Views: {views}")

            if cart_adds > 0:
                if views > 0:
                    cart_view_ratio = (cart_adds / views) * 100
                    funnel_stages.append(
                        f"  Cart Adds: {cart_adds} "
                        f"({cart_view_ratio:.1f}% relative to views)"
                    )
                else:
                    funnel_stages.append(
                        f"  Cart Adds: {cart_adds} "
                        f"(no preceding view observed)"
                    )

            if purchases > 0:
                if cart_adds > 0 and purchases <= cart_adds:
                    purchase_cart_ratio = (purchases / cart_adds) * 100
                    funnel_stages.append(
                        f"  Purchases: {purchases} "
                        f"({purchase_cart_ratio:.1f}% relative to cart adds)"
                    )
                elif cart_adds > 0:
                    funnel_stages.append(
                        f"  Purchases: {purchases} "
                        f"(includes purchases without observed cart add)"
                    )
                else:
                    funnel_stages.append(
                        f"  Purchases: {purchases} "
                        f"(no preceding cart add observed)"
                    )

                if views > 0:
                    purchase_view_ratio = (purchases / views) * 100
                    funnel_stages.append(
                        f"  Purchase/View Event Ratio: "
                        f"{purchase_view_ratio:.2f}%"
                    )

            if len(funnel_stages) > 1:
                features.append("\n".join(funnel_stages))

        except Exception as exc:
            self.logger.debug(f"Error extracting purchase funnel: {exc}")
            features.append("Error extracting purchase funnel")

    def _extract_Browse_sequences(self, events: pl.DataFrame, features: List[str]) -> None:
        try:
            page_visits = (
                events.filter(
                    (pl.col('event_type') == pl.lit('page_visit', dtype=pl.Categorical)) &
                    pl.col('category_id').is_not_null()
                )
                .sort('timestamp')
            )
            if page_visits.height < 3:
                return
    
            # 1) add 30-minute session ids (vectorised)
            gaps = page_visits['timestamp'].diff().dt.total_seconds() / 60
            page_visits = page_visits.with_columns(
                ((gaps.is_null()) | (gaps > 30)).cum_sum().alias('sid')
            )
    
            # 2) iterate zero-copy over sessions
            from collections import Counter
            trigram_counter = Counter()
    
            for sess in page_visits.partition_by('sid', as_dict=False):
                cats = (sess['category_id']
                          .drop_nulls()
                          .unique()
                          .to_list())
                if len(cats) >= 3:
                    trigrams = self._create_ngrams(cats, 3)
                    trigram_counter.update(trigrams)
    
            if trigram_counter:
                top_tri, cnt = trigram_counter.most_common(1)[0]
                if cnt >= 2:
                    features.append(
                        f"Common category sequence: "
                        f"{' -> '.join([f'CAT_{c}' for c in top_tri])} ({cnt}x)"
                    )
        except Exception as e:
            self.logger.debug(f"Error extracting Browse sequences: {e}")
            features.append("Error extracting Browse sequences")


    def _create_ngrams(self, sequence: list, n: int) -> List[tuple]:
        if len(sequence) < n: return []
        return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]


class GraphFeatureExtractor(FeatureExtractorBase):
    """Extract graph-based behavioral features"""

    def extract_features(self, client_id: int, events: pl.DataFrame, now: datetime) -> List[str]:
        if events.height < 5: return []
        features = []
        try:
            if 'category_id' in events.columns:
                 self._extract_category_graph_features(client_id, events, features)
            else: features.append("Category graph skipped (no category_id).")
            if events.height >= 10 and 'sku' in events.columns:
                 self._extract_product_graph_features(client_id, events, features)
            else: features.append("Product graph skipped (few events or no sku).")
        except Exception as e: self.logger.error(f"Err graph client {client_id}: {e}"); features.append("Err graph")
        return features

    def _extract_category_graph_features(self, client_id: int,
                                         events: pl.DataFrame,
                                         features: list[str]) -> None:
        """
        Same logic as before but using NetworKit for the expensive bits.
        """
        try:
            page_visits = events.filter(
                (pl.col('event_type') == pl.lit('page_visit', dtype=pl.Categorical))
                & pl.col('category_id').is_not_null()
            ).sort('timestamp')

            if page_visits.height < 5:
                return

            cats = page_visits['category_id']
            uniq_mask = cats.diff().fill_null(1) != 0
            categories = cats.filter(uniq_mask).to_list()

            if len(categories) < 2:
                return

            # ------------------------------------------------------------------
            # 1) build directed weighted edge list
            src, dst = np.array(categories[:-1], dtype=int), np.array(categories[1:], dtype=int)
            weights  = np.ones_like(src, dtype=float)
            g, id2orig = _nk_graph_from_edges(src, dst, weights, directed=True)

            features.append(f"Category exploration: {g.numberOfNodes()} unique.")

            # ------------------------------------------------------------------
            # 2) PageRank
            pr = nk.centrality.PageRank(g, damp=0.85, tol=1e-4)
            pr.run()
            scores = pr.scores()
            if scores:
                top_idx = int(np.argmax(scores))
                top_cat = int(id2orig[top_idx])
                features.append(f"Dominant cat (PR): CAT_{top_cat} ({scores[top_idx]:.3f})")

            # ------------------------------------------------------------------
            # 3) Average clustering coefficient, where supported by NetworKit.
            # Some local NetworKit versions do not expose nk.clustering.
            try:
                if hasattr(nk, "clustering") and hasattr(
                    nk.clustering, "LocalClusteringCoefficient"
                ):
                    lu = nk.clustering.LocalClusteringCoefficient(
                        g.toUndirected(), weighted=True
                    )
                    lu.run()
                    avg_clust = sum(lu.scores()) / g.numberOfNodes()
                    features.append(f"Avg cat clustering: {avg_clust:.3f}")
            except Exception as exc:
                self.logger.debug(
                    f"Category clustering coefficient skipped: {exc}"
                )

            # ------------------------------------------------------------------
            # 4) Top transition (weight ≥ 2)

            # ------------------------------------------------------------------
            # 4) Top transition (weight ≥ 2)
            best_w, best_pair = 0, None
            for (u, v) in g.iterEdges():
                w = g.weight(u, v)
                if w > best_w:
                    best_w, best_pair = (u, v)
            if best_pair and best_w >= 2:
                u, v = best_pair
                features.append(f"Top cat transition: CAT_{int(id2orig[u])}->CAT_{int(id2orig[v])} ({int(best_w)}x)")

        except Exception as e:
            self.logger.debug(f"Err cat graph: {e}")
            features.append("Err cat graph")

            
    # ─────────────────────────────────────────────────────────────────────
    # Product-level co-interaction graph (NetworKit, fully vectorised)
    # ─────────────────────────────────────────────────────────────────────
    def _extract_product_graph_features(self,
                                        client_id: int,
                                        events: pl.DataFrame,
                                        features: list[str]) -> None:
        """
        Build a session-level SKU co-interaction graph, compute basic
        structure metrics and a central product using NetworKit.
        Much faster and safer than the previous NetworkX version.
        """
        try:
            # 0) keep only page / cart / buy events with a valid SKU
            rel_evt = (
                events.filter(
                    pl.col('event_type')
                      .is_in(['page_visit', 'add_to_cart', 'product_buy'])
                    & pl.col('sku').is_not_null()
                )
                .sort('timestamp')
            )

            if rel_evt.height < 3:
                return

            # 1) rebuild 30-minute sessions   (Polars → NumPy, no slow loops)
            sess_gap = 30        # minutes
            ts = rel_evt['timestamp']
            gaps_min = (
                ts.diff().dt.total_seconds()      # diff is None at row 0
                  .fill_null(sess_gap * 60 + 1)   # force new session on 1st row
                  / 60                            # seconds → minutes
            )
            sess_id = gaps_min.gt(sess_gap).cum_sum()  # fast cumulative ids
            rel_evt = rel_evt.with_columns(pl.Series('sid', sess_id))

            # 2) count SKU co-occurrences inside each session
            from itertools import combinations
            from collections import Counter
            pair_cnt = Counter()

            for _, sess in rel_evt.group_by('sid'):
                skus = sess['sku'].drop_nulls().unique().to_list()
                if len(skus) < 2:
                    continue
                for i, j in combinations(sorted(skus), 2):
                    pair_cnt[(int(i), int(j))] += 1

            if not pair_cnt:
                return

            pairs   = np.array(list(pair_cnt.keys()),   dtype=int)
            weights = np.array(list(pair_cnt.values()), dtype=float)

            # 3) build NetworKit graph (undirected, weighted)
            g, id2sku = _nk_graph_from_edges(pairs[:, 0], pairs[:, 1],
                                             weights, directed=False)

            n = g.numberOfNodes()
            e = g.numberOfEdges()
            features.append(f"Product exploration: {n} unique.")

            # density (undirected simple graph)
            density = (2 * e) / (n * (n - 1)) if n > 1 else 0.0
            features.append(f"Product graph density: {density:.3f}")
            if density > 0.50:
                features.append("Dense product co-interaction.")
            elif density < 0.10:
                features.append("Sparse product co-interaction.")

            # 4) weighted degree centrality (normalised)
            deg = nk.centrality.DegreeCentrality(g, True, True)
            deg.run()
            scores = deg.scores()
            if scores:
                top_idx = int(np.argmax(scores))
                central_sku = int(id2sku[top_idx])
                features.append(
                    f"Central product (degree): SKU_{central_sku} "
                    f"({scores[top_idx]:.0f})"
                )

        except Exception as e:
            self.logger.debug(f"Err product graph: {e}")
            features.append("Err product graph")

class IntentFeatureExtractor(FeatureExtractorBase):
    """Extract search intent and interest patterns, with simple cart abandon signal."""

    def extract_features(
        self,
        client_id: int,
        events: pl.DataFrame,
        now: datetime,
    ) -> List[str]:
        features: List[str] = []

        if events.height == 0:
            return ["No activity data for intent analysis"]

        try:
            # Retailrocket contains views, cart additions and transactions,
            # but no search-query events.
            self._extract_Browse_intent(events, features)
            self._extract_funnel_position(events, features, now)
            self._extract_cart_abandon_signal(events, features)

        except Exception as exc:
            self.logger.error(
                f"Error extracting intent features for client {client_id}: {exc}",
                exc_info=self.parent.debug_mode,
            )
            features.append("Error during intent feature extraction.")

        return features

    def _extract_Browse_intent(self, events: pl.DataFrame, features: List[str]) -> None:
        # Initialize cat_counts and total_cat_visits to default values
        cat_counts = pl.DataFrame()  # Default empty DataFrame
        total_cat_visits = 0         # Default to 0
    
        try:
            page_visits = events.filter(pl.col('event_type') == pl.lit('page_visit', dtype=pl.Categorical))
            
            if page_visits.height < 3:
                return # Early exit if not enough page visits to analyze
    
            num_sessions = self._count_sessions(events)
            
            if num_sessions > 0:
                visits_per_session = page_visits.height / num_sessions
                if visits_per_session > 15:
                    features.append(f"Intensive browser (~{visits_per_session:.1f} pages/session)")
                elif visits_per_session < 3: 
                    features.append(f"Shallow browser (~{visits_per_session:.1f} pages/session)")
    
            if 'category_id' in page_visits.columns:
                valid_category_page_visits = page_visits.filter(pl.col('category_id').is_not_null())
                
                if not valid_category_page_visits.is_empty():
                    calculated_cat_counts = valid_category_page_visits.group_by('category_id').agg(
                        pl.col('category_id').count().alias('count')  # Changer pl.count() en pl.col().count()
                    )
                
                    if not calculated_cat_counts.is_empty():
                        cat_counts = calculated_cat_counts
                        if 'count' in cat_counts.columns:
                            total_cat_visits = cat_counts.get_column('count').sum()
                        else:
                            self.logger.debug("Browse intent: 'count' column unexpectedly missing after grouping categories.")
                            total_cat_visits = 0
    
                        if total_cat_visits > 0 and 'count' in cat_counts.columns: 
                            max_cat_visits = cat_counts.get_column('count').max()
                            if max_cat_visits is not None: 
                                top_category_share = (max_cat_visits / total_cat_visits)
                                if top_category_share > 0.75:
                                    features.append(f"Single-category focus browse")
                                elif top_category_share < 0.40 and cat_counts.height >= 3:
                                    features.append(f"Multi-category explorer browse")
    
        except Exception as e:
            client_id_info = f"client {events['client_id'][0]}" if not events.is_empty() and 'client_id' in events.columns else "unknown client"
            self.logger.debug(f"Error in _extract_Browse_intent for {client_id_info}: {e}", exc_info=True)
            features.append("Err browse intent.")

        # --- Diversity score ---
        # This section is now safe because cat_counts and total_cat_visits are always defined.
        if not cat_counts.is_empty() and cat_counts.height > 1 and total_cat_visits > 0:
            try:
                if 'count' in cat_counts.columns: # Double check 'count' column exists
                    probs_series = cat_counts.get_column('count') / total_cat_visits 
                    probs_numpy = probs_series.to_numpy()
                    
                    # Filter out zero or negative probabilities to avoid log2 issues and ensure valid input
                    probs_filtered = probs_numpy[probs_numpy > 0]
                    
                    if probs_filtered.size > 0: # Ensure there are valid probabilities after filtering
                        H = float(-(probs_filtered * np.log2(probs_filtered)).sum()) # Shannon entropy
                        features.append(f"Category Diversity:{H:.2f}")
                    else:
                        self.logger.debug("Diversity score not calculated: no valid probabilities after filtering.")
                else:
                    self.logger.debug("Diversity score not calculated: 'count' column missing in cat_counts.")
            except Exception as e_diversity:
                 self.logger.debug(f"Error calculating diversity score: {e_diversity}", exc_info=True)
                 # features.append("ErrCalculatingDiversityScore") # Optionally add a specific error feature



    def _extract_funnel_position(self, events: pl.DataFrame, features: List[str], now: datetime) -> None:
        try:
            tmp = events.group_by('event_type').agg(pl.col('event_type').count().alias('count'))  # Changer pl.count() en pl.col().count()
            event_counts = {row['event_type']: row['count'] for row in tmp.iter_rows(named=True)}
            views = event_counts.get("page_visit", 0)
            cart_adds = event_counts.get("add_to_cart", 0)
            purchases = event_counts.get("product_buy", 0)

            if purchases > 0:
                features.append("Funnel Stage: Conversion")
            elif cart_adds > 0:
                features.append("Funnel Stage: Consideration")
            elif views > 5:
                features.append("Funnel Stage: Browsing")
            elif views > 0:
                features.append("Funnel Stage: Awareness")
            else:
                features.append("Funnel Stage: Inactive")

            # Last action type already handled by TemporalExtractor recency
            # last_event = events.sort("timestamp", descending=True).row(0, named=True)
            # if last_event:
            #     last_type = last_event['event_type']; last_time = last_event['timestamp']
            #     days_since_last = (now - last_time).days if last_time else -1
            #     recency_tag = f"(last {days_since_last+1}d)" if days_since_last < 30 and days_since_last >=0 else "(>30d ago)" if days_since_last >=0 else ""
            #     # Mapping simple
            #     action_map = {'product_buy':'Purchase', 'add_to_cart':'Cart Add', 'search_query':'Search', 'page_visit':'Visit', 'remove_from_cart':'Cart Remove'}
            #     features.append(f"Last action type: {action_map.get(last_type, last_type)} {recency_tag}")

        except Exception as e: self.logger.debug(f"Err funnel pos: {e}"); features.append("Err funnel position.")

    def _extract_cart_abandon_signal(self, events: pl.DataFrame, features: List[str]) -> None:
        try:
            adds  = events.filter(pl.col('event_type') == 'add_to_cart').height
            buys  = events.filter(pl.col('event_type') == 'product_buy').height
            if adds > 0 and buys / adds < 0.2:
                features.append("Cart Behavior: High abandon ratio")
            elif adds > 0 and buys / adds > 0.8:
                features.append("Cart Behavior: High conversion ratio")            
            
        except Exception as e:
            self.logger.debug(f"Err cart abandon signal: {e}")
            features.append("Err cart abandon signal.")



    def _count_sessions(self, events: pl.DataFrame) -> int:
        if events.height <= 1: return events.height
        try:
            timestamps_sorted = events.sort("timestamp")['timestamp']
            SESSION_GAP_MINUTES = 30
            time_diffs_minutes = timestamps_sorted.diff().dt.total_seconds() / 60
            session_starts = time_diffs_minutes.is_null() | (time_diffs_minutes > SESSION_GAP_MINUTES)
            num_sessions = session_starts.sum()
            return int(num_sessions)
        except Exception as e: self.logger.error(f"Err counting sessions: {e}"); return 1


class PriceFeatureExtractor(FeatureExtractorBase):
    """Extract price sensitivity and purchase behavior features"""
    # --- Code inchangé ---
    # (Ajouter 'now' comme argument non utilisé)
    def extract_features(self, client_id: int, events: pl.DataFrame, now: datetime) -> List[str]:
        if 'price_bucket' not in events.columns: return ["Price features skipped."]
        features = []
        try:
            events_with_price = events.filter(pl.col('price_bucket').is_not_null())
            if events_with_price.height == 0: return ["No price data."]
            self._extract_price_range(events_with_price, features)
            self._extract_price_sensitivity(client_id, events_with_price, features)
            has_discount_cols = any(c in events.columns for c in ['discount', 'discount_percentage', 'original_price'])
            if has_discount_cols: self._extract_discount_patterns(events_with_price, features)
            # else: features.append("Discount patterns skipped.") # Optionnel
        except Exception as e: self.logger.error(f"Err price client {client_id}: {e}"); features.append("Err price")
        # --- RFM quick tag ---
        purchases = events_with_price.filter(pl.col('event_type') == 'product_buy')
        if purchases.height:
            rec = (now - purchases['timestamp'].max()).days               # Recency
            freq = purchases.filter(pl.col('timestamp') >= now - timedelta(days=90)).height
            mon = purchases['price_bucket'].mean()                        # Monetary (moyenne des buckets)
            features.append(f"RFM:{rec}:{freq}:{mon:.0f}")
               
        return features

    def _extract_price_range(self, events: pl.DataFrame, features: List[str]) -> None:
        try:
            relevant_events = events.filter( pl.col('event_type').is_in(['page_visit', 'add_to_cart', 'product_buy']) )
            if relevant_events.height == 0: return
            price_stats = relevant_events.select(pl.col('price_bucket')).describe() # Utiliser 'price_bucket'
            stats_dict = {row[0]: row[1] for row in price_stats.iter_rows()}
            min_price = stats_dict.get('min'); max_price = stats_dict.get('max')
            avg_price = stats_dict.get('mean'); std_price = stats_dict.get('std')
            count = stats_dict.get('count')
            if count is not None and count >= 2:
                if min_price is not None and max_price is not None:
                     features.append(f"Interacted price range (bucket): {min_price:.0f} - {max_price:.0f} (avg {avg_price:.0f})")
                     price_range = max_price - min_price
                     if price_range > 30: features.append("Wide price exploration.")
                     elif price_range < 10: features.append("Narrow price focus.")
                purchase_prices = relevant_events.filter(pl.col('event_type') == pl.lit('product_buy', dtype=pl.Categorical))['price_bucket']
                if purchase_prices.len() >= 2:
                    avg_purchase = purchase_prices.mean(); std_purchase = purchase_prices.std()
                    if avg_purchase is not None and avg_purchase > 0 and std_purchase is not None:
                         cv = std_purchase / avg_purchase
                         if cv < 0.15: features.append("Consistent purchase price.")
                         elif cv > 0.4: features.append("Varied purchase prices.")
        except Exception as e: self.logger.debug(f"Err price range: {e}"); features.append("Err price range.")

    def _extract_price_sensitivity(self, client_id: int, events: pl.DataFrame, features: List[str]) -> None:
        try:
            price_col = 'price_bucket'
            cart_events = events.filter(pl.col('event_type') == pl.lit('add_to_cart', dtype=pl.Categorical))
            purchase_events = events.filter(pl.col('event_type') == pl.lit('product_buy', dtype=pl.Categorical))
            if cart_events.height > 0 and purchase_events.height > 0:
                avg_cart_price = cart_events[price_col].mean()
                avg_purchase_price = purchase_events[price_col].mean()
                if avg_cart_price is not None and avg_purchase_price is not None and avg_cart_price > 0:
                    ratio = avg_purchase_price / avg_cart_price
                    if ratio < 0.8: features.append("Sensitivity: High (buys cheaper than adds)")
                    elif ratio > 1.2: features.append("Sensitivity: Low (buys similar/pricier)")
                    else: features.append("Sensitivity: Moderate")

            # Abandon vs price logic
            if cart_events.height > 0 and 'sku' in events.columns:
                cart_skus_prices = cart_events.select(['sku', price_col]).drop_nulls()
                if cart_skus_prices.height > 0:
                     purchased_skus = purchase_events.select('sku').drop_nulls()['sku'].unique().to_list()
                     if purchased_skus:
                          abandoned_items = cart_skus_prices.filter(~pl.col('sku').is_in(purchased_skus))
                          purchased_carted_items = cart_skus_prices.filter(pl.col('sku').is_in(purchased_skus))
                          if abandoned_items.height > 0 and purchased_carted_items.height > 0:
                               avg_abandoned_price = abandoned_items[price_col].mean()
                               avg_purchased_price = purchased_carted_items[price_col].mean()
                               if avg_abandoned_price is not None and avg_purchased_price is not None:
                                    if avg_abandoned_price > avg_purchased_price * 1.2:
                                         features.append("Tends to abandon higher-priced cart items.")
        except Exception as e: self.logger.error(f"Err price sensitivity client {client_id}: {e}"); features.append("Err price sensitivity.")

    def _extract_discount_patterns(self, events: pl.DataFrame, features: List[str]) -> None:
        # Placeholder - logic depends on actual discount columns
        relevant_discount_cols = [c for c in ['discount', 'discount_percentage', 'original_price'] if c in events.columns]
        if relevant_discount_cols:
             features.append(f"Discount info present ({', '.join(relevant_discount_cols)}), analysis TBD.")

class AvailabilityFeatureExtractor(FeatureExtractorBase):
    """Extract stock-availability interaction patterns from Retailrocket."""

    def extract_features(
        self,
        client_id: int,
        events: pl.DataFrame,
        now: datetime,
    ) -> List[str]:
        if "is_available" not in events.columns:
            return []

        product_events = events.filter(
            pl.col("event_type").is_in(
                ["page_visit", "add_to_cart", "product_buy"]
            )
            & pl.col("sku").is_not_null()
        )

        if product_events.is_empty():
            return []

        known = product_events.filter(
            pl.col("is_available").is_not_null()
        )

        if known.is_empty():
            return ["Availability status unavailable for interactions"]

        features: List[str] = []

        coverage = known.height / product_events.height
        features.append(
            f"Availability known for {coverage:.1%} of interactions"
        )

        available_share = (
            known.filter(pl.col("is_available") == 1).height / known.height
        )

        if available_share >= 0.80:
            features.append("Mostly interacted with available products")
        elif available_share <= 0.40:
            features.append("Frequent interaction with unavailable products")
        else:
            features.append("Mixed available and unavailable product interactions")

        views = known.filter(pl.col("event_type") == "page_visit")
        if not views.is_empty():
            unavailable_views = views.filter(
                pl.col("is_available") == 0
            ).height

            if unavailable_views > 0:
                unavailable_view_share = unavailable_views / views.height
                features.append(
                    f"Unavailable product views: {unavailable_view_share:.1%}"
                )

        carts = known.filter(pl.col("event_type") == "add_to_cart")
        if not carts.is_empty():
            available_cart_share = (
                carts.filter(pl.col("is_available") == 1).height
                / carts.height
            )
            features.append(
                f"Available-at-cart share: {available_cart_share:.1%}"
            )

        purchases = known.filter(pl.col("event_type") == "product_buy")
        if not purchases.is_empty():
            available_purchase_share = (
                purchases.filter(pl.col("is_available") == 1).height
                / purchases.height
            )

            if available_purchase_share == 1.0:
                features.append("All observed purchases were available")
            else:
                features.append(
                    f"Available-at-purchase share: "
                    f"{available_purchase_share:.1%}"
                )

        return features

class SocialFeatureExtractor(FeatureExtractorBase):
    """Extract social and competitive factors features"""

    def extract_features(self, client_id: int, events: pl.DataFrame, now: datetime) -> List[str]:
        features = []
        if self.parent.product_popularity is None or self.parent.product_popularity.height == 0:
            return ["Popularity data not available."]
        try:
            self._extract_popularity_patterns(client_id, events, features)
            self._extract_category_popularity_patterns(client_id, events, features)  # ← AJOUTER
        except Exception as e:
            self.logger.error(f"Err social client {client_id}: {e}")
            features.append("Err social")
        return features


    def _extract_category_popularity_patterns(
        self,
        client_id: int,
        events: pl.DataFrame,
        features: List[str]
    ) -> None:
        """Extract patterns based on category popularity"""
        try:
            if self.parent.category_popularity is None or self.parent.category_popularity.height == 0:
                return
                
            # Get categories this user interacted with
            user_cats = events.filter(
                pl.col('category_id').is_not_null()
            )['category_id'].unique().to_list()
            
            if not user_cats:
                return
                
            # Get popularity scores for user's categories
            user_cat_pop = self.parent.category_popularity.filter(
                pl.col('category_id').is_in(user_cats)
            )
            
            if user_cat_pop.height == 0:
                return
                
            # Average popularity of user's categories
            avg_user_cat_pop = user_cat_pop['category_popularity_score'].mean()
            global_avg_cat_pop = self.parent.category_popularity['category_popularity_score'].mean()
            
            if global_avg_cat_pop and global_avg_cat_pop > 0:
                ratio = avg_user_cat_pop / global_avg_cat_pop
                
                if ratio > 1.3:
                    features.append("Category affinity: Popular categories")
                elif ratio < 0.7:
                    features.append("Category affinity: Niche categories")
                    
            # Top category by popularity
            if 'category_id' in events.columns:
                cat_counts = (
                    events.filter(pl.col('category_id').is_not_null())
                    .group_by('category_id')
                    .agg(pl.len().alias('interactions'))
                    .join(
                        self.parent.category_popularity.select(['category_id', 'category_popularity_score']),
                        on='category_id',
                        how='left'
                    )
                )
                
                if cat_counts.height > 0:
                    # Most popular category they interact with
                    top_popular = cat_counts.sort('category_popularity_score', descending=True).head(1)
                    if top_popular.height > 0:
                        cat_id = top_popular['category_id'][0]
                        pop_score = top_popular['category_popularity_score'][0]
                        features.append(f"Most popular category: CAT_{cat_id} (score: {pop_score:.0f})")
                        
        except Exception as e:
            self.logger.debug(f"Error in category popularity: {e}")    
    # ------------------------------------------------------------------
    # SOCIAL : popularité + focus catégorie
    # ------------------------------------------------------------------
    def _extract_popularity_patterns(
        self,
        client_id: int,
        events: pl.DataFrame,
        features: List[str]
    ) -> None:
        try:
            # ---------- 1) Tous les événements produit ----------
            product_events = events.filter(
                pl.col('event_type')
                  .is_in(['page_visit', 'add_to_cart', 'product_buy'])
                & pl.col('sku').is_not_null()
            )
            if product_events.height == 0:
                return

            user_skus = (
                product_events['sku']
                .unique()
                .drop_nulls()
                .to_list()
            )
            if not user_skus:
                return

            user_pop = self.parent.product_popularity.filter(
                pl.col('sku').is_in(user_skus)
            )
            if user_pop.height == 0:
                return

            avg_user_pop   = user_pop['popularity_score'].mean()
            global_avg_pop = self.parent.product_popularity[
                'popularity_score'
            ].mean()

            # ---------- 2) Popularité globale ----------
            if global_avg_pop and global_avg_pop > 0:
                ratio = avg_user_pop / global_avg_pop

                if ratio > 1.3:
                    features.append("Affinity: Popular products")
                elif ratio < 0.7:
                    features.append("Affinity: Niche products")
                else:
                    features.append("Affinity: Average popularity")

                delta = avg_user_pop - global_avg_pop
                if abs(delta) >= 1:
                    features.append(f"Popularity Δ: {delta:+.1f}")

            # ---------- 3) Popularité VS moyenne des catégories visitées ----------
            if (
                'category_id' in events.columns
                and self.parent.category_popularity is not None
                and self.parent.category_popularity.height > 0
            ):
                cat_ids = (
                    events.filter(pl.col('category_id').is_not_null())
                          ['category_id']
                          .unique()
                          .to_list()
                )
                if cat_ids:
                    avg_cat_pop = self.parent.category_popularity.filter(
                        pl.col('category_id').is_in(cat_ids)
                    )['category_popularity_score'].mean()

                    if avg_cat_pop and avg_cat_pop > 0:
                        cat_ratio = avg_user_pop / avg_cat_pop
                        if cat_ratio > 1.3:
                            features.append("Affinity: Popular VS category avg")
                        elif cat_ratio < 0.7:
                            features.append("Affinity: Niche VS category avg")

            # ---------- 4) Con­cen­tra­tion sur la TOP catégorie (focus) ----------
            if 'category_id' in events.columns:
                page_visits = events.filter(
                    (pl.col('event_type') == 'page_visit')
                    & pl.col('category_id').is_not_null()
                )
                if page_visits.height >= 10:
                    cat_cnts = (
                        page_visits
                        .group_by("category_id")
                        .agg(pl.len().alias("cnt"))
                        .sort("cnt", descending=True)
                    )

                    total_views = cat_cnts["cnt"].sum()

                    if total_views and total_views > 0:
                        top_row = cat_cnts.row(0, named=True)
                        top_share = top_row["cnt"] / total_views

                        if top_share >= 0.75:
                            features.append(
                                "Browsing highly concentrated on one category"
                            )
                        elif top_share <= 0.40 and cat_cnts.height >= 3:
                            features.append(
                                "Browsing spread across many categories"
                            )

            # ---------- 5) Différence vue ↔ panier ----------
            view_events = product_events.filter(
                pl.col('event_type') == pl.lit('page_visit', dtype=pl.Categorical)
            )
            cart_events = product_events.filter(
                pl.col('event_type') == pl.lit('add_to_cart', dtype=pl.Categorical)
            )
            if view_events.height > 0 and cart_events.height > 0:
                view_skus = (
                    view_events['sku'].unique().drop_nulls().to_list()
                )
                cart_skus = (
                    cart_events['sku'].unique().drop_nulls().to_list()
                )

                if view_skus and cart_skus:
                    avg_view_pop = self.parent.product_popularity.filter(
                        pl.col('sku').is_in(view_skus)
                    )['popularity_score'].mean()

                    avg_cart_pop = self.parent.product_popularity.filter(
                        pl.col('sku').is_in(cart_skus)
                    )['popularity_score'].mean()

                    if (
                        avg_view_pop is not None
                        and avg_cart_pop is not None
                        and avg_view_pop > 0
                    ):
                        ratio_cart_view = avg_cart_pop / avg_view_pop
                        if ratio_cart_view > 1.2:
                            features.append("Adds more popular items to cart than viewed.")
                        elif ratio_cart_view < 0.8:
                            features.append("Adds less popular items to cart than viewed.")

        except Exception as e:
            self.logger.debug(f"Err popularity: {e}")
            features.append("Err popularity patterns.")


class RetailrocketGlobalPopularityFeatureExtractor(FeatureExtractorBase):
    """
    Extract a user's interaction coverage with globally popular Retailrocket
    products and categories.

    The global top sets are derived from product_popularity and
    category_popularity, which are computed only from the active observation
    history. Therefore temporal-split runs remain leakage-free.
    """

    def extract_features(
        self,
        client_id: int,
        events: pl.DataFrame,
        now: datetime,
    ) -> List[str]:
        features: List[str] = []

        self._extract_top_sku_overlap(events, features)
        self._extract_top_category_overlap(events, features)

        return features

    def _extract_top_sku_overlap(
        self,
        events: pl.DataFrame,
        features: List[str],
    ) -> None:
        product_popularity = self.parent.product_popularity

        if (
            product_popularity is None
            or product_popularity.is_empty()
            or "popularity_score" not in product_popularity.columns
        ):
            return

        product_events = events.filter(
            pl.col("sku").is_not_null()
            & pl.col("event_type").is_in(
                ["page_visit", "add_to_cart", "product_buy"]
            )
        )

        if product_events.is_empty():
            return

        global_top_skus = (
            product_popularity
            .sort("popularity_score", descending=True)
            .head(TOP_RETAILROCKET_SKUS)
            .select(["sku", "popularity_score"])
        )

        top_rank_by_sku = {
            int(row["sku"]): rank
            for rank, row in enumerate(
                global_top_skus.iter_rows(named=True),
                start=1,
            )
            if row["sku"] is not None
        }

        matched_events = product_events.filter(
            pl.col("sku").is_in(list(top_rank_by_sku.keys()))
        )

        if matched_events.is_empty():
            return

        coverage = matched_events.height / product_events.height
        matched_counts = (
            matched_events
            .group_by("sku")
            .agg(pl.len().alias("event_count"))
            .to_dicts()
        )

        ranked_matches = sorted(
            matched_counts,
            key=lambda row: (
                top_rank_by_sku[int(row["sku"])],
                -int(row["event_count"]),
            ),
        )

        features.append(
            f"GLOBAL_TOP_SKU_COVERAGE:{coverage:.1%}"
        )
        features.append(
            f"GLOBAL_TOP_SKU_UNIQUE_HITS:{len(ranked_matches)}"
        )

        for row in ranked_matches[:3]:
            sku = int(row["sku"])
            features.append(
                f"GLOBAL_TOP_SKU_HIT:SKU_{sku}"
                f"(rank={top_rank_by_sku[sku]},events={int(row['event_count'])})"
            )

    def _extract_top_category_overlap(
        self,
        events: pl.DataFrame,
        features: List[str],
    ) -> None:
        category_popularity = self.parent.category_popularity

        if (
            category_popularity is None
            or category_popularity.is_empty()
            or "category_popularity_score" not in category_popularity.columns
            or "category_id" not in events.columns
        ):
            return

        category_events = events.filter(
            pl.col("category_id").is_not_null()
            & pl.col("event_type").is_in(
                ["page_visit", "add_to_cart", "product_buy"]
            )
        )

        if category_events.is_empty():
            return

        global_top_categories = (
            category_popularity
            .sort("category_popularity_score", descending=True)
            .head(TOP_RETAILROCKET_CATEGORIES)
            .select(["category_id", "category_popularity_score"])
        )

        top_rank_by_category = {
            int(row["category_id"]): rank
            for rank, row in enumerate(
                global_top_categories.iter_rows(named=True),
                start=1,
            )
            if row["category_id"] is not None
        }

        matched_events = category_events.filter(
            pl.col("category_id").is_in(list(top_rank_by_category.keys()))
        )

        if matched_events.is_empty():
            return

        coverage = matched_events.height / category_events.height
        matched_counts = (
            matched_events
            .group_by("category_id")
            .agg(pl.len().alias("event_count"))
            .to_dicts()
        )

        ranked_matches = sorted(
            matched_counts,
            key=lambda row: (
                top_rank_by_category[int(row["category_id"])],
                -int(row["event_count"]),
            ),
        )

        features.append(
            f"GLOBAL_TOP_CATEGORY_COVERAGE:{coverage:.1%}"
        )
        features.append(
            f"GLOBAL_TOP_CATEGORY_UNIQUE_HITS:{len(ranked_matches)}"
        )

        for row in ranked_matches[:3]:
            category_id = int(row["category_id"])
            features.append(
                f"GLOBAL_TOP_CATEGORY_HIT:CAT_{category_id}"
                f"(rank={top_rank_by_category[category_id]},"
                f"events={int(row['event_count'])})"
            )
# --- Main Generator Class ---
# --- Constants for raw sequence generation ---
SEP_TOKEN = "</s>"
SESSION_START_TOKEN = "T_SessionStart"
MAX_HISTORY_EVENTS_TO_CONSIDER = 8192
RAW_SEQUENCE_LAST_EVENTS = 512
# --- Temporal discretization helpers ---
def discretize_timedelta(delta: timedelta) -> str:
    seconds = delta.total_seconds()
    if seconds < 0: return "T_Error"
    if seconds <= 5: return "T_0-5s"
    if seconds <= 30: return "T_5-30s"
    if seconds <= 120: return "T_30s-2m"
    if seconds <= 600: return "T_2m-10m"
    if seconds <= 1800: return "T_10m-30m"
    return "T_30m+"

def discretize_time_of_day(ts: datetime) -> str:
    hour = ts.hour
    if 5 <= hour < 12: return "Morning"
    if 12 <= hour < 18: return "Afternoon"
    if 18 <= hour < 22: return "Evening"
    return "Night"

def discretize_day_of_week(ts: datetime) -> str:
    return "Weekday" if ts.weekday() < 5 else "Weekend"

# --- Helper: paires de co-achat / co-panier ---------------------------------
def top_co_pairs(events: pl.DataFrame, top_k: int = 5) -> List[str]:
    rel = (events.filter(
            pl.col('event_type').is_in(['product_buy', 'add_to_cart']) &
            pl.col('sku').is_not_null())
           .sort('timestamp'))
    if rel.height < 2:
        return []

    # 30-min sessions
    gaps = rel['timestamp'].diff().dt.total_seconds() / 60
    rel  = rel.with_columns(((gaps.is_null()) | (gaps > 30)).cum_sum().alias('sid'))

    c = Counter()
    for sess in rel.partition_by('sid', as_dict=False):
        skus = sess['sku'].drop_nulls().unique().to_list()
        if len(skus) >= 2:
            c.update(combinations(sorted(skus), 2))

    return [
        f"CO_PAIR:SKU_{i}~SKU_{j} ({cnt}x)"
        for (i, j), cnt in c.most_common(top_k) if cnt >= 2
    ]

def top_co_categories(events: pl.DataFrame, top_k: int = 5) -> List[str]:
    """
    Return tags  CAT_PAIR:CAT_i~CAT_j (cnt×)  for category pairs that co-occur
    within the same 30-minute session.
    """
    if events.height < 2 or 'category_id' not in events.columns:
        return []

    # keep only page-views & purchases that have a category
    rel = (
        events.filter(
            pl.col('event_type').is_in(['page_visit', 'product_buy']) &
            pl.col('category_id').is_not_null()
        )
        .sort('timestamp')
    )
    if rel.height < 2:
        return []

    # add 30-minute session IDs (vectorised)
    gaps = rel['timestamp'].diff().dt.total_seconds() / 60
    rel  = rel.with_columns(
        ((gaps.is_null()) | (gaps > 30)).cum_sum().alias('sid')
    )

    # count unique category pairs per session
    pair_counts: Counter[tuple[int, int]] = Counter()
    for sess in rel.partition_by('sid', as_dict=False):
        cats = sess['category_id'].drop_nulls().unique().to_list()
        if len(cats) >= 2:
            pair_counts.update(combinations(sorted(cats), 2))

    return [
        f"CAT_PAIR:CAT_{i}~CAT_{j} ({cnt}x)"
        for (i, j), cnt in pair_counts.most_common(top_k)
        if cnt >= 2
    ]

# ------------------------------------------------------------------
# Helper : stats conversion / abandon panier  (safe + polars only)
# ------------------------------------------------------------------
def cart_conversion_stats(events: pl.DataFrame) -> list[str]:
    """
    Retourne :
      • CART_CONV_RATE            – % des ajouts convertis en achat
      • AVG_CART2BUY_MIN          – délai moyen (min) entre add→buy
    Avec garde-fous quand il n'y a ni add_to_cart ni product_buy.
    """
    # 1) Sélection des lignes utiles
    adds = (
        events.filter(pl.col("event_type") == "add_to_cart")
              .filter(pl.col("sku").is_not_null())
    )
    buys = (
        events.filter(pl.col("event_type") == "product_buy")
              .filter(pl.col("sku").is_not_null())
    )

    if adds.is_empty():
        return ["Cart conversion: no cart activity"]          

    # 2) Taux de conversion panier → achat
    purchased_skus   = buys["sku"].unique().to_list()
    converted_height = (
        adds.filter(pl.col("sku").is_in(purchased_skus)).height
        if purchased_skus else
        0
    )
    conv_rate = converted_height / adds.height               

    lines = [f"CART_CONV_RATE={conv_rate:.2f}"]

    # 3) Délai moyen add→buy (si achetés)
    if not buys.is_empty() and purchased_skus:
        cart_times = (adds.select(["sku", "timestamp"])
                          .rename({"timestamp": "add_ts"}))
        buy_times  = (buys.select(["sku", "timestamp"])
                          .rename({"timestamp": "buy_ts"}))

        delays = (
            buy_times.join(cart_times, on="sku", how="inner")
                     .filter(pl.col("add_ts") < pl.col("buy_ts"))
                     .with_columns(
                         ((pl.col("buy_ts") - pl.col("add_ts"))
                          .dt.total_seconds() / 60).alias("delay_min")
                     )["delay_min"]
        )

        if delays.len() > 0:
            lines.append(f"AVG_CART2BUY_MIN={delays.mean():.1f}")

    return lines


class AdvancedUBMGenerator:
    """Generates advanced Universal Behavioral Profiles with
    multi-modal, multi-resolution features (lazy Polars pipeline)"""

    def __init__(self, data_dir: str, cache_dir: Optional[str] = None, debug_mode: bool = False):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else (self.data_dir / "cache")
        self.debug_mode = debug_mode
        os.makedirs(self.cache_dir, exist_ok=True)

        # For lazy pipeline
        self.lazy_all: Optional[pl.LazyFrame] = None

        # Cached or materialized data
        self.events_df: Optional[pl.DataFrame] = None
        self.sku_properties_for_join: Optional[pl.DataFrame] = None
        self.sku_properties_dict: Dict[int, Dict[str, Any]] = {}
        self.product_popularity: Optional[pl.DataFrame] = None
        self.category_popularity: Optional[pl.DataFrame] = None
        self.global_stats: Dict[str, Any] = {}
        self.user_segments: Dict[str, List[int]] = {}
        self._extractors: Dict[str, Any] = {}
        # Retailrocket has no predefined Synerise propensity target lists.
        # We will derive popular SKUs and categories from training data later.  
        self.dataset_end: Optional[datetime] = None
        self.reference_time: Optional[datetime] = None


        self.logger = logging.getLogger(self.__class__.__name__)
        if self.debug_mode:
            self.logger.setLevel(logging.DEBUG)
            
    def _reset_data(self):
        self.logger.warning("Resetting internal dataframes and stats.")
        self.lazy_all = None
        self.events_df = None
        self.sku_properties_for_join = None
        self.sku_properties_dict = {}
        self.product_popularity = None
        self.category_popularity = None
        self.global_stats = {}
        self.user_segments = {}
        self._extractors = {}
        gc.collect()

    def load_data(
        self,
        use_cache: bool = True,
        relevant_client_ids: Optional[List[int]] = None,
        observation_end: Optional[datetime] = None,
    ) -> None:
            """            
            Build the Retailrocket event pipeline and derived profile features.
    
            Events are loaded from events.csv and enriched with confirmed
            Retailrocket item properties: category_id and time-dependent
            availability. If observation_end is provided, only earlier
            interactions are used for profile construction and global statistics.
            """
            self.logger.info(f"=== load_data called with use_cache={use_cache}, "
                             f"relevant_clients={len(relevant_client_ids) if relevant_client_ids else 'None'}, "
                             f"observation_end={observation_end}")
            
                        # ============================================================
            # Load Retailrocket source data
            # ============================================================
            self.logger.info("Loading Retailrocket source files...")

            events_path = self.data_dir / "events.csv"

            if not events_path.exists():
                raise FileNotFoundError(
                    f"Retailrocket events file not found: {events_path}"
                )

            # Retailrocket raw schema:
            # timestamp, visitorid, event, itemid, transactionid
            #
            # Internal schema retained for the existing feature extractors:
            # client_id, timestamp, sku, event_type, transaction_id, url, query
            lf_all = (
                pl.scan_csv(events_path)
                .select([
                    pl.col("visitorid")
                      .cast(pl.Int64)
                      .alias("client_id"),

                    pl.from_epoch(
                        pl.col("timestamp").cast(pl.Int64),
                        time_unit="ms"
                    ).alias("timestamp"),

                    pl.col("itemid")
                      .cast(pl.Int64)
                      .alias("sku"),

                    pl.when(pl.col("event") == "view")
                      .then(pl.lit("page_visit"))
                      .when(pl.col("event") == "addtocart")
                      .then(pl.lit("add_to_cart"))
                      .when(pl.col("event") == "transaction")
                      .then(pl.lit("product_buy"))
                      .otherwise(pl.col("event"))
                      .cast(pl.Categorical)
                      .alias("event_type"),

                    pl.col("transactionid")
                      .cast(pl.Int64, strict=False)
                      .alias("transaction_id"),

                    # Retailrocket does not provide URL or search-query events.
                    pl.lit(None).cast(pl.Utf8).alias("url"),
                    pl.lit(None).cast(pl.Utf8).alias("query"),
                ])
            )

            # Optional filtering for fast local tests.
            # Dataset-relative reference time must be computed before optional
            # debug filtering, otherwise each test user appears artificially recent.
                        # ============================================================
            # Determine temporal observation boundary
            # ============================================================
            dataset_end_df = (
                lf_all
                .select(pl.col("timestamp").max().alias("dataset_end"))
                .collect(engine="streaming")
            )
            self.dataset_end = dataset_end_df["dataset_end"][0]

            if self.dataset_end is None:
                raise ValueError("Retailrocket dataset contains no valid timestamps.")

            # When observation_end is provided, all profiles and global
            # statistics must be built only from historical events.
            self.reference_time = observation_end or self.dataset_end

            if observation_end is not None:
                self.logger.info(
                    f"Using observation cutoff: {self.reference_time}"
                )
                lf_all = lf_all.filter(
                    pl.col("timestamp") < pl.lit(self.reference_time)
                )
            else:
                self.logger.info(
                    f"Retailrocket global reference time set to {self.reference_time}"
                )

        
            if relevant_client_ids is not None:
                self.logger.info(
                    f"Keeping full observation history for global statistics; "
                    f"{len(relevant_client_ids)} clients selected for profile testing"
                )

            # ============================================================
            # Load Retailrocket item categories
            # ============================================================
            properties_paths = [
                self.data_dir / "item_properties_part1.csv",
                self.data_dir / "item_properties_part2.csv",
            ]

            missing_property_files = [
                str(path) for path in properties_paths if not path.exists()
            ]
            if missing_property_files:
                raise FileNotFoundError(
                    "Missing Retailrocket item-properties files: "
                    + ", ".join(missing_property_files)
                )

            category_properties = (
                pl.concat([
                    pl.scan_csv(properties_paths[0]),
                    pl.scan_csv(properties_paths[1]),
                ])
                .filter(pl.col("property") == "categoryid")
                .select([
                    pl.col("itemid")
                      .cast(pl.Int64)
                      .alias("sku"),

                    pl.from_epoch(
                        pl.col("timestamp").cast(pl.Int64),
                        time_unit="ms"
                    ).alias("property_timestamp"),

                    pl.col("value")
                      .cast(pl.Int64, strict=False)
                      .alias("category_id"),
                ])
                .filter(
                    pl.col("category_id").is_not_null()
                    & (pl.col("property_timestamp") <= pl.lit(self.reference_time))
                )
                .sort(["sku", "property_timestamp"])
                .group_by("sku")
                .agg(
                    pl.col("category_id").last().alias("category_id")
                )
                .collect(engine="streaming")
            )

            self.logger.info(
                f"Loaded latest category assignments for "
                f"{category_properties.height:,} items."
            )

            self.sku_properties_for_join = category_properties
            self.sku_properties_dict = {
                int(row["sku"]): {"category": int(row["category_id"])}
                for row in category_properties.iter_rows(named=True)
                if row["sku"] is not None and row["category_id"] is not None
            }

            lf_all = lf_all.join(
                category_properties.lazy(),
                on="sku",
                how="left",
            )

            # ============================================================
            # Load Retailrocket item availability
            # ============================================================
            # Availability is time-dependent. Therefore it must be joined
            # as of the event timestamp and not as one static value per SKU.
            availability_properties = (
                pl.concat([
                    pl.scan_csv(properties_paths[0]),
                    pl.scan_csv(properties_paths[1]),
                ])
                .filter(pl.col("property") == "available")
                .select([
                    pl.col("itemid")
                      .cast(pl.Int64)
                      .alias("sku"),

                    pl.from_epoch(
                        pl.col("timestamp").cast(pl.Int64),
                        time_unit="ms"
                    ).alias("property_timestamp"),

                    pl.col("value")
                      .cast(pl.Int8, strict=False)
                      .alias("is_available"),
                ])
                .filter(
                    pl.col("is_available").is_not_null()
                    & (pl.col("property_timestamp") <= pl.lit(self.reference_time))
                )
                .sort(["sku", "property_timestamp"])
                .collect(engine="streaming")
            )

            self.logger.info(
                f"Loaded availability history with "
                f"{availability_properties.height:,} property rows."
            )

            # Join the most recent availability state known at each event time.
            lf_all = (
                lf_all
                .sort(["sku", "timestamp"])
                .join_asof(
                    availability_properties.lazy(),
                    left_on="timestamp",
                    right_on="property_timestamp",
                    by="sku",
                    strategy="backward",
                )
                .drop("property_timestamp")
            )

            self.lazy_all = lf_all
            self._extractors = {}

            # ============================================================
            # Restore or compute derived Retailrocket statistics
            # ============================================================
            # Shared caches may only be reused for a full-history run.
            # A temporal observation cutoff requires separate computation
            # to prevent future information from entering the profile.
            can_restore_shared_cache = (
                use_cache
                and not self.debug_mode
                and observation_end is None
            )

            cache_loaded = False

            if can_restore_shared_cache:
                cache_loaded = self._load_calculated_data_from_cache(
                    expected_reference_time=self.reference_time,
                )

            if not cache_loaded:
                self._compute_global_statistics()

            # ============================================================
            # Debug mode: materialize only selected clients and stop here
            # ============================================================
            if self.debug_mode and relevant_client_ids is not None:
                self.logger.debug(
                    f"DEBUG: materializing events_df for "
                    f"{len(relevant_client_ids)} clients"
                )

                self.events_df = (
                    self.lazy_all
                    .filter(pl.col("client_id").is_in(relevant_client_ids))
                    .sort(["client_id", "timestamp"])
                    .collect(engine="streaming")
                )

                # Required for correct buyer/browser labels in debug profiles.
                self._segment_users()
                return

            # ============================================================
            # Full-mode derived computations
            # ============================================================
            if not cache_loaded:
                # Retailrocket provides behavioral events, categories and
                # availability, but no validated product-name or URL embeddings.
                self._segment_users()
                self._build_global_centralities()

                if use_cache and not self.debug_mode:
                    if observation_end is None:
                        self._save_calculated_data_to_cache()
                        self.logger.info(
                            "Saved derived Retailrocket statistics to cache. "
                            "Event data remains lazy."
                        )
                    else:
                        self.logger.info(
                            "Skipping shared derived-cache write for "
                            "cutoff-based run to avoid mixing temporal "
                            "evaluation states."
                        )
            else:
                self.logger.info(
                    "Reusing cached Retailrocket derived features; "
                    "skipping global statistics, segmentation, and "
                    "centrality recomputation."
                )
            
    def _collect_client_events(self, client_id: int) -> pl.DataFrame:
        """Pulls down only one client's events into memory."""
        if self.lazy_all is None:
            raise RuntimeError("Must call load_data() first.")
        return (
            self.lazy_all
              .filter(pl.col("client_id") == client_id)
              .sort("timestamp")
              .collect(engine='streaming')
        )
    # ─────────────────────────────────────────────────────────────
    #  AdvancedUBMGenerator._build_url_graph_embeddings  (NEW)
    # ─────────────────────────────────────────────────────────────
    def _build_url_graph_embeddings(self) -> None:
        """
        Full-streaming URL⇆SKU bipartite graph:
          1. écrit (src_hash, dst_hash) dans un CSV par blocs de 1 M lignes
          2. lit le CSV chunk par chunk pour créer le graphe NetworKit
          3. Node2Vec 32 d puis k-means (20 clusters)
        Pic RAM ≈ 3-4 Go quel que soit le dataset.
        """
        # ---------------------------------------------------------
    
        if self.lazy_all is None:
            return
        if os.getenv("SKIP_URL_GRAPH", "0") == "1":
            self.logger.info("SKIP_URL_GRAPH=1 → URL-SKU graph bypassed.")
            self.url_embed, self.url_centroid, self.url_cluster_map = {}, None, {}
            return
    
        self.logger.info("Building URL–SKU bipartite graph (streaming)…")
    
        # ── 1) TOP-N URLs (petit collect) ─────────────────────────
        TOP_URLS = 500
        top_urls = (
            self.lazy_all
              .filter(pl.col("url").is_not_null())
              .group_by("url")
              .agg(pl.count().alias("cnt"))
              .sort("cnt", descending=True)
              .limit(TOP_URLS)             # .head() == .limit()
              .collect()                   # ← on retire le streaming=True
              ["url"]
              .to_list()
        )
        top_urls = set(top_urls)
    
        # ── 2) génère (sid, url_hash) et (sid, sku_hash) ──────────
        with_sid = (
            self.lazy_all
              .filter(pl.col("url").is_not_null() | pl.col("sku").is_not_null())
              .with_columns(
                  (
                      (
                          (pl.col("timestamp")
                             .diff()
                             .over("client_id")
                             .dt.total_seconds() / 60)
                          .fill_null(1e9)  > 30
                      ) | (pl.col("client_id").diff().is_not_null())
                  ).alias("new_sess")
              )
              .with_columns(
                  pl.col("new_sess").cum_sum().over("client_id").alias("sid")
              )
        )
    
        MASK63 = (1 << 63) - 1            # 0x7FFF…FFFF
        
        urls = (
            with_sid
              .filter(pl.col("url").is_in(top_urls))
              .select([
                  "sid",
                  (
                      (pl.col("url")
                         .hash(seed=0)        # UInt64
                         % MASK63             # <= 2^63-1  
                      )
                      .cast(pl.Int64)         # signé OK
                  ).alias("url_hash")
              ])
              .unique()
        )
    
        skus = (
            with_sid
              .filter(pl.col("sku").is_not_null())
              .select([
                  "sid",
                  (pl.col("sku") * -1).cast(pl.Int64).alias("sku_hash")
              ])
              .unique()
        )
    
        # ── 3) jointure croisée → edges.csv (streaming v2 OK) ────
        edges_lf = (
            urls.join(skus, on="sid")
                .select([
                    pl.col("url_hash").alias("src_hash"),
                    pl.col("sku_hash").alias("dst_hash")
                ])
        )
        
        import networkit as nk
        g, label2nid = nk.Graph(0, weighted=True, directed=False), {}
        def _nid(lbl: int) -> int:
            return label2nid.setdefault(lbl, g.addNode())
        
        BATCH = 1_000_000   # lignes
        stream = edges_lf.iter_batches(batch_size=BATCH, streaming=True)
        for tbl in stream:                         # PyArrow Table
            src = tbl.column(0).to_numpy(zero_copy_only=False)
            dst = tbl.column(1).to_numpy(zero_copy_only=False)
            g.addEdges(np.vectorize(_nid)(src), np.vectorize(_nid)(dst))
        
        self.logger.info("Graph nodes=%d edges=%d", g.numberOfNodes(), g.numberOfEdges())
    
        # ── 5) Node2Vec 32 d  ─────────────────────────────────────
        n2v = nk_embed.Node2Vec(g, 1.0, 1.0, 10, 10, 32)
        n2v.run()
        emb = n2v.getFeatures()
    
        # ── 6) embeddings & centroid ──────────────────────────────
        self.url_embed = {
            lbl: emb[nid] for lbl, nid in label2nid.items() if lbl >= 0
        }
        if not self.url_embed:
            self.logger.warning("No URL embeddings – aborting.")
            self.url_centroid, self.url_cluster_map = None, {}
            return
        self.url_centroid = np.stack(list(self.url_embed.values())).mean(axis=0)
    
        # ── 7) k-means (20) pour clusteriser les URLs ─────────────
        try:
            u, vec = zip(*self.url_embed.items())
            km = MiniBatchKMeans(n_clusters=20, batch_size=4096, random_state=42)
            labels = km.fit_predict(np.stack(vec))
            self.url_cluster_map = {ui: int(lb) for ui, lb in zip(u, labels)}
            self.logger.info("URL clusters: %d", len(set(labels)))
        except Exception as e:
            self.logger.error("URL clustering failed: %s", e)
            self.url_cluster_map = {}
    
        self.logger.info("Node2Vec done: %d URL vectors", len(self.url_embed))
    # ─────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────
    # ------------------------------------------------------------------ #
    # Cat->Cat centralité globale (PageRank sur transitions)           #
    # ------------------------------------------------------------------ #
    def _compute_category_centrality(self) -> None:
        """
        Remplace le calcul classique de centrality par une passe sparse,
        tirée exclusivement du lazy frame.
        """
        if self.lazy_all is None:
            self.cat_centrality = {}
            return

        # build transition pairs lazily then collect
        df = (
            self.lazy_all
              .filter(
                  pl.col('category_id').is_not_null() &
                  pl.col('event_type').is_in(['page_visit','product_buy'])
              )
              .with_columns(
                  pl.col('category_id').shift(1).over('client_id').alias('prev_cat')
              )
              .filter(pl.col('prev_cat').is_not_null())
              .select(['prev_cat','category_id'])
        ).collect(engine='streaming')

        if df.height == 0:
            self.cat_centrality = {}
        else:
            arr = np.array(df.to_numpy(), dtype=int)
            src, dst = arr[:, 0], arr[:, 1]

            unique_edges, counts = np.unique(
                np.stack([src, dst], axis=1),
                axis=0,
                return_counts=True,
            )

            se, de = unique_edges[:, 0], unique_edges[:, 1]
            self.cat_centrality = compute_sparse_pagerank(se, de, counts)

        # Existing helper methods use both names.
        self.category_centrality = self.cat_centrality

        self.logger.info(
            f"Built sparse category centrality • CAT:{len(self.cat_centrality)}"
        )


    # ------------------------------------------------------------------ #
    # Global co‑occurrences (SKU & CAT) pour tags GLOBAL_CO_PAIR / GLOBAL_CAT_PAIR
    # ------------------------------------------------------------------ #
    # À ajouter dans text_representation_v3.py ou monkey-patch
    def _compute_global_co_pairs(self, top_k: int = 30) -> None:
        """Version optimisée qui échantillonne et utilise seulement les top items"""
        from itertools import combinations
        import numpy as np
        
        print("Computing global co-pairs (optimized)...")
        
        # 1. Limiter aux SKUs et catégories populaires
        top_skus = set()
        top_cats = set()
        
        if self.product_popularity is not None:
            # Top 5000 SKUs par popularité
            top_skus = set(
                self.product_popularity
                .sort('popularity_score', descending=True)
                .head(5000)['sku']
                .to_list()
            )
        
        if self.category_popularity is not None:
            # Top 500 catégories
            top_cats = set(
                self.category_popularity
                .sort('category_popularity_score', descending=True)
                .head(500)['category_id']
                .to_list()
            )
        
        # 2. Échantillonner les clients (10% ou 100k max)
        all_clients = (
            self.lazy_all
            .select('client_id')
            .unique()
            .collect()['client_id']
            .to_list()
        )
        
        sample_size = min(100_000, max(1, len(all_clients) // 10))
        sampled_clients = np.random.choice(all_clients, sample_size, replace=False)
        
        print(f"Sampling {sample_size:,} clients out of {len(all_clients):,}")
        
        # 3. Collecter seulement pour les clients échantillonnés
        df = (
            self.lazy_all
            .filter(
                pl.col('client_id').is_in(sampled_clients) &
                pl.col('event_type').is_in(['product_buy','add_to_cart'])
            )
            .select(['client_id','timestamp','sku','category_id'])
            .collect(engine='streaming')
        )
        
        # Filtrer par top items si disponibles
        if top_skus:
            df = df.with_columns(
                pl.when(pl.col('sku').is_in(top_skus))
                .then(pl.col('sku'))
                .otherwise(None)
                .alias('sku')
            )
        
        if top_cats:
            df = df.with_columns(
                pl.when(pl.col('category_id').is_in(top_cats))
                .then(pl.col('category_id'))
                .otherwise(None)
                .alias('category_id')
            )
        
        # 4. Sessions optimisées
        df = df.sort(['client_id','timestamp'])
        
        # Calcul vectorisé des sessions
        df = df.with_columns(
            (
                pl.col('timestamp').diff().over('client_id').dt.total_seconds() / 60
            ).alias('time_diff_min')
        )
        
        df = df.with_columns(
              (
                ((pl.col('time_diff_min') > 30) | pl.col('time_diff_min').is_null())
                .cum_sum()
                .over('client_id')
            ).alias('session_id')
        )
        
        # 5. Compter les paires par chunks pour éviter OOM
        sku_pair_counter = Counter()
        cat_pair_counter = Counter()
        
        # Grouper par chunks de 10k sessions
        unique_sessions = df.select(['client_id', 'session_id']).unique()
        n_sessions = unique_sessions.height
        chunk_size = 10_000
        
        print(f"Processing {n_sessions:,} sessions...")
        
        for i in range(0, n_sessions, chunk_size):
            chunk_sessions = unique_sessions[i:i+chunk_size]
            
            # Filtrer le df pour ce chunk
            chunk_df = df.join(chunk_sessions, on=['client_id', 'session_id'])
            
            # Traiter chaque session du chunk
            for (cid, sid), sess in chunk_df.group_by(['client_id', 'session_id']):
                skus = sess['sku'].drop_nulls().unique().to_list()
                cats = sess['category_id'].drop_nulls().unique().to_list()
                
                # Limiter les combinaisons si trop nombreuses
                if len(skus) > 20:
                    skus = skus[:20]
                if len(cats) > 10:
                    cats = cats[:10]
                
                for i, j in combinations(sorted(set(skus)), 2):
                    sku_pair_counter[(int(i), int(j))] += 1
                
                for c1, c2 in combinations(sorted(set(cats)), 2):
                    cat_pair_counter[(int(c1), int(c2))] += 1
        
        # Store results
        self.global_sku_pairs = sku_pair_counter.most_common(top_k)
        self.global_cat_pairs = cat_pair_counter.most_common(top_k)
        
        self.global_stats['global_sku_pairs'] = self.global_sku_pairs
        self.global_stats['global_cat_pairs'] = self.global_cat_pairs
        
        print(f"Found {len(sku_pair_counter)} SKU pairs, {len(cat_pair_counter)} category pairs")
        
    # --- _save & _load calculated data (unchanged) ---
    def _save_calculated_data_to_cache(self) -> None:
        """
        Save derived Retailrocket statistics and graph features.

        The raw event pipeline is intentionally not materialized here.
        It is rebuilt lazily from the CSV files on each run, while expensive
        derived statistics can be restored from this cache.
        """
        if not self.cache_dir:
            return

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # ------------------------------------------------------------
            # Cache metadata: prevents accidental reuse for another setup.
            # ------------------------------------------------------------
            metadata = {
                "cache_version": "retailrocket_full_v1",
                "dataset_end": (
                    self.dataset_end.isoformat()
                    if self.dataset_end is not None
                    else None
                ),
                "reference_time": (
                    self.reference_time.isoformat()
                    if self.reference_time is not None
                    else None
                ),
                "dataset_type": "retailrocket",
            }

            with open(
                self.cache_dir / "retailrocket_cache_metadata.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(metadata, f, indent=2)

            # ------------------------------------------------------------
            # Global statistics
            # ------------------------------------------------------------
            serializable_stats = {}

            for key, value in self.global_stats.items():
                if isinstance(value, np.ndarray):
                    serializable_stats[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_stats[key] = value.item()
                else:
                    serializable_stats[key] = value

            with open(
                self.cache_dir / "global_stats.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(serializable_stats, f, indent=2)

            # ------------------------------------------------------------
            # User segments
            # ------------------------------------------------------------
            serializable_segments = {
                key: list(value) if isinstance(value, (set, list)) else value
                for key, value in self.user_segments.items()
            }

            with open(
                self.cache_dir / "user_segments.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(serializable_segments, f)

            # ------------------------------------------------------------
            # Popularity tables
            # ------------------------------------------------------------
            if self.product_popularity is not None:
                self.product_popularity.write_parquet(
                    self.cache_dir / "product_popularity.parquet"
                )

            if self.category_popularity is not None:
                self.category_popularity.write_parquet(
                    self.cache_dir / "category_popularity.parquet"
                )

            # ------------------------------------------------------------
            # SKU properties used by text formatting
            # ------------------------------------------------------------
            with open(
                self.cache_dir / "sku_properties_dict.pkl",
                "wb",
            ) as f:
                pickle.dump(self.sku_properties_dict, f)

            # ------------------------------------------------------------
            # Graph-derived centralities
            # ------------------------------------------------------------
            with open(
                self.cache_dir / "sku_centrality.pkl",
                "wb",
            ) as f:
                pickle.dump(getattr(self, "sku_centrality", {}), f)

            with open(
                self.cache_dir / "cat_centrality.pkl",
                "wb",
            ) as f:
                pickle.dump(getattr(self, "cat_centrality", {}), f)

            self.logger.info("Calculated Retailrocket data saved to cache.")

        except Exception as exc:
            self.logger.error(
                f"Failed to save calculated Retailrocket data: {exc}",
                exc_info=True,
            )

    def _load_calculated_data_from_cache(
        self,
        expected_reference_time: Optional[datetime] = None,
    ) -> bool:
        """
        Load derived statistics for a full Retailrocket run.

        This cache is intentionally only reused for runs without an
        observation cutoff. Temporal-split runs must recompute their
        statistics from the observation history to avoid leakage.
        """
        if not self.cache_dir:
            return False

        required_files = [
            self.cache_dir / "retailrocket_cache_metadata.json",
            self.cache_dir / "global_stats.json",
            self.cache_dir / "user_segments.json",
            self.cache_dir / "product_popularity.parquet",
            self.cache_dir / "category_popularity.parquet",
            self.cache_dir / "sku_properties_dict.pkl",
            self.cache_dir / "sku_centrality.pkl",
            self.cache_dir / "cat_centrality.pkl",
        ]

        missing_files = [
            path.name for path in required_files if not path.exists()
        ]

        if missing_files:
            self.logger.info(
                "Derived Retailrocket cache incomplete; recomputing. "
                f"Missing: {', '.join(missing_files)}"
            )
            return False

        try:
            # ------------------------------------------------------------
            # Validate metadata
            # ------------------------------------------------------------
            with open(
                self.cache_dir / "retailrocket_cache_metadata.json",
                "r",
                encoding="utf-8",
            ) as f:
                metadata = json.load(f)

            if metadata.get("cache_version") != "retailrocket_full_v1":
                self.logger.info(
                    "Retailrocket cache version does not match; recomputing."
                )
                return False

            if expected_reference_time is not None:
                expected_iso = expected_reference_time.isoformat()
                cached_reference_time = metadata.get("reference_time")

                if cached_reference_time != expected_iso:
                    self.logger.info(
                        "Retailrocket cache reference time does not match "
                        "the current full run; recomputing."
                    )
                    return False

            # ------------------------------------------------------------
            # Restore global statistics and segments
            # ------------------------------------------------------------
            with open(
                self.cache_dir / "global_stats.json",
                "r",
                encoding="utf-8",
            ) as f:
                self.global_stats = json.load(f)

            # rfm_recencies must be NumPy again for the existing metric code.
            if isinstance(self.global_stats.get("rfm_recencies"), list):
                self.global_stats["rfm_recencies"] = np.array(
                    self.global_stats["rfm_recencies"],
                    dtype=int,
                )

            with open(
                self.cache_dir / "user_segments.json",
                "r",
                encoding="utf-8",
            ) as f:
                self.user_segments = json.load(f)

            # ------------------------------------------------------------
            # Restore popularity tables
            # ------------------------------------------------------------
            self.product_popularity = pl.read_parquet(
                self.cache_dir / "product_popularity.parquet"
            )

            self.category_popularity = pl.read_parquet(
                self.cache_dir / "category_popularity.parquet"
            )

            # ------------------------------------------------------------
            # Restore SKU properties and graph centralities
            # ------------------------------------------------------------
            with open(
                self.cache_dir / "sku_properties_dict.pkl",
                "rb",
            ) as f:
                self.sku_properties_dict = pickle.load(f)

            with open(
                self.cache_dir / "sku_centrality.pkl",
                "rb",
            ) as f:
                self.sku_centrality = pickle.load(f)

            with open(
                self.cache_dir / "cat_centrality.pkl",
                "rb",
            ) as f:
                self.cat_centrality = pickle.load(f)

            # Keep both names available because existing helper methods use
            # both attribute spellings.
            self.category_centrality = self.cat_centrality

            # ------------------------------------------------------------
            # Rebuild in-memory popularity lookup used by RAW_SEQUENCE POP_Q
            # ------------------------------------------------------------
            global POP_QUANT_EDGES

            if (
                self.product_popularity is not None
                and "popularity_score" in self.product_popularity.columns
            ):
                scores = (
                    self.product_popularity["popularity_score"]
                    .drop_nulls()
                    .to_numpy()
                )

                if scores.size > 0:
                    POP_QUANT_EDGES = [
                        float(np.quantile(scores, quantile))
                        for quantile in (0.25, 0.50, 0.75)
                    ]

                self.pop_score_by_sku = {
                    int(row["sku"]): float(row["popularity_score"])
                    for row in self.product_popularity
                    .select(["sku", "popularity_score"])
                    .iter_rows(named=True)
                    if row["sku"] is not None
                    and row["popularity_score"] is not None
                }

            self.logger.info(
                "Loaded derived Retailrocket statistics and centralities "
                "from cache."
            )
            return True

        except Exception as exc:
            self.logger.warning(
                f"Unable to restore Retailrocket derived cache: {exc}. "
                "Recomputing statistics."
            )
            return False

    # ------------------------------------------------------------------ #
    # _compute_global_statistics (lazy-accelerated)                     #
    # ------------------------------------------------------------------ #
    def _compute_global_statistics(self) -> None:
        self.logger.info("Computing global statistics…")
        if self.lazy_all is None:
            return
        self.global_stats = {}

        # total unique users
        users_lf = self.lazy_all.select(pl.col('client_id')).unique()
        self.global_stats['total_users'] = users_lf.collect(engine='streaming').height

        # event counts
        cnts = self.lazy_all.group_by('event_type').agg(pl.count()).collect(engine='streaming')
        self.global_stats['event_counts'] = {
            row[0]: row[1] for row in cnts.iter_rows()
        }

        # transition matrix
        try:
            self.global_stats['transition_matrix'] = self._compute_global_transition_matrix()
        except Exception as e:
            self.logger.error(f"Err transition_matrix: {e}")
            self.global_stats['transition_matrix'] = {}

        # buyer percentage
        buyers = (
            self.lazy_all
              .filter(pl.col('event_type')=='product_buy')
              .select(pl.col('client_id')).unique()
              .collect(engine='streaming').height
        )
        tot = self.global_stats['total_users'] or 1
        self.global_stats['buyer_percentage'] = buyers / tot

        # RFM recencies
        recs_lf = (
            self.lazy_all
              .filter(pl.col('event_type')=='product_buy')
              .group_by('client_id')
              .agg(pl.max('timestamp').alias('last_purchase_ts'))
        )
        df = recs_lf.collect(engine='streaming')
        now_ts = self.reference_time or datetime.now()
        recs = [ (now_ts - row['last_purchase_ts']).days 
                 for row in df.to_dicts() if row['last_purchase_ts']]
        self.global_stats['rfm_recencies'] = np.array(recs, dtype=int)

        # run other global stats functions
        for fn in (
            self._compute_cart_to_purchase_times,
            self._identify_global_sessions,
            self._compute_product_popularities,
            self._compute_category_popularities,
            self._compute_category_centrality,
            self._compute_global_co_pairs
        ):
            try:
                fn()
            except Exception as e:
                self.logger.error(f"Err {fn.__name__}: {e}")

        self.logger.info(f"Global stats computed: {list(self.global_stats.keys())}")

    def _compute_global_transition_matrix(self) -> dict:
        if self.lazy_all is None:
            return {}
        try:
            df = (
                self.lazy_all
                  .select(["client_id","timestamp","event_type"] )
                  .sort(["client_id","timestamp"])           
                  .with_columns(
                      pl.col("event_type").shift(1).over("client_id").alias("prev_event_type")
                  )
                  .filter(pl.col("prev_event_type").is_not_null())
                  .group_by(["prev_event_type","event_type"]) 
                  .agg(pl.col("event_type").count().alias("count"))  # Utiliser pl.col().count()
                  .collect(engine='streaming')
            )
            totals = (
                df.group_by("prev_event_type").agg(pl.col("count").sum().alias("total"))  # Utiliser pl.col().sum()
            )
            df = df.join(totals, on="prev_event_type").with_columns(
                (pl.col("count")/pl.col("total")).alias("probability")
            )
            types = self.lazy_all.select(pl.col("event_type")).unique().collect(engine='streaming')["event_type"].to_list()
            probs = defaultdict(lambda: defaultdict(float))
            for row in df.iter_rows(named=True):
                probs[row['prev_event_type']][row['event_type']] = row['probability']
            return {f: {t: probs[f].get(t,0.0) for t in types} for f in types}
        except Exception as e:
            self.logger.error(f"Err transition matrix calc: {e}")
            return {}

    def _compute_cart_to_purchase_times(self) -> None:
        if self.lazy_all is None:
            return
        lf = (self.lazy_all
          .filter(pl.col('sku').is_not_null() & pl.col('event_type').is_in(['add_to_cart','product_buy']))
          .select(['client_id','sku','event_type','timestamp'])
             )
        df = lf.collect(engine='streaming')
        if df.filter(pl.col('event_type')=='product_buy').is_empty() or df.filter(pl.col('event_type')=='add_to_cart').is_empty():
            return
        buys = df.filter(pl.col('event_type')=='product_buy').rename({"timestamp":"purchase_ts"})
        carts = df.filter(pl.col('event_type')=='add_to_cart').rename({"timestamp":"cart_ts"})
        joined = buys.join(carts, on=["client_id","sku"]).filter(pl.col("cart_ts")<pl.col("purchase_ts"))
        if joined.is_empty():
            return
        rec = joined.group_by(["client_id","sku","purchase_ts"]).agg(pl.max("cart_ts").alias("last_cart_ts"))
        diffs = rec.with_columns(((pl.col("purchase_ts")-pl.col("last_cart_ts")).dt.total_seconds()/60).alias("diff_min"))
        vals = diffs.filter(pl.col("diff_min")>0)["diff_min"]
        if vals.len()>0:
            self.global_stats['avg_cart_to_purchase_time'] = vals.mean()
            self.global_stats['median_cart_to_purchase_time'] = vals.median()
            self.logger.info(f"Cart-to-purchase: Avg={self.global_stats['avg_cart_to_purchase_time']:.2f}m, Median={self.global_stats['median_cart_to_purchase_time']:.2f}m")

    def _identify_global_sessions(self, session_gap_minutes: int = 30) -> None:
        if self.lazy_all is None:
            return
        df = (
            self.lazy_all
              .select(["client_id","timestamp"])  
              .sort(["client_id","timestamp"])    
              .with_columns(
                  (pl.col("timestamp").diff().over("client_id")/timedelta(minutes=1)).alias("delta_min")
              )
              .with_columns(
                  ((pl.col("delta_min")>session_gap_minutes)|pl.col("delta_min").is_null()).alias("new_sess")
              )
              .with_columns(
                  pl.col("new_sess").cum_sum().over("client_id").alias("sid")
              )
        ).collect(engine='streaming')
        
        stats = df.group_by(["client_id", "sid"]).agg([
            pl.col("timestamp").min().alias("start"),
            pl.col("timestamp").max().alias("end"),
            pl.len().alias("count")
        ]).with_columns(
            (
                (pl.col("end") - pl.col("start")).dt.total_seconds() / 60
            ).alias("duration_min")
        )
        
        
        if stats.height>0:
            agg = stats.select([
                pl.col("duration_min").mean().alias("avg_session_duration"),
                pl.col("duration_min").median().alias("median_session_duration"),
                pl.col("count").mean().alias("avg_session_events"),
                pl.col("count").median().alias("median_session_events")
            ]).row(0, named=True)
            self.global_stats.update(agg)
            upu = df.group_by("client_id").agg(pl.col("sid").n_unique().alias("sessions")).select(pl.col("sessions").mean()).item()
            self.global_stats['avg_sessions_per_user'] = upu
            self.logger.info(f"Global Session Stats: AvgDur={agg['avg_session_duration']:.2f}m, AvgEvt={agg['avg_session_events']:.1f}, AvgSess/User={upu:.1f}")

    def _compute_product_popularities(self) -> None:
        if self.lazy_all is None:
            return
        df = (
            self.lazy_all
              .filter(pl.col('sku').is_not_null())
              .group_by(['sku','event_type'])
              .agg(pl.col('sku').count().alias('count'))
        ).collect(engine='streaming')
        df = (df.pivot(index='sku', columns='event_type', values='count', aggregate_function='first')
              .fill_null(0))
        mapping = {'page_visit':'view_count','add_to_cart':'cart_count','product_buy':'purchase_count'}
        df = df.rename({k:v for k,v in mapping.items() if k in df.columns})
        for col in ['view_count','cart_count','purchase_count']:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0).alias(col))
        df = df.with_columns(
            pl.when(pl.col('view_count').sum() == 0)
            # Si pas de vues du tout, formule ajustée
            .then(pl.col('cart_count')*2 + pl.col('purchase_count')*10)
            # Sinon, formule normale
            .otherwise(pl.col('view_count')*1 + pl.col('cart_count')*3 + pl.col('purchase_count')*10)
            .alias('popularity_score')
        )
        df = df.with_columns([
            (pl.when(pl.col('view_count')>0).then(pl.col('cart_count')/pl.col('view_count')).otherwise(0.0)).alias('view_to_cart_rate'),
            (pl.when(pl.col('cart_count')>0).then(pl.col('purchase_count')/pl.col('cart_count')).otherwise(0.0)).alias('cart_to_purchase_rate'),
            (pl.when(pl.col('view_count')>0).then(pl.col('purchase_count')/pl.col('view_count')).otherwise(0.0)).alias('view_to_purchase_rate')
        ])
        self.product_popularity = df
        try:
            scores = df['popularity_score'].to_numpy()
            global POP_QUANT_EDGES
            POP_QUANT_EDGES = [float(np.quantile(scores,q)) for q in (0.25,0.5,0.75)]
            self.pop_score_by_sku = {r['sku']:r['popularity_score'] for r in df[['sku','popularity_score']].to_dicts()}
        except Exception as e:
            self.logger.warning(f"Unable to compute popularity quantiles: {e}")
            self.pop_score_by_sku = {}
        self.logger.info(f"Computed product popularity for {df.height} SKUs.")
        # category popularity similar pattern (omitted)
        
    def _compute_category_popularities(self) -> None:
        """Compute popularity metrics for categories based on product events"""
        if self.lazy_all is None:
            return
            
        # Aggregate events by category
        df = (
            self.lazy_all
              .filter(pl.col('category_id').is_not_null())
              .group_by(['category_id', 'event_type'])
              .agg(pl.len().alias('count'))
        ).collect(engine='streaming')
        
        # Pivot to get counts per event type
        df = (df.pivot(index='category_id', columns='event_type', values='count', aggregate_function='first')
              .fill_null(0))
        
        # Map event types to count columns
        mapping = {
            'page_visit': 'view_count',
            'add_to_cart': 'cart_count', 
            'product_buy': 'purchase_count'
        }
        df = df.rename({k: v for k, v in mapping.items() if k in df.columns})
        
        # Ensure all columns exist
        for col in ['view_count', 'cart_count', 'purchase_count']:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0).alias(col))
        
        # Calculate popularity score (same formula as products)
        df = df.with_columns(
            pl.when(pl.col('view_count').sum() == 0)
            .then(pl.col('cart_count')*2 + pl.col('purchase_count')*10)
            .otherwise(pl.col('view_count')*1 + pl.col('cart_count')*3 + pl.col('purchase_count')*10)
            .alias('category_popularity_score')
        )
        
        self.category_popularity = df
        self.logger.info(f"Computed category popularity for {df.height} categories.")
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # === Lazy-aware Helpers for Global Computations ==================== #
    # ------------------------------------------------------------------ #

    def _build_category_centrality(self) -> None:
        """
        Builds a directed graph Cat_i→Cat_j from successive page_visit or product_buy events,
        computes PageRank, all via lazy Polars to avoid full materialization.
        """
        if self.lazy_all is None:
            self.cat_centrality = {}
            return
        # collect only category transitions
        df = (
            self.lazy_all
              .filter(pl.col('category_id').is_not_null())
              .select(['client_id','timestamp','category_id'])
              .sort(['client_id','timestamp'])
              .with_columns(
                  pl.col('category_id').shift(-1).over('client_id').alias('next_cat')
              )
              .filter(pl.col('next_cat').is_not_null())
              .select(['category_id','next_cat'])
        ).collect(engine='streaming')

        # build graph
        G = nx.DiGraph()
        for row in df.iter_rows(named=True):
            src, dst = int(row['category_id']), int(row['next_cat'])
            if src == dst:
                continue
            if G.has_edge(src, dst):
                G[src][dst]['weight'] += 1
            else:
                G.add_edge(src, dst, weight=1)
        if G.number_of_nodes() == 0:
            self.cat_centrality = {}
            self.logger.info("Category centrality skipped (empty graph).")
            return
        # PageRank
        pr = nx.pagerank(G, weight='weight', max_iter=100, tol=1e-4)
        self.cat_centrality = pr
        self.logger.info(
            f"Built centrality maps  • SKU:{len(getattr(self,'sku_centrality',{}))}  • CAT:{len(pr)}"
        )

    def _compute_global_co_occurrences(self, session_gap: int = 30) -> None:
        """
        Counts co-occurring SKU and category pairs within user sessions (lazy + collect small slice).
        """
        if self.lazy_all is None:
            return
        # collect relevant events
        df = (
            self.lazy_all
              .filter(pl.col('event_type').is_in(['product_buy','add_to_cart','page_visit']))
              .select(['client_id','timestamp','sku','category_id'])
              .sort(['client_id','timestamp'])
        ).collect(engine='streaming')
        # identify sessions
        diff = df['timestamp'].diff().dt.total_seconds() / 60
        df = df.with_columns(((diff.is_null()) | (diff > session_gap)).cum_sum().alias('sess_id'))
        sku_pairs = Counter()
        cat_pairs = Counter()
        from itertools import combinations
        for (_cid, sess), sub in df.group_by(['client_id','sess_id']):
            skus = sub['sku'].drop_nulls().unique().to_list()
            cats = sub['category_id'].drop_nulls().unique().to_list()
            for i,j in combinations(sorted(set(skus)),2):
                sku_pairs[(int(i),int(j))] += 1
            for c1,c2 in combinations(sorted(set(cats)),2):
                cat_pairs[(int(c1),int(c2))] += 1
        self.global_stats['global_sku_pairs'] = dict(sku_pairs)
        self.global_stats['global_cat_pairs'] = dict(cat_pairs)
        self.logger.info(
            f"Global co-occurrences  SKU_pairs:{len(sku_pairs)}  CAT_pairs:{len(cat_pairs)}"
        )

    # ------------------------------------------------------------------
    # Segmentation principale : acheteurs / navigateurs actifs, etc.
    # ------------------------------------------------------------------
    def _segment_users(self) -> None:
        """
        Attaches each client to high-level behavioral segments,
        using lazy scans + small collects.
        """
        self.logger.info("Segmenting users (with dataset-relative recency)...")
        if self.lazy_all is None:
            self.logger.warning("lazy_all is None; skipping segmentation.")
            return
    
        # 1) Event counts per client & type (collect BEFORE pivot)
        df_counts = (
            self.lazy_all
              .group_by(['client_id','event_type'])
              .agg(pl.col('client_id').count().alias('count'))  # Utiliser pl.col().count() au lieu de pl.count()
              .collect(engine='streaming')
        )
        df_counts = (
            df_counts
              .pivot(
                  index='client_id',
                  columns='event_type',
                  values='count',
                  aggregate_function='first'
              )
              .fill_null(0)
        )
        # ensure all event-type cols exist
        for c in ['page_visit','product_buy','add_to_cart','search_query']:
            if c not in df_counts.columns:
                df_counts = df_counts.with_columns(pl.lit(0).alias(c))
    
        # 2) Last activity timestamp
        df_last = (
            self.lazy_all
              .group_by('client_id')
              .agg(pl.col('timestamp').max().alias('last_ts'))  # Utiliser pl.col().max() au lieu de pl.max()
              .collect(engine='streaming')
        )
        df = df_counts.join(df_last, on='client_id', how='left')
    
        # 3) Recency metrics
        max_ts_df = self.lazy_all.select(pl.col('timestamp').max()).collect(engine='streaming')
        max_ts = max_ts_df.item() if max_ts_df.height > 0 else None
        if max_ts is None:
            max_ts = datetime.now()
    
        # **Plus besoin de collect(engine='streaming') ici : df est déjà un DataFrame**
        now = self.reference_time or max_ts
        df = df.with_columns([
            pl.Series(
                "days_since_run",
                [(now   - ts).days if ts is not None else 999 for ts in df['last_ts']]
            ),
            pl.Series(
                "days_since_data_end",
                [(max_ts - ts).days if ts is not None else 999 for ts in df['last_ts']]
            ),
        ])
    
        # 4) Flags
        df = df.with_columns([
            (pl.col('product_buy') > 0).alias('is_buyer'),
            ((pl.col('page_visit') >= 5) & (pl.col('days_since_run') <= 30)).alias('active_absolute'),
            ((pl.col('page_visit') >= 5) & (pl.col('days_since_data_end') <= 30)).alias('active_relative')
        ])
    
        # 5) Build segments
        segs = {
            'buyers': df.filter(pl.col('is_buyer'))['client_id'].to_list(),
            'non_buyers': df.filter(~pl.col('is_buyer'))['client_id'].to_list(),
            'active_buyers_absolute': df.filter(pl.col('is_buyer') & pl.col('active_absolute'))['client_id'].to_list(),
            'active_browsers_absolute': df.filter(~pl.col('is_buyer') & pl.col('active_absolute'))['client_id'].to_list(),
            'active_buyers_relative': df.filter(pl.col('is_buyer') & pl.col('active_relative'))['client_id'].to_list(),
            'inactive_buyers_relative': df.filter(pl.col('is_buyer') & ~pl.col('active_relative'))['client_id'].to_list(),
            'active_browsers_relative': df.filter(~pl.col('is_buyer') & pl.col('active_relative'))['client_id'].to_list(),
            'inactive_browsers_relative': df.filter(~pl.col('is_buyer') & ~pl.col('active_relative'))['client_id'].to_list()
        }
    
        # 6) Purchase frequency segments
        df = df.with_columns(
            pl.when(pl.col('product_buy') == 0).then(pl.lit('Non-Buyer'))
              .when(pl.col('product_buy') == 1).then(pl.lit('One-Time Buyer'))
              .when((pl.col('product_buy') >= 2) & (pl.col('product_buy') <= 5)).then(pl.lit('Occasional Buyer'))
              .when(pl.col('product_buy') > 5).then(pl.lit('Frequent Buyer'))
              .alias('purchase_freq')
        )

        segs['one_time_buyers']   = df.filter(pl.col('purchase_freq') == 'One-Time Buyer')['client_id'].to_list()
        segs['occasional_buyers'] = df.filter(pl.col('purchase_freq') == 'Occasional Buyer')['client_id'].to_list()
        segs['frequent_buyers']   = df.filter(pl.col('purchase_freq') == 'Frequent Buyer')['client_id'].to_list()
    
        self.user_segments = segs
    
        # 7) Further sub-segmentation
        try:
            schema = self.lazy_all.collect_schema()

            # Retailrocket currently has no validated price_bucket column.
            if "price_bucket" in schema:
                self._segment_users_by_price_sensitivity(df)

            if "category_id" in schema:
                self._segment_users_by_category_behavior(df)

        except Exception as e:
            self.logger.error(f"Err additional segmentation: {e}")
    
        self.logger.info(
            f"User segmentation done: Buyers={len(segs['buyers'])}, "
            f"Active relative buyers={len(segs['active_buyers_relative'])}"
        )

    def _segment_users_by_price_sensitivity(self, user_counts: pl.DataFrame) -> None:
        """
        Splits users into price sensitivity segments based on avg add_to_cart vs purchase price buckets,
        computed lazily to avoid full event tables in memory.
        """
        if self.lazy_all is None:
            return
        try:
            # compute avg price for cart and buy per user
            price_lf = (
                self.lazy_all
                  .filter(pl.col('price_bucket').is_not_null() & pl.col('event_type').is_in(['add_to_cart','product_buy']))
                  .group_by(['client_id','event_type'])
                  .agg(pl.mean('price_bucket').alias('avg_price'))
            ).collect(engine='streaming')
            price_lf = price_lf.pivot(index='client_id', columns='event_type', values='avg_price', aggregate_function='first')
            df_price = user_counts.select('client_id').join(price_lf, on='client_id', how='left').fill_null(0)
            if 'add_to_cart' not in df_price.columns or 'product_buy' not in df_price.columns:
                return

            df_price = df_price.with_columns(
                (pl.when(pl.col('add_to_cart')>0)
                   .then(pl.col('product_buy')/pl.col('add_to_cart'))
                   .otherwise(None)
                 ).alias('sensitivity_ratio')
            )
            valid = df_price.filter(pl.col('sensitivity_ratio').is_not_null() & pl.col('sensitivity_ratio').is_finite())['sensitivity_ratio']
            if valid.len() > 10:
                low, high = valid.quantile(0.33), valid.quantile(0.66)
                self.user_segments['price_sensitive'] = df_price.filter(pl.col('sensitivity_ratio') < low)['client_id'].to_list()
                self.user_segments['price_moderate'] = df_price.filter((pl.col('sensitivity_ratio') >= low) & (pl.col('sensitivity_ratio') <= high))['client_id'].to_list()
                self.user_segments['price_insensitive'] = df_price.filter(pl.col('sensitivity_ratio') > high)['client_id'].to_list()
                self.logger.info(
                    f"Price segmentation: Sens={len(self.user_segments['price_sensitive'])}, "
                    f"Mod={len(self.user_segments['price_moderate'])}, "
                    f"Insens={len(self.user_segments['price_insensitive'])}"
                )
        except Exception as e:
            self.logger.error(f"Err price segmentation: {e}")

    def _segment_users_by_category_behavior(self, user_counts: pl.DataFrame) -> None:
        """
        Classifies users into category loyalty/exploration segments based on distribution of page_visit counts,
        all computed on a small collected slice.
        """
        if self.lazy_all is None:
            return
        try:
            # 1) Count page visits by category per user
            cat_lf = (
                self.lazy_all
                  .filter((pl.col('event_type')=='page_visit') & pl.col('category_id').is_not_null())
                  .group_by(['client_id','category_id'])
                  .agg(pl.count().alias('view_count'))
            )
            df_cat = cat_lf.collect(engine='streaming')
            if df_cat.height == 0:
                return

            # 2) Aggregate per user: total views, max in one category, num categories
            user_cat = (
                df_cat
                  .group_by('client_id')
                  .agg(
                      pl.sum('view_count').alias('total_views'),
                      pl.max('view_count').alias('max_views_in_one_cat'),
                      pl.count().alias('n_cats')
                  )
                  .with_columns(
                      (pl.col('max_views_in_one_cat')/pl.col('total_views')).alias('category_loyalty_score')
                  )
            )

            # 3) Join with full user list
            df_stats = user_counts.select('client_id').join(user_cat, on='client_id', how='left').fill_null(0)

            # Thresholds
            loyal_thresh = 0.75
            explorer_thresh = 0.40

            # 4) Assign segments
            self.user_segments['category_loyal'] = (
                df_stats.filter(pl.col('category_loyalty_score') >= loyal_thresh)['client_id'].to_list()
            )
            self.user_segments['category_explorer'] = (
                df_stats.filter((pl.col('category_loyalty_score') <= explorer_thresh) & (pl.col('n_cats') >= 3))['client_id'].to_list()
            )
            self.user_segments['moderate_explorer'] = (
                df_stats.filter((pl.col('category_loyalty_score') > explorer_thresh) & (pl.col('category_loyalty_score') < loyal_thresh))['client_id'].to_list()
            )

            self.logger.info(
                f"Category segmentation: Loyal={len(self.user_segments['category_loyal'])}, "
                f"Moderate={len(self.user_segments['moderate_explorer'])}, "
                f"Explorer={len(self.user_segments['category_explorer'])}"
            )
        except Exception as e:
            self.logger.error(f"Err category segmentation: {e}")


    # --- Getters ---
    def get_feature_extractors(self) -> Dict[str, FeatureExtractorBase]:
            """
            Initialize feature extractors from the available Retailrocket schema.

            Important:
            In normal/full-run mode we keep the data lazy and do not materialize
            the complete event table into self.events_df. Therefore extractor
            activation must be based on the lazy schema whenever possible.
            """
            if self._extractors:
                return self._extractors

            self.logger.debug("Initializing feature extractors...")

            self._extractors = {
                "temporal": TemporalFeatureExtractor(self),
                "sequence": SequenceFeatureExtractor(self),
                "churn_propensity": ChurnPropensityFeatureExtractor(self),
            }

            # Determine available columns without forcing complete materialization.
            if self.lazy_all is not None:
                available_columns = set(self.lazy_all.collect_schema().names())
            elif self.events_df is not None:
                available_columns = set(self.events_df.columns)
            else:
                available_columns = set()

            if "category_id" in available_columns or "sku" in available_columns:
                self._extractors["graph"] = GraphFeatureExtractor(self)

            # We retain the intent extractor because it also creates funnel and
            # cart-behavior features. For Retailrocket, search-specific output
            # correctly reports that no search events exist.
            if "event_type" in available_columns:
                self._extractors["intent"] = IntentFeatureExtractor(self)

            if "price_bucket" in available_columns:
                self._extractors["price"] = PriceFeatureExtractor(self)

            if "is_available" in available_columns:
                self._extractors["availability"] = AvailabilityFeatureExtractor(self)

            if self.product_popularity is not None:
                self._extractors["social"] = SocialFeatureExtractor(self)

            if (
                self.product_popularity is not None
                and self.category_popularity is not None
            ):
                self._extractors["retailrocket_global_popularity"] = (
                    RetailrocketGlobalPopularityFeatureExtractor(self)
                )

            self.logger.info(
                f"Initialized extractors: {list(self._extractors.keys())}"
            )

            return self._extractors


    def get_client_events(self, client_id: int) -> pl.DataFrame:
        """
        Return all loaded Retailrocket history events for one client.

        The preferred path reads from the lazy Retailrocket event pipeline.
        In debug mode, a previously materialized test subset may be used.
        """
        if self.lazy_all is not None:
            return self._collect_client_events(client_id)

        if self.events_df is not None:
            return self.events_df.filter(
                pl.col("client_id") == client_id
            )

        raise RuntimeError(
            "No Retailrocket event pipeline is available. "
            "Call load_data() before generating representations."
        )



    def get_client_segment(self, client_id: int) -> dict:
        segments = {}
        for segment_name, users in self.user_segments.items():
            if isinstance(users, (list, set)) and client_id in users: segments[segment_name] = True
        return segments


    # --- Multi-Resolution History Helpers ---
    def _format_event_for_history(self, event_row: Dict[str, Any]) -> str:
        event_type = event_row.get("event_type"); sku = event_row.get("sku"); url = event_row.get("url")
        query = event_row.get("query"); ts = event_row.get("timestamp")
        ts_str = ts.strftime('%Y%m%d-%H%M') if isinstance(ts, datetime) else "NT" # Format plus court
        text_parts = [f"[{ts_str}] E:{event_type or '?'}"]
        try:
            if sku is not None:
                sku_int = int(sku); text_parts.append(f" S:{sku_int}")
                props = self.sku_properties_dict.get(sku_int, {})
                if props.get("category") is not None:
                    text_parts.append(f" C:{props['category']}")

                is_available = event_row.get("is_available")
                if is_available is not None:
                    text_parts.append(
                        f" A:{'IN' if int(is_available) == 1 else 'OUT'}"
                    )

                if props.get("price") is not None:
                    text_parts.append(f" P:{props['price']}")
            elif url is not None: text_parts.append(f" U:{url}")
            elif query is not None: text_parts.append(f" Q:{hash(str(query))%10000:04d}") # Hash court pour Q
        except Exception: pass # Ignorer erreurs de formatage individuelles
        return "".join(text_parts)

    def _generate_detailed_events_text(self, client_events: pl.DataFrame, limit=30) -> str:
        if client_events.height == 0: return "No recent activity."
        recent_events_rows = client_events.sort("timestamp", descending=True).head(limit).to_dicts()
        event_texts = [self._format_event_for_history(row) for row in recent_events_rows]
        return "\n".join(filter(None, event_texts))

    def _generate_summarized_events_text(self, client_events: pl.DataFrame, limit=10) -> str:
        if client_events.height == 0: return "No medium-term activity."
        summary = [f"Event count: {client_events.height}"]
        event_counts = client_events.group_by('event_type').agg(pl.count().alias('count')).sort('count', descending=True)
        summary.append("Event Types: " + ", ".join([f"{row['event_type']}:{row['count']}" for row in event_counts.iter_rows(named=True)]))
        if 'category_id' in client_events.columns:
            purchases = client_events.filter((pl.col('event_type') == pl.lit('product_buy', dtype=pl.Categorical)) & pl.col('category_id').is_not_null())
            if purchases.height > 0:
                category_counts = purchases.group_by('category_id').agg(pl.count().alias('count')).sort('count', descending=True)
                top_cats = category_counts.head(limit).to_dicts()
                summary.append("Top Purchased Cats: " + ", ".join([f"[CAT_{c['category_id']}]:{c['count']}" for c in top_cats]))
        return "\n".join(summary)

    def _generate_aggregated_events_text(self, client_events: pl.DataFrame) -> str:
        if client_events.height == 0: return "No historical activity."
        summary = []
        if client_events.height >= 2:
            first_ts, last_ts = client_events['timestamp'].min(), client_events['timestamp'].max()
            if first_ts and last_ts: tenure_days = (last_ts - first_ts).days; summary.append(f"Hist. Span: ~{tenure_days}d (end {last_ts.date()})")
        event_counts = client_events.group_by('event_type').agg(pl.count().alias('count')).sort('count', descending=True)
        summary.append("Hist. Event Counts: " + ", ".join([f"{r['event_type']}:{r['count']}" for r in event_counts.iter_rows(named=True)]))
        if 'category_id' in client_events.columns:
             purchases = client_events.filter((pl.col('event_type') == pl.lit('product_buy', dtype=pl.Categorical)) & pl.col('category_id').is_not_null())
             if purchases.height > 0:
                 cat_counts = purchases.group_by('category_id').agg(pl.count().alias('count')).sort('count', descending=True)
                 summary.append("Top Hist. Purchased Cats: " + ", ".join([f"[CAT_{c['category_id']}]:{c['count']}" for c in cat_counts.head(3).to_dicts()]))
        return "\n".join(summary)
    
  
  # --- Raw sequence formatting ---
    def _format_raw_event(self, event_row: Dict[str, Any]) -> str:
        parts = [f"EVENT: {event_row['event_type']}"]
        sku = event_row.get('sku')
        sku_int: Optional[int] = int(sku) if sku is not None else None   

        props = self.sku_properties_dict.get(int(sku), {}) if sku is not None else {}
        etype = event_row['event_type']
        if etype in ('page_visit', 'add_to_cart', 'product_buy') and sku is not None:
            parts.append(f"SKU:[SKU_{int(sku)}]")

            cat = event_row.get('category_id')
            if cat is None:
                cat = props.get('category')
            if cat is not None:
                parts.append(f"CAT:[CAT_{int(cat)}]")

            is_available = event_row.get("is_available")
            if is_available is not None:
                availability_token = (
                    "IN_STOCK" if int(is_available) == 1 else "OUT_OF_STOCK"
                )
                parts.append(f"AVAIL:[{availability_token}]")

        if hasattr(self, 'pop_score_by_sku') and self.pop_score_by_sku:
            score = self.pop_score_by_sku.get(sku_int)
            q_tag = pop_bin(score)
            parts.append(f"POP_Q:[{q_tag}]") 
        # --- NEW: how-many-days-ago bucket (coarse) ------------------
        if ts := event_row.get('timestamp'):
            if isinstance(ts, datetime):
                reference_time = self.reference_time or datetime.now()
                days_ago = (reference_time - ts).days
                if   days_ago <= 1:      parts.append("AGE:[D_0-1]")
                elif days_ago <= 7:      parts.append("AGE:[D_1-7]")
                elif days_ago <= 30:     parts.append("AGE:[D_7-30]")
                elif days_ago <= 180:    parts.append("AGE:[D_30-180]")
                else:                    parts.append("AGE:[D_180+]")            
        return " ".join(parts)

    def _generate_raw_sequence(self, client_events: pl.DataFrame, max_events: int = MAX_HISTORY_EVENTS_TO_CONSIDER) -> str:
        rows = client_events.sort('timestamp', descending=True).head(max_events).to_dicts()
        rows = list(reversed(rows))  # ordre chronologique

        seq_tokens = []
        prev_ts: Optional[datetime] = None
        SESSION_GAP_MIN = 30
        open_session = False

        for i, row in enumerate(rows):
            ts = row['timestamp']
            py_ts = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts

            # --- Détection nouveau bloc session ---
            new_session = False
            if prev_ts is None:
                new_session = True
            else:
                delta_min = (py_ts - prev_ts).total_seconds() / 60
                if delta_min > SESSION_GAP_MIN:
                    # fermer précédente
                    if open_session:
                        seq_tokens.append("<SESS_END>")
                    new_session = True
            if new_session:
                seq_tokens.append("<SESS_START>")
                open_session = True

            # --- Encodage standard de l’événement ---
            txt = self._format_raw_event(row)
            tod_tok = f"TOD:[{discretize_time_of_day(py_ts)}]"
            dow_tok = f"DOW:[{discretize_day_of_week(py_ts)}]"
            if prev_ts is None or new_session:
                delta_tok = f"TIME_DELTA:[{SESSION_START_TOKEN}]"
            else:
                delta = py_ts - prev_ts
                delta_tok = f"TIME_DELTA:[{discretize_timedelta(delta)}]"
            prev_ts = py_ts
            seq_tokens.append(" ".join([txt, tod_tok, dow_tok, delta_tok]))

        if open_session:
            seq_tokens.append("<SESS_END>")

        return SEP_TOKEN.join(seq_tokens)    
    
    def _compute_compact_metrics(self, events: pl.DataFrame) -> list[str]:
        """
        Renvoie des tags compacts (≤10 tokens chacun) :
          NAME_STD, Δ$, BURST, CAT_PR_TOP, H_cat, H_price
        S'adapte aux schémas avec event_type / price_bucket / emb_str.
        """
        tags = []
        
        # ---------- helpers internes -------------------------------------------
        evt_col   = "event_type" if "event_type" in events.columns else "event"
        price_col = "price" if "price" in events.columns else "price_bucket"
        
        def _bucket_to_num(s: pl.Series) -> np.ndarray:
            """Convertit price_bucket en valeurs numériques, en gérant les nulls"""
            if s.dtype == pl.Int64 or s.dtype == pl.Float64:
                return s.to_numpy()
            
            # Filtrer les nulls avant d'appliquer str.replace
            if s.null_count() > 0:
                # Option 1: Remplacer les nulls par une valeur par défaut
                s = s.fill_null("0")
            
            # Maintenant on peut appliquer str.replace en toute sécurité
            return s.str.replace(r"[^0-9]", "").cast(pl.Int32, strict=False).fill_null(0).to_numpy()
        
        # ---------- NAME_STD ----------------------------------------------------
        emb = None
        if "name_embedding" in events.columns:
            emb = np.vstack([vec for vec in events["name_embedding"].to_list() if vec is not None])
        elif "emb_str" in events.columns:
            str_vecs = [
                v for v in events["emb_str"].to_list()
                if v is not None and isinstance(v, (str, bytes)) and v.strip()
            ]
            if str_vecs:
                parsed_vecs = []
                for v in str_vecs:
                    try:
                        clean_v = v.strip()
                        if clean_v.startswith('[') and clean_v.endswith(']'):
                            clean_v = clean_v[1:-1]
                        vec = np.fromstring(clean_v, dtype=np.float32, sep=" ")
                        if vec.size > 0 and np.all(np.isfinite(vec)):
                            parsed_vecs.append(vec)
                    except Exception:
                        continue
                
                if parsed_vecs:
                    try:
                        emb = np.vstack(parsed_vecs)
                    except Exception:
                        emb = None
        
        if emb is not None and emb.size > 0:
            try:
                if emb.dtype.kind in ['f', 'i', 'u']:
                    name_std = round(float(np.std(emb)), 2)
                    tags.append(f"NAME_STD:{name_std}")
            except Exception:
                pass
        
        # ---------- Δ Panier / Prix ---------------------------------------------
        if price_col in events.columns:
            buy_mask   = pl.col(evt_col) == "product_buy"
            cart_mask  = pl.col(evt_col) == "add_to_cart"
            
            # Filtrer les événements avec prix non-null
            buy_events = events.filter(buy_mask & pl.col(price_col).is_not_null())
            cart_events = events.filter(cart_mask & pl.col(price_col).is_not_null())
            
            if buy_events.height > 0 and cart_events.height > 0:
                buy_vals = _bucket_to_num(buy_events[price_col])
                cart_vals = _bucket_to_num(cart_events[price_col])
                
                if buy_vals.size and cart_vals.size and cart_vals.mean() > 0:
                    delta_pct = 100 * (buy_vals.mean() - cart_vals.mean()) / cart_vals.mean()
                    tags.append(f"Δ$:{delta_pct:+.0f}%")
        
        # ---------- BURST score --------------------------------------------------
        if events.height and "timestamp" in events.columns:
            try:
                hour_counts = np.bincount(events["timestamp"].dt.hour().fill_null(0).to_numpy(), minlength=24)
                mu, var = hour_counts.mean(), hour_counts.var()
                if mu > 0:
                    tags.append(f"BURST:{round(var/mu,2)}")
            except Exception:
                pass
        
        # ---------- Graph centralité catégorie ----------------------------------
        if hasattr(self, "category_centrality") and "category_id" in events.columns:
            cats = [c for c in events["category_id"].drop_nulls().to_list()
                    if c in self.category_centrality]
            if cats:
                top_cat = max(cats, key=lambda c: self.category_centrality[c])
                score = round(self.category_centrality[top_cat], 2)
                tags.append(f"CAT_PR_TOP:{top_cat}({score})")
        
        # ---------- Entropies ----------------------------------------------------
        from collections import Counter
        
        # Helper pour calculer l'entropie Shannon
        def _shannon_entropy(counter: Counter) -> float:
            n = sum(counter.values())
            if n == 0:
                return 0.0
            from math import log2
            return -sum((c / n) * log2(c / n) for c in counter.values() if c > 0)
        
        # Entropie des catégories
        if "category_id" in events.columns:
            cat_list = events["category_id"].drop_nulls().to_list()
            if cat_list:
                cat_entropy = _shannon_entropy(Counter(cat_list))
                tags.append(f"H_cat:{round(cat_entropy,1)}")
        
        # Entropie des prix
        if price_col in events.columns:
            price_events = events.filter(pl.col(price_col).is_not_null())
            if price_events.height > 0:
                price_vals = _bucket_to_num(price_events[price_col])
                price_vals = price_vals[price_vals > 0]  # Filtrer les 0
                if price_vals.size > 0:
                    price_entropy = _shannon_entropy(Counter(price_vals))
                    tags.append(f"H_price:{round(price_entropy,1)}")
        
        return tags
        
    def _compute_extra_short_metrics(
        self,
        cid: int,
        events: pl.DataFrame,
        now: datetime
    ) -> list[str]:
        """
        Retourne une liste de tags ultra-compacts :
            - centralité SKU / Catégorie
            - entropie jour-semaine
            - variance horaire (circular)
            - rang de récence
            - durée de vie (span)
        Chaque tag est déjà « deduplicable » par son prefix.
        """
        tags: list[str] = []
    
        # ---------- 1) Centralité du SKU préféré ------------------------------
        buys = events.filter(pl.col("event_type") == "product_buy")
        if buys.height:
            sku_mode = buys["sku"].drop_nulls().mode()
            if not sku_mode.is_empty():
                fav_sku = int(sku_mode[0])        # on prend simplement le 1ᵉʳ mode
                if fav_sku in getattr(self, "sku_centrality", {}):
                    tags.append(f"CENT_SKU:{self.sku_centrality[fav_sku]:.2f}")
    
        # ---------- 2) Centralité de la Catégorie favorite --------------------
        if "category_id" in events.columns:
            cat_mode = events["category_id"].drop_nulls().mode()
            if not cat_mode.is_empty():
                fav_cat = int(cat_mode[0])        # idem : premier mode
                cat_centrality = getattr(self, "cat_centrality", {})
                if cat_centrality and fav_cat in cat_centrality:
                    tags.append(f"CENT_CAT:{fav_cat}({cat_centrality[fav_cat]:.2f})")
    
        # ---------- 3) Entropie des jours de semaine ----------------------------
        wd_series = events["timestamp"].dt.weekday()  # 0=Mon … 6=Sun
        if wd_series.len():
            wd_counts = np.bincount(wd_series.to_numpy(), minlength=7)
            h_dow = round(float(entropy(wd_counts, base=2)), 2)
            tags.append(f"DOW_ENT:{h_dow}")
    
        # ---------- 4) Variance horaire (circular) ------------------------------
        hr_series = events["timestamp"].dt.hour()
        if hr_series.len():
            angles = hr_series.to_numpy() / 24 * 2 * np.pi
            R = np.abs(np.mean(np.exp(1j * angles)))        # résultante
            tod_var = round(float(1 - R), 2)                # 0→mono-pic, 1→uniforme
            tags.append(f"TOD_VAR:{tod_var}")
    
        # ---------- 5) Rang de récence quantilé (0=très vieux, 1=très récent) ---
        last_ts = events["timestamp"].max()
        if last_ts is not None:
            days_since = (now - last_ts).days
            
            # Vérifier que rfm_recencies existe et est valide
            rfm_recencies = self.global_stats.get("rfm_recencies", [])
            if isinstance(rfm_recencies, np.ndarray) and rfm_recencies.size > 0:
                try:
                    # S'assurer que c'est un array numpy
                    if not isinstance(rfm_recencies, np.ndarray):
                        rfm_recencies = np.array(rfm_recencies, dtype=int)
                        
                    # Vérifier que l'array n'est pas vide et est 1D
                    if rfm_recencies.size > 0 and rfm_recencies.ndim == 1:
                        # S'assurer que l'array est trié
                        rfm_recencies = np.sort(rfm_recencies)
                        rec_q = np.searchsorted(rfm_recencies, days_since, side="right") / len(rfm_recencies)
                        tags.append(f"REC_RANK:{round(1 - rec_q, 2)}")
                    else:
                        self.logger.debug(f"rfm_recencies has invalid shape: {rfm_recencies.shape}")
                except Exception as e:
                    self.logger.debug(f"Error computing recency rank: {e}")
            else:
                self.logger.debug("No rfm_recencies data available")
    
        # ---------- 6) Durée de vie de l'historique -----------------------------
        first_ts = events["timestamp"].min()
        last_ts = events["timestamp"].max()
        if first_ts is not None and last_ts is not None:
            span_days = (last_ts - first_ts).days
            if span_days > 0:
                tags.append(f"LIFETIME:~{span_days}d")
    
        return tags        
    def _build_global_centralities(self) -> None:
        """
        Calcule une centralité « SKU PageRank » à partir des co-occurrences
        de SKU dans les mêmes sessions (30 min). 100 % Polars pour l'I/O,
        Counter + NumPy pour l'agrégation, NetworKit pour le PageRank sparse.
        """
        import numpy as np
        from datetime import timedelta
        from itertools import combinations
        from collections import Counter
    
        # ─── garde-fou ───────────────────────────────────────────────────────
        if self.lazy_all is None:
            self.sku_centrality = {}
            return
    
        # ─── 0)  Filtre SKU populaires ──────────────────────────────────────
        MIN_EVENTS, TOP_K = 50, 2_000  # Changé de MIN_VIEWS à MIN_EVENTS
        pop = self.product_popularity
    
        if pop is None:
            self.logger.warning("No product_popularity → disabling SKU centrality")
            allowed_skus = set()
        else:
            # Utiliser popularity_score qui est toujours présent
            # ou chercher n'importe quelle colonne de count disponible
            if "popularity_score" in pop.columns:
                # Utiliser le score de popularité global
                allowed_skus = set(
                    pop.filter(pl.col("popularity_score") > 0)
                       .sort("popularity_score", descending=True)
                       .head(TOP_K)["sku"]
                       .to_list()
                )
            elif "view_count" in pop.columns:
                # Si view_count existe, l'utiliser
                allowed_skus = set(
                    pop.filter(pl.col("view_count") >= MIN_EVENTS)
                       .sort("view_count", descending=True)
                       .head(TOP_K)["sku"]
                       .to_list()
                )
            elif "cart_count" in pop.columns:
                # Sinon utiliser cart_count
                allowed_skus = set(
                    pop.filter(pl.col("cart_count") >= MIN_EVENTS // 5)  # Seuil plus bas pour les paniers
                       .sort("cart_count", descending=True)
                       .head(TOP_K)["sku"]
                       .to_list()
                )
            elif "purchase_count" in pop.columns:
                # En dernier recours, utiliser purchase_count
                allowed_skus = set(
                    pop.filter(pl.col("purchase_count") >= MIN_EVENTS // 10)  # Seuil encore plus bas
                       .sort("purchase_count", descending=True)
                       .head(TOP_K)["sku"]
                       .to_list()
                )
            else:
                self.logger.warning("No count columns found in product_popularity")
                allowed_skus = set()
                
        self.logger.info("SKU centrality filter → kept %d SKUs", len(allowed_skus))
    
        # Si on n'a pas assez de SKUs, essayer sans filtre
        if len(allowed_skus) < 100 and pop is not None:
            self.logger.info("Too few SKUs after filter, using top SKUs by any metric")
            # Prendre juste les TOP_K SKUs les plus populaires
            allowed_skus = set(
                pop.sort("popularity_score", descending=True)
                   .head(min(TOP_K, pop.height))["sku"]
                   .to_list()
            )
            self.logger.info("Using %d SKUs without strict filtering", len(allowed_skus))
    
        # ─── 1)  Collect (client_id, ts, sku) trié ───────────────────────────
        if not allowed_skus:
            self.sku_centrality = {}
            self.logger.warning("No SKUs to process – centrality skipped")
            return
            
        df_sku = (
            self.lazy_all
                .filter(pl.col("sku").is_in(allowed_skus))
                .select(["client_id", "timestamp", "sku"])
                .sort(["client_id", "timestamp"])
                .collect(engine="streaming")
        )
        if df_sku.is_empty():
            self.sku_centrality = {}
            self.logger.warning("No SKU events after filter – centrality skipped")
            return
    
        # ─── 2)  Session IDs (gap 30 min) ────────────────────────────────────
        # Méthode simple sans window expressions
        df_sku = df_sku.sort(["client_id", "timestamp"])
        
        # Ajouter colonnes décalées
        df_sku = df_sku.with_columns([
            pl.col("client_id").shift(1).alias("prev_client_id"),
            pl.col("timestamp").shift(1).alias("prev_timestamp")
        ])
        
        # Calculer nouvelle session
        df_sku = df_sku.with_columns([
            pl.when(
                (pl.col("client_id") != pl.col("prev_client_id")) |
                ((pl.col("timestamp") - pl.col("prev_timestamp")).dt.total_seconds() / 60 > 30) |
                pl.col("prev_client_id").is_null()
            ).then(1)
            .otherwise(0)
            .alias("new_session")
        ])
        
        # ID de session cumulatif
        df_sku = df_sku.with_columns([
            pl.col("new_session").cum_sum().alias("sid")
        ])
        
        # Nettoyer
        df_sku = df_sku.drop(["prev_client_id", "prev_timestamp", "new_session"])
    
        # ─── 3)  Compte des paires SKU par session (Counter) ─────────────────
        pair_cnt: Counter[tuple[int, int]] = Counter()
        for (_cid, sid), sess in df_sku.group_by(["client_id", "sid"]):
            skus = sess["sku"].drop_nulls().unique().to_list()
            for i, j in combinations(sorted(skus), 2):
                pair_cnt[(int(i), int(j))] += 1
    
        if not pair_cnt:
            self.sku_centrality = {}
            self.logger.warning("No co-occurrences – centrality skipped")
            return
    
        edges   = np.array(list(pair_cnt.keys()), dtype=int)
        weights = np.array(list(pair_cnt.values()), dtype=float)
        src, dst = edges[:, 0], edges[:, 1]
    
        # ─── 4)  PageRank sparse avec helper NetworKit ───────────────────────
        self.sku_centrality = compute_sparse_pagerank(src, dst, weights)
    
        self.logger.info(
            "Built sparse SKU centrality for %d SKUs", len(self.sku_centrality)
        )
        # --- Main Generation Method ---
    def generate_representations(
        self, 
        client_ids: list, 
        max_length: int = MAX_RICH_TOKENS
    ) -> dict:
        """Return {client_id: json‑string} with a rich‑text block and *all* features ‑
        without duplicate lines.  Uses _build_rich_text() for final formatting so that
        sections follow SECTIONS_ORDER and long lists are automatically trimmed.
        """
        
        # ✅ FIX : Normaliser client_ids une fois pour toutes
        if isinstance(client_ids, int):
            client_ids = [client_ids]
        
        # ✅ FIX : Vérification UNIQUE et SIMPLE
        if self.events_df is None and self.lazy_all is None:
            self.logger.info("Loading data for clients...")
            self.load_data(use_cache=True, relevant_client_ids=client_ids)
            
            # Vérifier qu'on a maintenant des données
            if self.events_df is None and self.lazy_all is None:
                raise RuntimeError("No data available after load_data()")
        
        # ✅ Si on a events_df mais pas lazy_all, c'est OK (mode cache)
        if self.lazy_all is None and self.events_df is not None:
            self.logger.debug("Using events_df (cache mode)")
        
        # ========================================================
        # GARDEZ TOUT LE CODE ORIGINAL À PARTIR D'ICI !
        # ========================================================
        now = self.reference_time or datetime.now()
        extractors = self.get_feature_extractors()
        reps: dict[int, str] = {}
        
        # LE CODE CONTINUE ICI - NE PAS SUPPRIMER !
        for cid in client_ids:
            events = self.get_client_events(cid)
            
            if events.height == 0:
                reps[cid] = json.dumps(
                    {"profile": {"client_id": cid, "error": "No activity"}},
                    ensure_ascii=False,
                )
                continue
    
            # ==============================================================
            # 1)  OVERVIEW -------------------------------------------------
            # ==============================================================
            seg          = self.get_client_segment(cid)
            user_type    = "buyer" if seg.get("buyers") else "browser"
            overview_sec = [f"[CLIENT_{cid}]", f"User Type: {user_type}"]
            seg_list     = sorted(
                s for s, active in seg.items() if active and s not in {"buyers", "non_buyers"}
            )
            if seg_list:
                overview_sec.append("Segments: " + ", ".join(seg_list))

            # ==============================================================
            # 2)  FEATURE EXTRACTION  -> section_map ----------------------
            # ==============================================================
            section_map: dict[str, list[str]] = defaultdict(list)
            section_map["OVERVIEW"].extend(overview_sec)
            features_json: list[dict[str, str]] = []
            features_list = []

            # Additional compact behavioral metrics for the textual profile.
            extra_tags = self._compute_extra_short_metrics(cid, events, now)
            compact_tags = self._compute_compact_metrics(events)
            # where to dump each extractor's lines → logical section name
            ex_to_sec = {
                "temporal": "TEMPORAL",
                "sequence": "SEQUENCE",
                "social": "SOCIAL",
                "retailrocket_global_popularity": "GLOBAL_POPULARITY",
                "price": "PRICE",
                "availability": "AVAILABILITY",
                "intent": "OVERVIEW",
                "graph": "CUSTOM",
            }
            for ex_name, extractor in extractors.items():
                default_sec = ex_to_sec.get(ex_name, "CUSTOM")

                try:
                    feats = extractor.extract_features(cid, events, now)
                except Exception as err:
                    self.logger.error(
                        f"Feature extraction failed for {ex_name}, client {cid}: {err}"
                    )
                    feats = [f"{ex_name}-error"]

                repeat = (
                    IMPLICIT_WEIGHT_REPEAT
                    if ex_name in ("top_sku", "top_category")
                    else 1
                )

                for ft in feats:
                    target_sec = default_sec

                    # The churn_propensity extractor returns several logical
                    # feature groups, so route them by their prefix.
                    if ex_name == "churn_propensity":
                        if ft.startswith((
                            "CHURN_",
                            "PURCHASE_RECENCY:",
                            "PURCHASE_PATTERN:",
                            "AVG_PURCHASE_INTERVAL:",
                            "POST_PURCHASE",
                            "LTV_INDICATOR:",
                        )):
                            target_sec = "CHURN_PROPENSITY"

                        elif ft.startswith((
                            "CAT_PROPENSITY:",
                            "CAT_EXPLORATION_BREADTH:",
                            "PURCHASE_CAT_FOCUS:",
                        )):
                            target_sec = "CAT_PROPENSITY"

                        elif ft.startswith((
                            "SKU_PROPENSITY",
                            "REPEAT_PURCHASE_SKUS:",
                            "TOP_REPEAT_SKU:",
                        )):
                            target_sec = "SKU_PROPENSITY"

                    for _ in range(repeat):
                        section_map[target_sec].append(ft)

                    features_json.append({
                        "type": ex_name,
                        "value": ft,
                    })
            
            for t in extra_tags:
                features_list.append({"type": "extra", "value": t})
                section_map["CUSTOM"].append(t)
                
            for tag in compact_tags:
                features_list.append({"type": "compact", "value": tag})
                section_map["CUSTOM"].append(tag)          # ou une section dédiée "STATS"
            # ==============================================================
            # 3)  BEHAVIORAL METRICS --------------------------------------
            # ==============================================================

            co_pairs   = top_co_pairs(events)
            cat_pairs  = top_co_categories(events)
            cart_conv  = cart_conversion_stats(events)

            # --- GLOBAL cross-user pairs – filtrés sur l'historique utilisateur ----
            if 'global_sku_pairs' in self.global_stats:
                user_skus = events.filter(pl.col('sku').is_not_null())['sku'].unique().to_list()
                global_pairs = self.global_stats['global_sku_pairs']
                
                # Gérer le cas où c'est un dict ou une liste
                if isinstance(global_pairs, dict):
                    # Si c'est un dict, convertir en liste de tuples
                    pairs_list = []
                    for key, count in global_pairs.items():
                        if isinstance(key, str) and ',' in key:
                            # Format "(i, j)": count
                            try:
                                i, j = eval(key)  # Attention: eval est dangereux, mais ici on contrôle le format
                                pairs_list.append(((i, j), count))
                            except:
                                pass
                    global_pairs = pairs_list
                
                # Maintenant traiter comme une liste
                for pair_data in global_pairs[:20]:
                    if isinstance(pair_data, (list, tuple)) and len(pair_data) == 2:
                        (i, j), cnt = pair_data
                        if i in user_skus and j in user_skus:
                            section_map['CUSTOM'].append(f"GLOBAL_CO_PAIR:SKU_{i}~SKU_{j} ({cnt}x)")
            
            if 'global_cat_pairs' in self.global_stats and 'category_id' in events.columns:
                user_cats = events['category_id'].unique().drop_nulls().to_list()
                global_cat_pairs = self.global_stats['global_cat_pairs']
                
                # Même traitement pour les cat_pairs
                if isinstance(global_cat_pairs, dict):
                    pairs_list = []
                    for key, count in global_cat_pairs.items():
                        if isinstance(key, str) and ',' in key:
                            try:
                                c1, c2 = eval(key)
                                pairs_list.append(((c1, c2), count))
                            except:
                                pass
                    global_cat_pairs = pairs_list
                
                for pair_data in global_cat_pairs[:20]:
                    if isinstance(pair_data, (list, tuple)) and len(pair_data) == 2:
                        (c1, c2), cnt = pair_data
                        if c1 in user_cats and c2 in user_cats:
                            section_map['CUSTOM'].append(f"GLOBAL_CAT_PAIR:CAT_{c1}~CAT_{c2} ({cnt}x)")

                
                
            # ==============================================================
            # 4)  HISTORY SNAPSHOTS ---------------------------------------
            # ==============================================================
            recent_cut  = now - timedelta(days=14)
            medium_cut  = now - timedelta(days=90)

            recent_txt = self._generate_detailed_events_text(
                events.filter(pl.col("timestamp") >= recent_cut)
            )
            medium_txt = self._generate_summarized_events_text(
                events.filter((pl.col("timestamp") >= medium_cut) & (pl.col("timestamp") < recent_cut))
            )
            hist_txt   = self._generate_aggregated_events_text(
                events.filter(pl.col("timestamp") < medium_cut)
            )

            if recent_txt != "No recent activity.":
                section_map["RECENT_HISTORY_14D"].append(recent_txt)
            if medium_txt != "No medium-term activity.":
                section_map["SEQUENCE"].append(medium_txt)
            if hist_txt != "No historical activity.":
                section_map["CUSTOM"].append(hist_txt)

            # ==============================================================
            # 5)  RAW SEQUENCE  -------------------------------------------
            # ==============================================================
            # Keep the raw chronological event sequence separate from the
            # shuffled feature sections. It is appended once after the rich
            # profile text has been built.
            raw_seq = self._generate_raw_sequence(
                events,
                max_events=RAW_SEQUENCE_LAST_EVENTS,
            )

            section_map["CUSTOM"] = list(dict.fromkeys(section_map["CUSTOM"]))

            # ==============================================================
            # 6)  BUILD RICH TEXT (deduplicated / token‑limited) -----------
            # ==============================================================
            try:
                rich_text = _build_rich_text(
                    section_map=section_map,
                    max_tokens=max_length,
                    implicit_repeat=IMPLICIT_WEIGHT_REPEAT,
                    top_per_section=TOP_FEATURES_PER_SECTION,
                    shuffle_seed=cid,
                )
            except Exception as exc:
                self.logger.error(f"_build_rich_text failed for {cid}: {exc}")
                # fallback – very plain
                rich_text = "\n\n".join(
                    f"## {sec} ##\n" + "\n".join(lines)
                    for sec, lines in section_map.items()
                )

            # ==============================================================
            # 7)  JSON PROFILE (for downstream) ---------------------------
            # ==============================================================
            profile = {
                "client_id": cid,
                "overview": {"user_type": user_type, "segments": seg_list},
                "features": features_json,
                "behavioral_metrics": {
                    "co_pairs": co_pairs,
                    "category_pairs": cat_pairs,
                    "cart_conversion": cart_conv,
                },
                "recent_activity": recent_txt,
                "medium_term_summary": medium_txt,
                "historical_aggregates": hist_txt,
                "raw_sequence": raw_seq,
                "insights": {},
                "recommendations": [],
            }

            if any(
                ft["type"] == "temporal" and "Inactive" in ft["value"]
                for ft in features_json
            ):
                profile["insights"]["inactivity"] = "High inactivity"
                profile["recommendations"].append("Send re‑engagement email")

            if raw_seq and "[END]" in rich_text:
                rich_text = rich_text.replace("[END]", f"\n## RAW_SEQUENCE ##\n{raw_seq}\n[END]")
            
            reps[cid] = json.dumps(
                {"profile": profile, "rich_text": rich_text}, ensure_ascii=False
            )

        return reps




class ChurnPropensityFeatureExtractor(FeatureExtractorBase):
    """Extract features specifically for churn and propensity prediction tasks"""
    
    def extract_features(self, client_id: int, events: pl.DataFrame, now: datetime) -> List[str]:
        features = []
        
        # Pass client_id to all methods for better error tracking
        self._extract_churn_signals(client_id, events, features, now)
        self._extract_category_propensity(client_id, events, features, now)
        self._extract_sku_propensity(client_id, events, features, now)
        
        return features
    
    def _extract_churn_signals(self, client_id: int, events: pl.DataFrame, features: List[str], now: datetime) -> None:
        """Extract signals relevant to churn prediction"""
        try:
            # Get purchase history with proper categorical comparison
            purchases = events.filter(
                pl.col('event_type') == pl.lit('product_buy', dtype=pl.Categorical)
            )
            
            if purchases.height == 0:
                features.append("CHURN_STATUS:NO_PURCHASE")
                return
            
            # Get timestamps as Series
            purchase_timestamps = purchases['timestamp']
            last_purchase = purchase_timestamps.max()
            first_purchase = purchase_timestamps.min()
            
            if last_purchase is None:
                features.append("CHURN_STATUS:NO_VALID_PURCHASE_TIME")
                return
                
            days_since_last_purchase = (now - last_purchase).days
            
            # Official churn definition: 14+ days after purchase
            if days_since_last_purchase >= 14:
                features.append("CHURN_RISK:HIGH")
            else:
                features.append("CHURN_RISK:LOW")
            
            features.append(f"PURCHASE_RECENCY:{days_since_last_purchase}d")
            
            # Purchase frequency pattern
            if purchases.height > 1:
                purchase_intervals = []
                # Get sorted timestamps, filter nulls
                sorted_purchases = purchases.sort('timestamp')
                purchase_times = sorted_purchases['timestamp'].drop_nulls().to_list()
                
                for i in range(1, len(purchase_times)):
                    interval_days = (purchase_times[i] - purchase_times[i-1]).days
                    purchase_intervals.append(interval_days)
                
                if purchase_intervals:
                    avg_interval = np.mean(purchase_intervals)
                    features.append(f"AVG_PURCHASE_INTERVAL:{avg_interval:.1f}d")
                    
                    if days_since_last_purchase > avg_interval * 2:
                        features.append("PURCHASE_PATTERN:UNUSUAL_GAP")
            
            # Activity after last purchase
            post_purchase_activity = events.filter(pl.col('timestamp') > last_purchase)
            
            if post_purchase_activity.height > 0:
                features.append(f"POST_PURCHASE_EVENTS:{post_purchase_activity.height}")
                
                # Type of post-purchase activity with safe access
                post_types = post_purchase_activity.group_by('event_type').agg(
                    pl.count().alias('count')
                )
                
                # Create a dict for easier access
                post_dict = {row['event_type']: row['count'] for row in post_types.iter_rows(named=True)}
                
                if post_dict.get('page_visit', 0) > 0:
                    features.append("POST_PURCHASE:BROWSING")
                if post_dict.get('add_to_cart', 0) > 0:
                    features.append("POST_PURCHASE:CART_ACTIVITY")
            else:
                features.append("POST_PURCHASE:NO_ACTIVITY")
            
            # Lifetime value indicators - FIXED: Keep as DataFrame or use .len() for Series
            if 'price_bucket' in purchases.columns:
                # Option 1: Keep as DataFrame
                price_df = purchases.filter(pl.col('price_bucket').is_not_null()).select('price_bucket')
                if price_df.height > 0:
                    total_purchase_value = price_df['price_bucket'].sum()
                    features.append(f"LTV_INDICATOR:{total_purchase_value}")
                else:
                    features.append("LTV_INDICATOR:0")
            else:
                features.append("LTV_INDICATOR:NO_PRICE_DATA")
            
        except Exception as e:
            self.logger.error(f"Error in churn signals for client {client_id}: {e}", exc_info=True)
            features.append("CHURN_SIGNALS:ERROR")
    
    def _extract_category_propensity(self, client_id: int, events: pl.DataFrame, features: List[str], now: datetime) -> None:
        """Extract signals for category propensity prediction"""
        try:
            if 'category_id' not in events.columns:
                features.append("CAT_PROPENSITY:NO_CATEGORY_DATA")
                return
            
            # Category interaction patterns
            cat_events = events.filter(pl.col('category_id').is_not_null())
            if cat_events.height == 0:
                features.append("CAT_PROPENSITY:NO_CATEGORY_EVENTS")
                return
            
            # Recency-weighted category interest
            # Use proper datetime handling
            cat_events_with_weight = cat_events.with_columns(
                (
                    ((pl.lit(now) - pl.col("timestamp")).dt.total_seconds() / 86400 + 1.0)
                    .pow(-0.5)
                    .alias("recency_weight")
                )
            )
            
            # Top categories by weighted interaction
            cat_scores = (
                cat_events_with_weight
                .group_by('category_id')
                .agg([
                    pl.sum('recency_weight').alias('weighted_score'),
                    pl.count().alias('interaction_count'),
                    pl.max('timestamp').alias('last_interaction')
                ])
                .sort('weighted_score', descending=True)
            )
            
            # Add features for top categories
            top_cats = cat_scores.head(5)
            for row in top_cats.to_dicts():
                cat_id = row['category_id']
                score = row['weighted_score']
                features.append(f"CAT_PROPENSITY:CAT_{cat_id}(score={score:.2f})")
            
            # Category diversity for propensity
            unique_cats = cat_events['category_id'].n_unique()
            features.append(f"CAT_EXPLORATION_BREADTH:{unique_cats}")
            
            # Purchase concentration
            purchase_cats = events.filter(
                (pl.col('event_type') == pl.lit('product_buy', dtype=pl.Categorical)) & 
                pl.col('category_id').is_not_null()
            )
            
            if purchase_cats.height > 0:
                purchase_cat_dist = purchase_cats.group_by('category_id').agg(
                    pl.count().alias('count')
                ).sort('count', descending=True)
                
                if purchase_cat_dist.height > 0:
                    top_row = purchase_cat_dist.row(0, named=True)
                    top_purchase_cat = top_row['category_id']
                    purchase_concentration = top_row['count'] / purchase_cats.height
                    features.append(f"PURCHASE_CAT_FOCUS:CAT_{top_purchase_cat}({purchase_concentration:.2f})")
            
        except Exception as e:
            self.logger.error(f"Error in category propensity for client {client_id}: {e}", exc_info=True)
            features.append("CAT_PROPENSITY:ERROR")
    
    def _extract_sku_propensity(self, client_id: int, events: pl.DataFrame, features: List[str], now: datetime) -> None:
        """Extract signals for SKU propensity prediction"""
        try:
            if 'sku' not in events.columns:
                features.append("SKU_PROPENSITY:NO_SKU_DATA")
                return
            
            sku_events = events.filter(pl.col('sku').is_not_null())
            if sku_events.height == 0:
                features.append("SKU_PROPENSITY:NO_SKU_EVENTS")
                return
            
            # SKU interaction patterns with recency weighting
            sku_events_weighted = sku_events.with_columns(
                (
                    ((pl.lit(now) - pl.col("timestamp")).dt.total_seconds() / 86400 + 1.0)
                    .pow(-0.5)
                    .alias("recency_weight")
                )
            )
            
            # Event type weights - using when/then for categorical column
            sku_events_weighted = sku_events_weighted.with_columns(
                pl.when(pl.col('event_type') == pl.lit('page_visit', dtype=pl.Categorical)).then(1.0)
                .when(pl.col('event_type') == pl.lit('add_to_cart', dtype=pl.Categorical)).then(3.0)
                .when(pl.col('event_type') == pl.lit('product_buy', dtype=pl.Categorical)).then(5.0)
                .when(pl.col('event_type') == pl.lit('remove_from_cart', dtype=pl.Categorical)).then(-2.0)
                .otherwise(1.0)
                .alias('event_weight')
            )
            
            # Combined score
            sku_events_weighted = sku_events_weighted.with_columns(
                (pl.col('recency_weight') * pl.col('event_weight')).alias('combined_score')
            )
            
            # Top SKUs by combined score
            sku_scores = (
                sku_events_weighted
                .group_by('sku')
                .agg([
                    pl.sum('combined_score').alias('total_score'),
                    pl.count().alias('interaction_count'),
                    pl.max('timestamp').alias('last_interaction')
                ])
                .sort('total_score', descending=True)
            )
            
            # Features for top SKUs
            top_skus = sku_scores.head(10)
            for i, row in enumerate(top_skus.to_dicts()):
                sku = row['sku']
                score = row['total_score']
                features.append(f"SKU_PROPENSITY_TOP{i+1}:SKU_{sku}(score={score:.2f})")
            
            # Re-interaction patterns
            sku_repeat_purchases = (
                events.filter(pl.col('event_type') == pl.lit('product_buy', dtype=pl.Categorical))
                .filter(pl.col('sku').is_not_null())
                .group_by('sku')
                .agg(pl.count().alias('purchase_count'))
                .filter(pl.col('purchase_count') > 1)
            )
            
            if sku_repeat_purchases.height > 0:
                features.append(f"REPEAT_PURCHASE_SKUS:{sku_repeat_purchases.height}")
                top_repeat = sku_repeat_purchases.sort('purchase_count', descending=True).head(1)
                if top_repeat.height > 0:
                    top_row = top_repeat.row(0, named=True)
                    features.append(f"TOP_REPEAT_SKU:SKU_{top_row['sku']}({top_row['purchase_count']}x)")
            
            # Brand loyalty (if available through properties)
            if hasattr(self.parent, 'sku_properties_dict') and self.parent.sku_properties_dict:
                sku_list = sku_events['sku'].unique().to_list()
                brands = []
                for sku in sku_list:
                    if sku is not None:
                        props = self.parent.sku_properties_dict.get(int(sku), {})
                        if 'brand' in props:
                            brands.append(props['brand'])
                
                if brands:
                    brand_counts = Counter(brands)
                    if brand_counts:
                        top_brand = brand_counts.most_common(1)[0]
                        features.append(f"BRAND_AFFINITY:{top_brand[0]}({top_brand[1]})")
            
        except Exception as e:
            self.logger.error(f"Error in SKU propensity for client {client_id}: {e}", exc_info=True)
            features.append("SKU_PROPENSITY:ERROR")

            
# --- Wrapper Class ---
class TextRepresentationGenerator:
    """Pipeline = (load data ➜ rich_text ➜ portrait LLM ➜ save)."""

    def __init__(self, use_polars: bool = True) -> None:
        self.data_dir:    Optional[str]               = None
        self.cache_dir:   Optional[str]               = None
        self.debug_mode:  bool                        = False
        self.advanced_generator: Optional[AdvancedUBMGenerator] = None

    # ------------------------------------------------------------------ #
    #                         SET-UP & HELPERS                           #
    # ------------------------------------------------------------------ #
    def prepare_data(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        debug_mode: bool = False,
    ) -> "TextRepresentationGenerator":
        self.data_dir   = data_dir
        self.cache_dir  = cache_dir
        self.debug_mode = debug_mode
        return self

    # ------------------------------------------------------------------ #
    #                             MAIN CALL                              #
    # ------------------------------------------------------------------ #
    def generate_text_representations(
        self,
        client_ids: List[int],
        output_file: Optional[str] = None,
        max_length: int = 3_500,
    ) -> Dict[int, str]:
        """Return {client_id: rich_text + PORTRAIT}"""

        if not self.data_dir:
            raise ValueError("Call prepare_data() first.")

        # 1) Load / cache data
        if self.advanced_generator is None:
            logger.info("Instantiating AdvancedUBMGenerator …")
            self.advanced_generator = AdvancedUBMGenerator(
                self.data_dir,
                cache_dir=self.cache_dir,
                debug_mode=self.debug_mode,
            )
            relevant_filter = client_ids if self.debug_mode else None
            self.advanced_generator.load_data(
                use_cache=True, relevant_client_ids=relevant_filter
            )

        # 2) Generate base rich-texts
        base_texts: Dict[int, str] = self.advanced_generator.generate_representations(
            client_ids, max_length=max_length
        )

        # 3) Strip out RAW_SEQUENCE for the LLM
        def _strip_raw(txt: str) -> str:
            return re.split(r'(?i)RAW_SEQUENCE:', txt, maxsplit=1)[0].rstrip()
        
        stripped_texts: Dict[int, str] = {}
        for cid, rt in base_texts.items():
            payload = json.loads(rt)
            summary = payload.get("rich_text", "")
            stripped = _strip_raw(summary)
            stripped_texts[cid] = stripped



        # 4) Ask the LLM for plain-text bullet portraits
        from .portrait_generator import generate_portraits

        portraits = generate_portraits(stripped_texts)

        # 5) Merge summary, portrait & RAW_SEQUENCE into one plain-text blob
        enriched_texts: Dict[int, str] = {}
        for cid, full_txt in base_texts.items():
            rep      = json.loads(full_txt)
            summary  = rep.get("rich_text", "")
            raw_seq  = rep.get("profile", {}).get("raw_sequence", "")

            # get the bullet list string from our LLM
            portrait_str   = portraits.get(cid, "")
            portrait_block = "\n## PORTRAIT ##\n" + portrait_str

            raw_block = ("\n## RAW_SEQUENCE ##\n" + raw_seq) if raw_seq else ""
            if summary.strip().endswith("```"):
                summary += "\n```"
            final_rich = "\n".join([
                summary,
                "## PORTRAIT ##",
                portrait_str.strip(),
                "## RAW_SEQUENCE ##",
                "\n".join(raw_seq.split("</s>")[-50:]) + "\n…",
            ])

            enriched_texts[cid] = final_rich

        # 6) Optionally write out…
        if output_file:
            self._write_to_file(enriched_texts, output_file)

        return enriched_texts

    # ------------------------------------------------------------------ #
    #                            I/O helper                              #
    # ------------------------------------------------------------------ #
    def _write_to_file(self, texts: Dict[int, str], path: str) -> None:
        logger.info(f"Saving {len(texts)} representations ➜ {path}")
        is_gcs = path.startswith("gs://")

        if is_gcs:
            try:
                import gcsfs
                fs = gcsfs.GCSFileSystem()
                f  = fs.open(path, "wt", encoding="utf-8")
            except ImportError:
                logger.error("gcsfs missing – cannot write to GCS.")
                return
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            f = open(path, "w", encoding="utf-8")

        with f:
            for cid, txt in texts.items():
                json.dump({"client_id": cid, "rich_text": txt}, f, ensure_ascii=False)
                f.write("\n")

        logger.info("✅  Representations saved.")
