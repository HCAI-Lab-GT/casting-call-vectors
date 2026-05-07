#!/usr/bin/env python3
"""
Persona Steering Evaluation Metrics (Hybrid spaCy + Regex)
===========================================================
Comprehensive NLP metrics for evaluating persona steering methodologies.
Supports single file and batch folder processing.

Uses spaCy where it adds value (lemmatization, sentence segmentation, POS tagging)
and regex where it's already optimal (fixed patterns, pronoun counting).

Requirements:
    pip install pandas numpy scipy spacy
    python -m spacy download en_core_web_sm  # or en_core_web_lg for better accuracy

Usage:
    # Single file
    python nlp_evaluator.py --input Comparison_GoldStandard_accountant.csv
    
    # Folder of roles
    python nlp_evaluator.py --input ./comparisons/ --output ./results/
    
    # With visualizations
    python nlp_evaluator.py --input ./comparisons/ --visualize
    
    # Force disable spaCy (use regex fallback)
    python nlp_evaluator.py --input ./comparisons/ --no-spacy

    # Parallel processing (folder mode)
    python nlp_evaluator.py --input ./comparisons/ --workers 4
"""

import pandas as pd
import numpy as np
import re
import os
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass, asdict, field
from scipy import stats
import concurrent.futures
from tqdm import tqdm
import warnings

from nlp_helpers import PersonaMetrics, MethodSummary, StatisticalComparison, Lexicons, Filters
from nlp_visualizer import generate_visualizations

warnings.filterwarnings('ignore')
 
# =============================================================================
# SPACY INITIALIZATION (OPTIONAL)
# =============================================================================

SPACY_AVAILABLE = False
nlp = None

def init_spacy(model_name: str = "en_core_web_sm", disable_components: List[str] = None):
    """Initialize spaCy with specified model. Returns True if successful."""
    global SPACY_AVAILABLE, nlp
    
    if disable_components is None:
        # Disable NER for speed - we only need tokenizer, lemmatizer, POS tagger, parser
        disable_components = ["ner"]
    
    try:
        import spacy
        nlp = spacy.load(model_name, disable=disable_components)
        # Increase max length for long degenerate responses
        nlp.max_length = 2_000_000
        SPACY_AVAILABLE = True
        print(f"spaCy loaded: {model_name}")
        return True
    except ImportError:
        print("spaCy not installed. Using regex fallback.")
        print("  Install with: pip install spacy && python -m spacy download en_core_web_sm")
        return False
    except OSError:
        print(f"spaCy model '{model_name}' not found. Using regex fallback.")
        print(f"  Install with: python -m spacy download {model_name}")
        return False


# =============================================================================
# BLEU & COMPRESSION RATIO
# =============================================================================

import zlib


def _compute_compression_ratio(text: str) -> float:
    """Compute compression ratio: compressed_size / original_size.
    Lower values indicate more repetitive/compressible text."""
    if not text or not text.strip():
        return 0.0
    encoded = text.encode('utf-8')
    compressed = zlib.compress(encoded)
    return len(compressed) / len(encoded)


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _compute_bleu_against_reference(texts: List[str], references: List[str], max_n: int = 4) -> List[float]:
    """Compute BLEU for each text against its corresponding reference.
    Higher BLEU = more similar to reference."""
    assert len(texts) == len(references), "texts and references must have the same length"

    results = []
    for text, ref in zip(texts, references):
        hyp_tokens = re.findall(r'\b\w+\b', text.lower())
        ref_tokens = re.findall(r'\b\w+\b', ref.lower())

        if len(hyp_tokens) < max_n or len(ref_tokens) < max_n:
            results.append(0.0)
            continue

        # Modified precision for each n-gram order
        precisions = []
        for n in range(1, max_n + 1):
            hyp_ngrams = Counter(_ngrams(hyp_tokens, n))
            ref_ngrams = Counter(_ngrams(ref_tokens, n))
            if not hyp_ngrams:
                precisions.append(0.0)
                continue
            clipped = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in hyp_ngrams.items())
            precisions.append(clipped / sum(hyp_ngrams.values()))

        # Geometric mean of precisions
        log_avg = 0.0
        valid = 0
        for p in precisions:
            if p > 0:
                log_avg += np.log(p)
                valid += 1
        if valid == 0:
            results.append(0.0)
        else:
            log_avg /= max_n  # uniform weights
            # Brevity penalty
            bp = min(1.0, np.exp(1 - len(ref_tokens) / len(hyp_tokens))) if len(hyp_tokens) > 0 else 0.0
            results.append(bp * np.exp(log_avg))

    return results


def _process_file_worker(args):
    """Process a single file in a worker process."""
    filepath, columns, use_spacy, filters = args
    return analyze_single_file(filepath, columns, use_spacy=use_spacy, filters=filters)


# =============================================================================
# EVALUATOR CLASS
# =============================================================================
 
class PersonaEvaluator:
    """Evaluator for persona steering quality - hybrid spaCy + regex"""
    
    def __init__(self, role: str = "default", use_spacy: bool = True):
        self.role = role
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        # self.domain_vocab_lemmas = Lexicons.get_domain_vocab_lemmas(role)
        # self.domain_vocab_surface = Lexicons.get_domain_vocab_surface(role)
    
    # -------------------------------------------------------------------------
    # Regex-based methods (always used)
    # -------------------------------------------------------------------------
    
    def _get_words_regex(self, text: str) -> List[str]:
        """Tokenize text into lowercase words using regex"""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _get_bigrams(self, words: List[str]) -> List[Tuple[str, str]]:
        """Get bigrams from word list"""
        return [(words[i], words[i+1]) for i in range(len(words)-1)]
    
    def _count_pattern(self, text: str, pattern: str) -> int:
        """Count occurrences of regex pattern"""
        return len(re.findall(pattern, text, re.IGNORECASE))
    
    def _get_sentences_regex(self, text: str) -> List[str]:
        """Split text into sentences using regex (fallback)"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _count_first_person(self, words: List[str]) -> int:
        """Count first-person pronouns (regex-based, optimal)"""
        return sum(1 for w in words if w in Lexicons.FIRST_PERSON)
    
    def _count_epistemic_markers(self, text: str) -> Dict[str, int]:
        """Count epistemic markers (regex-based, optimal)"""
        markers = {}
        for pattern, name in Lexicons.EPISTEMIC_MARKERS:
            count = self._count_pattern(text, pattern)
            if count > 0:
                markers[name] = count
        return markers
    
    def _count_ai_phrases(self, text: str) -> int:
        """Count AI identity leakage phrases (regex-based, optimal)"""
        return sum(self._count_pattern(text, p) for p in Lexicons.AI_PHRASES)
    
    def _compute_repetition(self, words: List[str]) -> Tuple[float, bool]:
        """Compute bigram uniqueness ratio (regex-based, optimal)"""
        bigrams = self._get_bigrams(words)
        if len(bigrams) == 0:
            return 1.0, False
        ratio = len(set(bigrams)) / len(bigrams)
        is_repetitive = ratio < 0.5
        return ratio, is_repetitive
    
    # -------------------------------------------------------------------------
    # spaCy-enhanced methods
    # -------------------------------------------------------------------------
    
    def _analyze_with_spacy(self, text: str) -> Dict:
        """Run spaCy analysis and extract enhanced metrics"""
        if not self.use_spacy or nlp is None:
            return None
        
        # Truncate very long texts to avoid memory issues
        max_chars = 100_000
        if len(text) > max_chars:
            text = text[:max_chars]
        
        doc = nlp(text)
        
        # Sentence segmentation (spaCy handles abbreviations better)
        sentences = list(doc.sents)
        sentence_count = len(sentences)
        
        # Lemmatized tokens
        lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
        
        # Domain vocabulary (lemmatized matching)
        # domain_lemma_count = sum(1 for lemma in lemmas if lemma in self.domain_vocab_lemmas)
        
        # Modal verbs (POS tag MD)
        modal_count = sum(1 for token in doc if token.tag_ == 'MD')
        
        # Passive voice detection (aux + past participle pattern)
        passive_count = 0
        for token in doc:
            if token.dep_ == 'auxpass' or token.dep_ == 'nsubjpass':
                passive_count += 1
        
        return {
            'sentence_count': sentence_count,
            'lemmas': lemmas,
            # 'domain_lemma_count': domain_lemma_count,
            'modal_count': modal_count,
            'passive_count': passive_count,
        }
    
    # -------------------------------------------------------------------------
    # Main compute method
    # -------------------------------------------------------------------------
    
    def compute_metrics(self, text: str, spacy_result: dict = None, bleu: float = 0.0) -> PersonaMetrics:
        """Compute all metrics for a single response (hybrid approach)"""
        
        if pd.isna(text) or not str(text).strip():
            return PersonaMetrics()
        
        text = str(text)
        
        # ----- Regex-based metrics (always used) -----
        words = self._get_words_regex(text)
        word_count = len(words)
        
        if word_count == 0:
            return PersonaMetrics()
        
        # First-person (regex - optimal)
        first_person_count = self._count_first_person(words)
        first_person_rate = first_person_count / word_count * 100
        
        # Epistemic markers (regex - optimal)
        epistemic_markers = self._count_epistemic_markers(text)
        epistemic_total = sum(epistemic_markers.values())
        
        # Repetition (regex - optimal)
        unique_bigram_ratio, is_repetitive = self._compute_repetition(words)
        is_degenerate = word_count > 500
        
        # AI phrases (regex - optimal)
        ai_phrase_count = self._count_ai_phrases(text)
        
        # Questions (regex - optimal)
        question_count = text.count('?')
        
        # Hedging and assertive (word list)
        hedge_count = sum(1 for w in words if w in Lexicons.HEDGE_WORDS)
        hedge_rate = hedge_count / word_count * 100
        assertive_count = sum(1 for w in words if w in Lexicons.ASSERTIVE_WORDS)
        assertive_rate = assertive_count / word_count * 100
        
        # Transitions
        transition_count = sum(1 for w in words if w in Lexicons.TRANSITIONS)
        
        # Domain vocabulary (surface form - regex fallback)
        # domain_vocab_count = sum(1 for w in words if w in self.domain_vocab_surface)
        # domain_vocab_rate = domain_vocab_count / word_count * 100
        
        # ----- spaCy-enhanced metrics (when available) -----
        if spacy_result is not None:
            spacy_results = spacy_result
        else:
            spacy_results = self._analyze_with_spacy(text)
        
        if spacy_results:
            # Use spaCy sentence segmentation
            sentence_count = spacy_results['sentence_count']
            avg_sentence_length = word_count / max(sentence_count, 1)
            
            # Lemmatized domain vocabulary (better recall)
            # domain_vocab_lemmatized_count = spacy_results['domain_lemma_count']
            # domain_vocab_lemmatized_rate = domain_vocab_lemmatized_count / word_count * 100
            
            # Modal verbs
            modal_verb_count = spacy_results['modal_count']
            modal_verb_rate = modal_verb_count / word_count * 100
            
            # Passive voice
            passive_voice_count = spacy_results['passive_count']
            passive_voice_rate = passive_voice_count / max(sentence_count, 1) * 100
            
            used_spacy = True
        else:
            # Regex fallback
            sentences = self._get_sentences_regex(text)
            sentence_count = len(sentences)
            avg_sentence_length = word_count / max(sentence_count, 1)
            
            # domain_vocab_lemmatized_count = domain_vocab_count
            # domain_vocab_lemmatized_rate = domain_vocab_rate
            
            # Modal verbs (word list fallback)
            modal_verb_count = sum(1 for w in words if w in Lexicons.MODAL_VERBS)
            modal_verb_rate = modal_verb_count / word_count * 100
            
            passive_voice_count = 0
            passive_voice_rate = 0.0
            
            used_spacy = False
        
        # Role consistency
        role_consistency_score = self._compute_role_consistency(words)
        
        return PersonaMetrics(
            first_person_rate=first_person_rate,
            epistemic_markers=epistemic_markers,
            epistemic_total=epistemic_total,
            word_count=word_count,
            unique_bigram_ratio=unique_bigram_ratio,
            is_repetitive=is_repetitive,
            is_degenerate=is_degenerate,
            hedge_rate=hedge_rate,
            assertive_rate=assertive_rate,
            question_rate=question_count / word_count * 100,
            transition_rate=transition_count / word_count * 100,
            avg_sentence_length=avg_sentence_length,
            sentence_count=sentence_count,
            modal_verb_rate=modal_verb_rate,
            passive_voice_rate=passive_voice_rate,
            # domain_vocab_count=domain_vocab_count,
            # domain_vocab_rate=domain_vocab_rate,
            # domain_vocab_lemmatized_count=domain_vocab_lemmatized_count,
            # domain_vocab_lemmatized_rate=domain_vocab_lemmatized_rate,
            ai_phrase_rate=ai_phrase_count / word_count * 100,
            role_consistency_score=role_consistency_score,
            bleu=bleu,
            compression_ratio=_compute_compression_ratio(text),
            used_spacy=used_spacy
        )
    
    def _compute_role_consistency(self, words: List[str]) -> float:
        """Compute consistency of first-person markers across response thirds"""
        if len(words) < 30:
            return 1.0
        
        third = len(words) // 3
        parts = [words[:third], words[third:2*third], words[2*third:]]
        
        densities = []
        for part in parts:
            if len(part) > 0:
                fp_count = sum(1 for w in part if w in Lexicons.FIRST_PERSON)
                densities.append(fp_count / len(part))
            else:
                densities.append(0)
        
        std = np.std(densities)
        if std == 0:
            return 1.0
        
        return max(0, 1 - std * 10)
 
 
# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
 
def extract_role_from_filename(filename: str) -> str:
    """Extract role name from filename like 'Comparison_GoldStandard_accountant.csv'"""
    name = Path(filename).stem
    parts = name.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[2:])
    return "default"
 
 
def compute_method_summary(metrics_list: List[PersonaMetrics]) -> MethodSummary:
    """Compute aggregated summary for a list of metrics"""
    if not metrics_list:
        return MethodSummary()
    
    n = len(metrics_list)
    
    return MethodSummary(
        n_responses=n,
        first_person_rate_mean=np.mean([m.first_person_rate for m in metrics_list]),
        first_person_rate_std=np.std([m.first_person_rate for m in metrics_list]),
        epistemic_total_mean=np.mean([m.epistemic_total for m in metrics_list]),
        epistemic_total_std=np.std([m.epistemic_total for m in metrics_list]),
        word_count_mean=np.mean([m.word_count for m in metrics_list]),
        word_count_std=np.std([m.word_count for m in metrics_list]),
        unique_bigram_ratio_mean=np.mean([m.unique_bigram_ratio for m in metrics_list]),
        unique_bigram_ratio_std=np.std([m.unique_bigram_ratio for m in metrics_list]),
        repetitive_pct=np.mean([m.is_repetitive for m in metrics_list]) * 100,
        degenerate_pct=np.mean([m.is_degenerate for m in metrics_list]) * 100,
        hedge_rate_mean=np.mean([m.hedge_rate for m in metrics_list]),
        assertive_rate_mean=np.mean([m.assertive_rate for m in metrics_list]),
        question_rate_mean=np.mean([m.question_rate for m in metrics_list]),
        transition_rate_mean=np.mean([m.transition_rate for m in metrics_list]),
        avg_sentence_length_mean=np.mean([m.avg_sentence_length for m in metrics_list]),
        modal_verb_rate_mean=np.mean([m.modal_verb_rate for m in metrics_list]),
        passive_voice_rate_mean=np.mean([m.passive_voice_rate for m in metrics_list]),
        # domain_vocab_rate_mean=np.mean([m.domain_vocab_rate for m in metrics_list]),
        # domain_vocab_lemmatized_rate_mean=np.mean([m.domain_vocab_lemmatized_rate for m in metrics_list]),
        ai_phrase_rate_mean=np.mean([m.ai_phrase_rate for m in metrics_list]),
        role_consistency_mean=np.mean([m.role_consistency_score for m in metrics_list]),
        bleu_mean=np.mean([m.bleu for m in metrics_list]),
        bleu_std=np.std([m.bleu for m in metrics_list]),
        compression_ratio_mean=np.mean([m.compression_ratio for m in metrics_list]),
        compression_ratio_std=np.std([m.compression_ratio for m in metrics_list]),
    )
 
 
def compute_statistical_comparison(
    values1: List[float], 
    values2: List[float],
    metric: str,
    method1: str,
    method2: str
) -> StatisticalComparison:
    """Compute statistical comparison between two sets of values"""
    
    if len(values1) != len(values2) or len(values1) == 0:
        return StatisticalComparison(metric=metric, method1=method1, method2=method2)
    
    mean1 = np.mean(values1)
    mean2 = np.mean(values2)
    
    try:
        t_stat, p_value = stats.ttest_rel(values1, values2)
    except:
        t_stat, p_value = 0.0, 1.0
    
    diff = np.array(values1) - np.array(values2)
    std_diff = np.std(diff)
    cohens_d = np.mean(diff) / std_diff if std_diff > 0 else 0
    
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        effect_size = "negligible"
    elif abs_d < 0.5:
        effect_size = "small"
    elif abs_d < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    
    return StatisticalComparison(
        metric=metric,
        method1=method1,
        method2=method2,
        mean1=mean1,
        mean2=mean2,
        t_statistic=t_stat,
        p_value=p_value,
        cohens_d=cohens_d,
        significant=p_value < 0.05,
        effect_size=effect_size
    )
 
 
def _compute_group_metrics(
    df: pd.DataFrame,
    columns: List[str],
    evaluator: PersonaEvaluator,
    use_spacy: bool,
    label: str = "",
    show_columns: bool = True,
) -> Tuple[Dict, Dict, List]:
    """Compute metrics, summaries, and comparisons for a DataFrame group."""
    all_metrics = {}
    summaries = {}

    prefix = f"  [{label}] " if label else "  "

    for col in columns:
        if col not in df.columns:
            continue
        texts = df[col].tolist()
        if show_columns:
            tqdm.write(f"{prefix}{col}: {len(texts)} texts")
        spacy_batch = [None] * len(texts)
        if use_spacy and SPACY_AVAILABLE and nlp is not None:
            valid = []
            for i, t in enumerate(texts):
                if not pd.isna(t) and str(t).strip():
                    valid.append((i, str(t)[:100_000]))
            if valid:
                tqdm.write(f"{prefix}{col}: running spaCy on {len(valid)} texts")
                indices, truncated = zip(*valid)
                for j, doc in enumerate(tqdm(nlp.pipe(truncated, batch_size=50), total=len(valid), desc=f"spaCy {label}/{col}", leave=False)):
                    sentences = list(doc.sents)
                    spacy_batch[indices[j]] = {
                        'sentence_count': len(sentences),
                        'lemmas': [tk.lemma_.lower() for tk in doc if tk.is_alpha],
                        'modal_count': sum(1 for tk in doc if tk.tag_ == 'MD'),
                        'passive_count': sum(
                            1 for tk in doc if tk.dep_ in ('auxpass', 'nsubjpass')
                        ),
                    }
        clean_texts = [str(t) if not pd.isna(t) else '' for t in texts]
        if col == 'baseline':
            bleu_batch = [1.0] * len(clean_texts)
        else:
            if 'baseline' in df.columns:
                ref_texts = [str(t) if not pd.isna(t) else '' for t in df['baseline'].tolist()]
                tqdm.write(f"{prefix}{col}: computing BLEU against baseline on {len(clean_texts)} texts")
                bleu_batch = _compute_bleu_against_reference(clean_texts, ref_texts)
            else:
                bleu_batch = [0.0] * len(clean_texts)
        metrics_list = [
            evaluator.compute_metrics(text, spacy_result=sr, bleu=sb)
            for text, sr, sb in zip(texts, spacy_batch, bleu_batch)
        ]
        all_metrics[col] = metrics_list
        summaries[col] = compute_method_summary(metrics_list)

    comparison_attrs = [
        'first_person_rate', 'unique_bigram_ratio',
        'hedge_rate', 'modal_verb_rate',
    ]
    comparisons = []
    cols = list(all_metrics.keys())
    for i, col1 in enumerate(cols):
        for col2 in cols[i + 1:]:
            for attr in comparison_attrs:
                values1 = [getattr(m, attr) for m in all_metrics[col1]]
                values2 = [getattr(m, attr) for m in all_metrics[col2]]
                comp = compute_statistical_comparison(values1, values2, attr, col1, col2)
                comparisons.append(comp)

    return all_metrics, summaries, comparisons


def analyze_single_file(
    filepath: str,
    columns: List[str] = ['assistant_axis', 'baseline', 'steered'],
    role: Optional[str] = None,
    use_spacy: bool = True,
    filters: Optional[Filters] = None
) -> Dict:
    """Analyze a single CSV file"""

    df = pd.read_csv(filepath)
    original_count = len(df)

    # Apply filters if provided
    if filters is not None and not filters.is_empty():
        df = filters.apply(df)
        filtered_count = len(df)
        if filtered_count == 0:
            print(f"  Warning: No rows remaining after filtering in {filepath}")
            return {
                'role': role or extract_role_from_filename(filepath),
                'filepath': filepath,
                'n_questions': 0,
                'n_questions_original': original_count,
                'filters_applied': filters.describe(),
                'spacy_used': False,
                'summaries': {},
                'comparisons': [],
                'raw_metrics': {},
                'per_alpha': {},
            }
    else:
        filtered_count = original_count

    if role is None:
        role = extract_role_from_filename(filepath)

    tqdm.write(f"[{role}] {filtered_count} rows")

    evaluator = PersonaEvaluator(role=role, use_spacy=use_spacy)

    per_alpha = {}
    combined_metrics: Dict[str, List[PersonaMetrics]] = {}

    if 'alpha' in df.columns:
        alpha_vals = sorted(df['alpha'].unique())
        tqdm.write(f"  [{role}] per-alpha ({len(alpha_vals)} alphas)")
        for alpha_val in alpha_vals:
            alpha_df = df[df['alpha'].apply(lambda x: abs(x - alpha_val) < 0.001)]
            tqdm.write(f"  [{role}] alpha={alpha_val:.1f} ({len(alpha_df)} rows)")
            a_metrics, a_summaries, a_comparisons = _compute_group_metrics(
                alpha_df, columns, evaluator, use_spacy, label=f"{role} a={alpha_val:.1f}", show_columns=False
            )
            per_alpha[float(alpha_val)] = {
                'n_questions': len(alpha_df),
                'summaries': {k: asdict(v) for k, v in a_summaries.items()},
                'comparisons': [asdict(c) for c in a_comparisons],
                'raw_metrics': {k: [asdict(m) for m in v] for k, v in a_metrics.items()},
            }
            for col, metrics in a_metrics.items():
                combined_metrics.setdefault(col, []).extend(metrics)

        summaries = {col: compute_method_summary(metrics) for col, metrics in combined_metrics.items()}
        comparison_attrs = ['first_person_rate', 'unique_bigram_ratio', 'hedge_rate', 'modal_verb_rate']
        comparisons = []
        cols_list = list(combined_metrics.keys())
        for i, col1 in enumerate(cols_list):
            for col2 in cols_list[i + 1:]:
                for attr in comparison_attrs:
                    values1 = [getattr(m, attr) for m in combined_metrics[col1]]
                    values2 = [getattr(m, attr) for m in combined_metrics[col2]]
                    comparisons.append(compute_statistical_comparison(values1, values2, attr, col1, col2))
        all_metrics = combined_metrics
    else:
        tqdm.write(f"  [{role}] overall metrics")
        all_metrics, summaries, comparisons = _compute_group_metrics(
            df, columns, evaluator, use_spacy, label=role
        )

    spacy_used = any(m.used_spacy for metrics in all_metrics.values() for m in metrics)

    return {
        'role': role,
        'filepath': filepath,
        'n_questions': filtered_count,
        'n_questions_original': original_count,
        'filters_applied': filters.describe() if filters else "None",
        'spacy_used': spacy_used,
        'summaries': {k: asdict(v) for k, v in summaries.items()},
        'comparisons': [asdict(c) for c in comparisons],
        'raw_metrics': {k: [asdict(m) for m in v] for k, v in all_metrics.items()},
        'per_alpha': per_alpha,
    }
 
 
def _metric_fields() -> frozenset:
    from dataclasses import fields as dc_fields
    excluded = {'epistemic_markers', 'used_spacy'}
    return frozenset(f.name for f in dc_fields(PersonaMetrics) if f.name not in excluded)


def _load_role_cache(cache_path: Path, current_fields: frozenset, filters_key: str, columns: List[str] = None) -> Optional[Dict]:
    if not cache_path.exists():
        return None
    with open(cache_path) as f:
        cached = json.load(f)
    if 'raw_metrics' not in cached:
        return None
    if cached.get('_cached_filters') != filters_key:
        print(f"  Cache filters changed for {cache_path.stem}, recomputing")
        return None
    cached_fields = frozenset(cached.get('_cached_fields', []))
    missing = current_fields - cached_fields
    if missing:
        print(f"  Cache outdated for {cache_path.stem}: missing {missing}, recomputing")
        return None
    if columns is not None:
        cached_columns = sorted(cached.get('_cached_columns', []))
        if cached_columns != sorted(columns):
            print(f"  Cache columns changed for {cache_path.stem}, recomputing")
            return None
    return cached


def _save_role_cache(cache_path: Path, result: Dict, current_fields: frozenset, filters_key: str, columns: List[str] = None):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    to_save = {**result, '_cached_fields': sorted(current_fields), '_cached_filters': filters_key}
    if columns is not None:
        to_save['_cached_columns'] = sorted(columns)
    with open(cache_path, 'w') as f:
        json.dump(to_save, f, indent=2, default=str)


def analyze_folder(
    folder_path: str,
    columns: List[str] = ['assistant_axis', 'baseline', 'steered'],
    pattern: str = "*.csv",
    use_spacy: bool = True,
    filters: Optional[Filters] = None,
    workers: int = 1,
    spacy_model: str = "en_core_web_sm",
    cache_dir: Optional[Path] = None,
) -> Dict:
    """Analyze all CSV files in a folder"""

    folder = Path(folder_path)
    files = list(folder.glob(pattern))

    print(f"Found {len(files)} CSV files in {folder_path}")
    if filters and not filters.is_empty():
        print(f"Applying filters: {filters.describe()}")

    per_role_results = {}
    aggregated_metrics = defaultdict(list)
    total_original = 0
    total_filtered = 0

    current_fields = _metric_fields()
    filters_key = filters.describe() if filters else "None"

    cached_roles = []
    uncached_files = []
    for fp in files:
        role = extract_role_from_filename(str(fp))
        if cache_dir is not None:
            hit = _load_role_cache(cache_dir / f"{role}.json", current_fields, filters_key, columns)
            if hit is not None:
                cached_roles.append((role, hit))
                continue
        uncached_files.append(fp)

    if cached_roles:
        print(f"  Cache hits: {len(cached_roles)}, recomputing: {len(uncached_files)}")

    for role, cached in cached_roles:
        per_role_results[role] = cached
        total_original += cached.get('n_questions_original', cached['n_questions'])
        total_filtered += cached['n_questions']
        for col, metrics in cached['raw_metrics'].items():
            for m in metrics:
                aggregated_metrics[col].append(m)

    def _register_result(result: Dict):
        role = result['role']
        per_role_results[role] = result
        total_original_local = result.get('n_questions_original', result['n_questions'])
        total_filtered_local = result['n_questions']
        for col, metrics in result['raw_metrics'].items():
            for m in metrics:
                aggregated_metrics[col].append(m)
        if cache_dir is not None:
            _save_role_cache(cache_dir / f"{role}.json", result, current_fields, filters_key, columns)
        return total_original_local, total_filtered_local

    n_workers = max(1, min(workers, len(uncached_files))) if uncached_files else 1

    if n_workers > 1 and uncached_files:
        work_args = [
            (str(fp), columns, use_spacy, filters) for fp in uncached_files
        ]
        print(f"  Using {n_workers} worker processes")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_spacy if use_spacy else None,
            initargs=(spacy_model,) if use_spacy else (),
        ) as executor:
            results_iter = executor.map(_process_file_worker, work_args)
            for result in tqdm(results_iter, total=len(uncached_files), desc="Processing"):
                o, f = _register_result(result)
                total_original += o
                total_filtered += f
    else:
        for filepath in tqdm(uncached_files, desc="Processing"):
            result = analyze_single_file(
                str(filepath), columns, use_spacy=use_spacy, filters=filters
            )
            o, f = _register_result(result)
            total_original += o
            total_filtered += f
    
    aggregated_summaries = {}
    for col, metrics_dicts in aggregated_metrics.items():
        metrics_list = []
        for m in metrics_dicts:
            m_copy = {k: v for k, v in m.items() if k != 'epistemic_markers'}
            metrics_list.append(PersonaMetrics(**m_copy))
        aggregated_summaries[col] = asdict(compute_method_summary(metrics_list))

    aggregated_comparisons = []
    metric_names = [
        'first_person_rate',
        'unique_bigram_ratio',
        'modal_verb_rate',
    ]

    cols = list(aggregated_metrics.keys())
    for i, col1 in enumerate(cols):
        for col2 in cols[i+1:]:
            for attr in metric_names:
                values1 = [m[attr] for m in aggregated_metrics[col1]]
                values2 = [m[attr] for m in aggregated_metrics[col2]]
                comp = compute_statistical_comparison(values1, values2, attr, col1, col2)
                aggregated_comparisons.append(asdict(comp))

    # Aggregate per-alpha across roles
    aggregated_per_alpha = defaultdict(lambda: defaultdict(list))
    for role_result in per_role_results.values():
        for alpha_str, alpha_data in role_result.get('per_alpha', {}).items():
            alpha_key = float(alpha_str)
            for col, metrics in alpha_data['raw_metrics'].items():
                aggregated_per_alpha[alpha_key][col].extend(metrics)

    per_alpha_summaries = {}
    for alpha_key in sorted(aggregated_per_alpha.keys()):
        alpha_sums = {}
        for col, metrics_dicts in aggregated_per_alpha[alpha_key].items():
            metrics_list = []
            for m in metrics_dicts:
                m_copy = {k: v for k, v in m.items() if k != 'epistemic_markers'}
                metrics_list.append(PersonaMetrics(**m_copy))
            alpha_sums[col] = asdict(compute_method_summary(metrics_list))
        per_alpha_summaries[alpha_key] = alpha_sums

    return {
        'n_roles': len(per_role_results),
        'n_total_responses': total_filtered,
        'n_total_responses_original': total_original,
        'filters_applied': filters.describe() if filters else "None",
        'roles': list(per_role_results.keys()),
        'aggregated_summaries': aggregated_summaries,
        'aggregated_comparisons': aggregated_comparisons,
        'per_role_results': per_role_results,
        'per_alpha_summaries': per_alpha_summaries,
    }
 
 
# =============================================================================
# REPORT GENERATION
# =============================================================================
 
def generate_markdown_report(results: Dict, is_folder: bool = False) -> str:
    """Generate a markdown report from results"""
    
    lines = []
    lines.append("# Persona Steering Evaluation Report\n")
    
    if is_folder:
        lines.append(f"**Roles Analyzed**: {results['n_roles']}")
        lines.append(f"**Total Responses**: {results['n_total_responses']}")
        if results.get('n_total_responses_original') and results['n_total_responses_original'] != results['n_total_responses']:
            lines.append(f"**Original Responses**: {results['n_total_responses_original']}")
        lines.append(f"**Filters**: {results.get('filters_applied', 'None')}")
        lines.append(f"**spaCy**: {'Enabled' if SPACY_AVAILABLE else 'Disabled (regex fallback)'}\n")
        summaries = results['aggregated_summaries']
        comparisons = results['aggregated_comparisons']
    else:
        lines.append(f"**Role**: {results['role']}")
        lines.append(f"**Questions**: {results['n_questions']}")
        if results.get('n_questions_original') and results['n_questions_original'] != results['n_questions']:
            lines.append(f"**Original Questions**: {results['n_questions_original']}")
        lines.append(f"**Filters**: {results.get('filters_applied', 'None')}")
        lines.append(f"**spaCy**: {'Used' if results.get('spacy_used', False) else 'Not used'}\n")
        summaries = results['summaries']
        comparisons = results['comparisons']
    
    lines.append("## Method Summaries\n")
    lines.append("| Metric | " + " | ".join(summaries.keys()) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(summaries)) + "|")
    
    metrics_to_show = [
        ('first_person_rate_mean', 'First-Person Rate (%)', '.2f'),
        ('word_count_mean', 'Avg Word Count', '.1f'),
        ('unique_bigram_ratio_mean', 'Unique Bigram Ratio', '.3f'),
        ('repetitive_pct', 'Repetitive (%)', '.1f'),
        ('degenerate_pct', 'Degenerate Length (%)', '.1f'),
        ('modal_verb_rate_mean', 'Modal Verb Rate (%)', '.2f'),
        ('question_rate_mean', 'Questions/Response', '.2f'),
        ('ai_phrase_rate_mean', 'AI Phrase Leakage', '.3f'),
        ('bleu_mean', 'BLEU', '.3f'),
        ('compression_ratio_mean', 'Compression Ratio', '.3f'),
    ]
    
    for key, label, fmt in metrics_to_show:
        values = []
        for method in summaries.keys():
            val = summaries[method].get(key, 0)
            values.append(f"{val:{fmt}}")
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    
    lines.append("\n## Statistical Comparisons\n")
    
    for comp in comparisons:
        sig = "***" if comp['p_value'] < 0.001 else "**" if comp['p_value'] < 0.01 else "*" if comp['p_value'] < 0.05 else ""
        lines.append(f"### {comp['method1']} vs {comp['method2']}: {comp['metric']}")
        lines.append(f"- Means: {comp['mean1']:.3f} vs {comp['mean2']:.3f}")
        lines.append(f"- p-value: {comp['p_value']:.4f} {sig}")
        lines.append(f"- Cohen's d: {comp['cohens_d']:.3f} ({comp['effect_size']})")
        lines.append("")
    
    lines.append("\n## Key Findings\n")
    
    key_metrics = [
        ('first_person_rate_mean', 'First-Person Rate', True),
        ('unique_bigram_ratio_mean', 'Unique Bigram Ratio', True),
        ('repetitive_pct', 'Repetitive %', False),
        ('modal_verb_rate_mean', 'Modal Verb Rate', True),
    ]
    
    for key, label, higher_is_better in key_metrics:
        if higher_is_better:
            best_method = max(summaries.keys(), key=lambda m: summaries[m].get(key, 0))
        else:
            best_method = min(summaries.keys(), key=lambda m: summaries[m].get(key, float('inf')))
        
        best_val = summaries[best_method].get(key, 0)
        lines.append(f"- **{label}**: {best_method} leads with {best_val:.2f}")
    
    return "\n".join(lines)
 
 
def generate_per_role_summary(results: Dict) -> str:
    """Generate per-role summary table"""
    
    if 'per_role_results' not in results:
        return ""
    
    lines = []
    lines.append("\n## Per-Role Summary\n")
    lines.append("| Role | Best (First-Person) | Best (Bigram Ratio) | Repetitive Issues |")
    lines.append("|------|---------------------|---------------------|-------------------|")

    for role, role_result in sorted(results['per_role_results'].items()):
        summaries = role_result['summaries']

        if not summaries:
            continue

        best_fp = max(summaries.keys(), key=lambda m: summaries[m].get('first_person_rate_mean', 0))
        best_br = max(summaries.keys(), key=lambda m: summaries[m].get('unique_bigram_ratio_mean', 0))

        repetitive_issues = [m for m, s in summaries.items() if s.get('repetitive_pct', 0) > 50]
        rep_str = ", ".join(repetitive_issues) if repetitive_issues else "None"

        lines.append(f"| {role} | {best_fp} | {best_br} | {rep_str} |")
    
    return "\n".join(lines)


def generate_per_alpha_report(results: Dict, is_folder: bool = False) -> str:
    """Generate per-alpha breakdown table."""
    if is_folder:
        per_alpha = results.get('per_alpha_summaries', {})
    else:
        per_alpha = results.get('per_alpha', {})

    if not per_alpha:
        return ""

    lines = []
    lines.append("\n## Per-Alpha Breakdown\n")

    # For single-file mode, per_alpha values contain full result dicts with 'summaries'
    # For folder mode, per_alpha_summaries values are already {col: summary_dict}
    alpha_keys = sorted(per_alpha.keys(), key=float)

    # Determine available methods from first alpha entry
    first_entry = per_alpha[alpha_keys[0]]
    if 'summaries' in first_entry:
        methods = list(first_entry['summaries'].keys())
        get_summaries = lambda entry: entry['summaries']
    else:
        methods = list(first_entry.keys())
        get_summaries = lambda entry: entry

    metrics_to_show = [
        ('first_person_rate_mean', '1P Rate %', '.2f'),
        ('word_count_mean', 'Words', '.0f'),
        ('unique_bigram_ratio_mean', 'Bigram Ratio', '.3f'),
        ('repetitive_pct', 'Repet. %', '.1f'),
        ('degenerate_pct', 'Degen. %', '.1f'),
    ]

    for method in methods:
        lines.append(f"### {method}\n")
        header_labels = [label for _, label, _ in metrics_to_show]
        lines.append("| Alpha | " + " | ".join(header_labels) + " |")
        lines.append("|-------|" + "|".join(["--------"] * len(metrics_to_show)) + "|")

        for alpha_key in alpha_keys:
            sums = get_summaries(per_alpha[alpha_key])
            if method not in sums:
                continue
            method_summary = sums[method]
            vals = []
            for key, _, fmt in metrics_to_show:
                val = method_summary.get(key, 0)
                vals.append(f"{val:{fmt}}")
            lines.append(f"| {float(alpha_key):.1f} | " + " | ".join(vals) + " |")

        lines.append("")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================
 
def main():
    parser = argparse.ArgumentParser(
        description='Evaluate persona steering methodologies (hybrid spaCy + regex)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python nlp_evaluator.py --input comparison.csv
  python nlp_evaluator.py --input ./comparisons/ --output ./results/
  
  # With visualizations
  python nlp_evaluator.py --input ./comparisons/ --visualize
  
  # Filter by specific parameters
  python nlp_evaluator.py --input ./comparisons/ --layer 16
  python nlp_evaluator.py --input ./comparisons/ --alpha 2.5 --temperature 0.2
  python nlp_evaluator.py --input ./comparisons/ --role accountant teacher doctor
  
  # Combine multiple filters
  python nlp_evaluator.py --input ./comparisons/ --layer 16 32 --alpha 2.5
  
  # Disable spaCy
  python nlp_evaluator.py --input ./comparisons/ --no-spacy

  # Parallel processing with 4 workers
  python nlp_evaluator.py --input ./comparisons/ --workers 4
        """
    )
    
    parser.add_argument('--input', '-i', type=str,
                       default='experiment_data/gold_prompt_experiments',
                       help='Input CSV file or folder containing CSVs')
    parser.add_argument('--output', '-o', type=str, default='./analysis/empirical/nlp_experiments/figures/',
                       help='Output directory')
    parser.add_argument('--columns', '-c', nargs='+', 
                       default=['assistant_axis', 'baseline', 'steered'],
                       help='Response columns to compare')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--boxplot', action='store_true',
                       help='Use box-and-whisker plots instead of line graphs')
    parser.add_argument('--no-spacy', action='store_true',
                       help='Disable spaCy (use regex fallback)')
    parser.add_argument('--spacy-model', type=str, default='en_core_web_sm',
                       help='spaCy model to use')
    parser.add_argument('--workers', '-w', type=int, default=8,
                       help='Number of worker processes for folder mode (default: 8)')
    # Filter arguments
    parser.add_argument('--role', type=str, nargs='+', default=None,
                       help='Filter by role(s), e.g., --role accountant teacher')
    parser.add_argument('--layer', type=int, nargs='+', default=[16],
                       help='Filter by layer(s) (default: 16)')
    parser.add_argument('--sample-count', type=int, nargs='+', default=[50],
                       help='Filter by sample_count(s) (default: 50)')
    parser.add_argument('--alpha', type=float, nargs='+', default=None,
                       help='Filter by alpha(s) (default: all of 1.0 1.5 2.0 2.5)')
    parser.add_argument('--temperature', type=float, nargs='+', default=None,
                       help='Filter by temperature(s), e.g., --temperature 0.2 0.5')

    args = parser.parse_args()

    alpha = args.alpha if args.alpha is not None else [1.0, 1.5, 2.0, 2.5]

    filters = Filters(
        role=args.role,
        layer=args.layer,
        sample_count=args.sample_count,
        alpha=alpha,
        temperature=args.temperature
    )
    
    use_spacy = True
    if args.no_spacy:
        use_spacy = False
        print("spaCy disabled by --no-spacy flag")
    else:
        init_spacy(args.spacy_model)
        use_spacy = SPACY_AVAILABLE

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    is_folder = input_path.is_dir()
    
    print(f"\nInput: {input_path} ({'folder' if is_folder else 'file'})")
    print(f"Output: {output_dir}")
    print(f"Columns: {args.columns}")
    print(f"Filters: {filters.describe()}")
    print(f"spaCy: {'Enabled' if use_spacy else 'Disabled'}")
    if is_folder and args.workers > 1:
        print(f"Workers: {args.workers}")
    print()

    cache_dir = output_dir / "per_role" if is_folder else None

    if is_folder:
        results = analyze_folder(
            str(input_path), args.columns, use_spacy=use_spacy, filters=filters,
            workers=args.workers, spacy_model=args.spacy_model,
            cache_dir=cache_dir,
        )
    else:
        results = analyze_single_file(str(input_path), args.columns, use_spacy=use_spacy, filters=filters)
    
    json_path = output_dir / "results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {json_path}")
    
    report = generate_markdown_report(results, is_folder)
    report += generate_per_alpha_report(results, is_folder)
    if is_folder:
        report += generate_per_role_summary(results)
    
    md_path = output_dir / "report.md"
    with open(md_path, 'w') as f:
        f.write(report)
    print(f"Saved: {md_path}")
    
    print("\n" + "="*60)
    print(report[:2500])
    if len(report) > 2500:
        print("\n... (see full report in file)")
    
    if args.visualize:
        print("\nGenerating visualizations...")
        generate_visualizations(results, str(output_dir), is_folder, boxplot=args.boxplot)
    
    if is_folder and cache_dir is not None:
        print(f"Per-role cache: {cache_dir} ({len(results.get('per_role_results', {}))} roles)")
    
    print("\nDone!")
 
 
if __name__ == "__main__":
    main()