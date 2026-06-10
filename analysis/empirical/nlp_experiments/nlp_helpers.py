#!/usr/bin/env python3
"""
Persona Steering Evaluation Metrics (Hybrid spaCy + Regex)

Supports nlp_evaluator.py
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
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA CLASSES
# =============================================================================
 
@dataclass
class PersonaMetrics:
    """Container for all persona-related metrics for a single response"""
    # Identity metrics (regex-based)
    first_person_rate: float = 0.0
    epistemic_markers: Dict[str, int] = field(default_factory=dict)
    epistemic_total: int = 0

    # Quality metrics (regex-based)
    word_count: int = 0
    unique_bigram_ratio: float = 1.0
    is_repetitive: bool = False
    is_degenerate: bool = False

    # Style metrics (hybrid)
    hedge_rate: float = 0.0
    assertive_rate: float = 0.0
    question_rate: float = 0.0
    transition_rate: float = 0.0
    avg_sentence_length: float = 0.0  # spaCy-enhanced
    sentence_count: int = 0

    # spaCy-enhanced metrics
    modal_verb_rate: float = 0.0
    passive_voice_rate: float = 0.0

    # Role metrics (spaCy-enhanced for lemmatization)
    # domain_vocab_count: int = 0
    # domain_vocab_rate: float = 0.0
    # domain_vocab_lemmatized_count: int = 0
    # domain_vocab_lemmatized_rate: float = 0.0
    ai_phrase_rate: float = 0.0

    # Consistency
    role_consistency_score: float = 1.0

    # Diversity / compression
    bleu: float = 0.0
    compression_ratio: float = 0.0

    # Meta
    used_spacy: bool = False
 
 
@dataclass
class MethodSummary:
    """Aggregated metrics for a single method"""
    n_responses: int = 0
    
    # Means and stds
    first_person_rate_mean: float = 0.0
    first_person_rate_std: float = 0.0
    epistemic_total_mean: float = 0.0
    epistemic_total_std: float = 0.0
    word_count_mean: float = 0.0
    word_count_std: float = 0.0
    unique_bigram_ratio_mean: float = 0.0
    unique_bigram_ratio_std: float = 0.0
    
    # Percentages
    repetitive_pct: float = 0.0
    degenerate_pct: float = 0.0
    
    # Style means
    hedge_rate_mean: float = 0.0
    assertive_rate_mean: float = 0.0
    question_rate_mean: float = 0.0
    transition_rate_mean: float = 0.0
    avg_sentence_length_mean: float = 0.0
    
    # spaCy-enhanced means
    modal_verb_rate_mean: float = 0.0
    passive_voice_rate_mean: float = 0.0
    
    # Role means
    # domain_vocab_rate_mean: float = 0.0
    # domain_vocab_lemmatized_rate_mean: float = 0.0
    ai_phrase_rate_mean: float = 0.0
    role_consistency_mean: float = 0.0
    bleu_mean: float = 0.0
    bleu_std: float = 0.0
    compression_ratio_mean: float = 0.0
    compression_ratio_std: float = 0.0
 
 
@dataclass 
class StatisticalComparison:
    """Statistical comparison between two methods"""
    metric: str
    method1: str
    method2: str
    mean1: float = 0.0
    mean2: float = 0.0
    t_statistic: float = 0.0
    p_value: float = 1.0
    cohens_d: float = 0.0
    significant: bool = False
    effect_size: str = "negligible"
 
 
@dataclass
class Filters:
    """Filters to apply to the data before analysis"""
    role: Optional[List[str]] = None
    layer: Optional[List[int]] = None
    sample_count: Optional[List[int]] = None
    alpha: Optional[List[float]] = None
    temperature: Optional[List[float]] = None

    def is_empty(self) -> bool:
        """Check if no filters are set"""
        return all(v is None for v in [self.role, self.layer, self.sample_count,
                                        self.alpha, self.temperature])

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply filters to a DataFrame, return filtered copy"""
        filtered = df.copy()

        if self.role is not None and 'role' in filtered.columns:
            filtered = filtered[filtered['role'].isin(self.role)]

        if self.layer is not None and 'layer' in filtered.columns:
            filtered = filtered[filtered['layer'].isin(self.layer)]

        if self.sample_count is not None and 'sample_count' in filtered.columns:
            filtered = filtered[filtered['sample_count'].isin(self.sample_count)]

        if self.alpha is not None and 'alpha' in filtered.columns:
            filtered = filtered[filtered['alpha'].apply(
                lambda x: any(abs(x - a) < 0.001 for a in self.alpha)
            )]

        if self.temperature is not None and 'temperature' in filtered.columns:
            filtered = filtered[filtered['temperature'].apply(
                lambda x: any(abs(x - t) < 0.001 for t in self.temperature)
            )]

        return filtered

    def describe(self) -> str:
        """Return human-readable description of active filters"""
        parts = []
        if self.role:
            parts.append(f"role={self.role}")
        if self.layer:
            parts.append(f"layer={self.layer}")
        if self.sample_count:
            parts.append(f"sample_count={self.sample_count}")
        if self.alpha:
            parts.append(f"alpha={self.alpha}")
        if self.temperature:
            parts.append(f"temperature={self.temperature}")
        return ", ".join(parts) if parts else "None"
 
 
# =============================================================================
# LEXICONS
# =============================================================================
 
class Lexicons:
    """Centralized lexicon storage"""
    
    # Regex-based (closed class, no ambiguity)
    FIRST_PERSON = {'i', 'me', 'my', 'mine', 'myself'}
    
    EPISTEMIC_MARKERS = [
        (r'\bI think\b', 'I think'),
        (r'\bI believe\b', 'I believe'),
        (r'\bI feel\b', 'I feel'),
        (r'\bI consider\b', 'I consider'),
        (r'\bI find\b', 'I find'),
        (r'\bI prefer\b', 'I prefer'),
        (r'\bI would say\b', 'I would say'),
        (r'\bI\'d say\b', "I'd say"),
        (r'\bI\'d argue\b', "I'd argue"),
        (r'\bIn my view\b', 'In my view'),
        (r'\bIn my opinion\b', 'In my opinion'),
        (r'\bIn my experience\b', 'In my experience'),
        (r'\bFrom my perspective\b', 'From my perspective'),
        (r'\bTo me\b', 'To me'),
        (r'\bPersonally\b', 'Personally'),
        (r'\bI\'ve found\b', "I've found"),
        (r'\bI\'ve seen\b', "I've seen"),
        (r'\bI\'ve noticed\b', "I've noticed"),
        (r'\bMy sense is\b', 'My sense is'),
        (r'\bMy take is\b', 'My take is'),
    ]
    
    AI_PHRASES = [
        r'\bas an ai\b',
        r'\bas a language model\b',
        r'\bas an assistant\b',
        r'\bi don\'t have personal\b',
        r'\bi\'m not able to\b',
        r'\bi cannot\b',
        r'\bmy training\b',
        r'\bi am designed\b',
        r'\bi was trained\b',
        r'\blarge language model\b',
        r'\bi don\'t have feelings\b',
        r'\bi don\'t have opinions\b',
    ]
    
    # Word lists (regex fallback, spaCy can enhance)
    HEDGE_WORDS = {
        'perhaps', 'maybe', 'possibly', 'might', 'could', 'may',
        'seems', 'seem', 'appears', 'appear', 'likely', 'unlikely',
        'probably', 'presumably', 'arguably', 'somewhat', 'relatively',
        'tend', 'tends', 'generally', 'typically', 'usually', 'often',
        'sometimes', 'occasionally', 'roughly', 'approximately'
    }
    
    ASSERTIVE_WORDS = {
        'definitely', 'certainly', 'clearly', 'obviously', 'undoubtedly',
        'absolutely', 'must', 'always', 'never', 'entirely', 'completely',
        'totally', 'surely', 'precisely', 'exactly', 'fundamentally'
    }
    
    TRANSITIONS = {
        'however', 'therefore', 'moreover', 'furthermore', 'consequently',
        'nevertheless', 'nonetheless', 'although', 'though', 'because',
        'since', 'while', 'whereas', 'hence', 'thus', 'accordingly',
        'additionally', 'similarly', 'conversely', 'alternatively',
        'meanwhile', 'subsequently', 'ultimately', 'specifically'
    }
    
    # Modal verbs (spaCy POS tag: MD)
    MODAL_VERBS = {'would', 'could', 'should', 'might', 'may', 'can', 'will', 'must', 'shall'}
    
    # # Domain vocabulary by role - LEMMATIZED forms for spaCy matching
    # DOMAIN_VOCAB_LEMMAS = {
    #     'accountant': {
    #         'financial', 'budget', 'audit', 'tax', 'compliance', 'regulation',
    #         'regulatory', 'fiscal', 'invest', 'investment', 'capital', 'asset',
    #         'liability', 'revenue', 'expense', 'accounting', 'balance', 'ledger',
    #         'statement', 'gaap', 'depreciation', 'amortization', 'equity',
    #         'cashflow', 'receivable', 'payable', 'reconciliation', 'accrual',
    #         'debit', 'credit', 'journal', 'variance', 'forecast', 'quarter',
    #         'fiscal', 'margin', 'profit', 'loss', 'income', 'cost'
    #     },
    #     'doctor': {
    #         'patient', 'diagnosis', 'treatment', 'symptom', 'prescription',
    #         'medication', 'medical', 'clinical', 'health', 'disease', 'condition',
    #         'therapy', 'prognosis', 'examination', 'hospital', 'surgery',
    #         'chronic', 'acute', 'vital', 'dosage', 'physician', 'nurse',
    #         'diagnosis', 'treatment', 'care', 'wellness', 'recovery'
    #     },
    #     'lawyer': {
    #         'legal', 'court', 'statute', 'plaintiff', 'defendant', 'jurisdiction',
    #         'litigation', 'contract', 'liability', 'precedent', 'verdict',
    #         'testimony', 'evidence', 'counsel', 'attorney', 'judge', 'trial',
    #         'settlement', 'appeal', 'law', 'case', 'client', 'sue', 'ruling'
    #     },
    #     'teacher': {
    #         'student', 'curriculum', 'lesson', 'learning', 'education',
    #         'classroom', 'assessment', 'teaching', 'pedagogy', 'grade',
    #         'homework', 'instruction', 'academic', 'school', 'exam', 'test',
    #         'lecture', 'assignment', 'course', 'knowledge', 'skill'
    #     },
    #     'engineer': {
    #         'design', 'system', 'specification', 'prototype', 'testing',
    #         'technical', 'implementation', 'architecture', 'component',
    #         'integration', 'performance', 'optimization', 'code', 'algorithm',
    #         'software', 'hardware', 'debug', 'deploy', 'scale', 'efficiency'
    #     },
    #     'chef': {
    #         'ingredient', 'recipe', 'cooking', 'flavor', 'dish', 'kitchen',
    #         'culinary', 'taste', 'seasoning', 'preparation', 'cuisine', 'menu',
    #         'plating', 'sauce', 'broth', 'roast', 'grill', 'bake', 'sauté'
    #     },
    #     'therapist': {
    #         'therapy', 'client', 'emotional', 'feeling', 'mental', 'counseling',
    #         'session', 'anxiety', 'depression', 'trauma', 'coping', 'mindfulness',
    #         'psychological', 'support', 'behavior', 'cognitive', 'emotion'
    #     },
    #     'default': {
    #         'professional', 'experience', 'expertise', 'approach', 'strategy',
    #         'analysis', 'process', 'solution', 'quality', 'skill', 'knowledge'
    #     }
    # }
    
    # # Surface forms for regex fallback (non-lemmatized)
    # DOMAIN_VOCAB_SURFACE = {
    #     'accountant': {
    #         'financial', 'budget', 'budgeting', 'audit', 'auditing', 'audited',
    #         'tax', 'taxes', 'taxation', 'compliance', 'regulation', 'regulations',
    #         'regulatory', 'fiscal', 'invest', 'investment', 'investments', 'investing',
    #         'capital', 'asset', 'assets', 'liability', 'liabilities',
    #         'revenue', 'revenues', 'expense', 'expenses', 'accounting',
    #         'balance', 'ledger', 'statement', 'statements', 'gaap',
    #         'depreciation', 'amortization', 'equity', 'cashflow',
    #         'receivable', 'receivables', 'payable', 'payables', 'reconciliation', 'accrual'
    #     },
    #     'default': {
    #         'professional', 'experience', 'expertise', 'approach',
    #         'strategy', 'analysis', 'process', 'solution', 'quality'
    #     }
    # }
    
    # @classmethod
    # def get_domain_vocab_lemmas(cls, role: str) -> Set[str]:
    #     """Get lemmatized domain vocabulary for a role (for spaCy)"""
    #     role_lower = role.lower()
    #     for key in cls.DOMAIN_VOCAB_LEMMAS:
    #         if key in role_lower:
    #             return cls.DOMAIN_VOCAB_LEMMAS[key]
    #     return cls.DOMAIN_VOCAB_LEMMAS['default']
    
    # @classmethod
    # def get_domain_vocab_surface(cls, role: str) -> Set[str]:
    #     """Get surface form domain vocabulary for a role (for regex fallback)"""
    #     role_lower = role.lower()
    #     for key in cls.DOMAIN_VOCAB_SURFACE:
    #         if key in role_lower:
    #             return cls.DOMAIN_VOCAB_SURFACE[key]
    #     return cls.get_domain_vocab_lemmas(role)