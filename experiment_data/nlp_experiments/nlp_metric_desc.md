# NLP Metrics for Evaluating Persona Steering Methodologies

## Overview

This evaluation suite measures the quality of persona steering across three methodologies:
- **Baseline**: SOTA roleplaying prompt engineering (gold standard)
- **Steered**: Your methodology  
- **Assistant_Axis**: Lu et al.'s activation steering approach

The suite supports both **single-role evaluation** and **multi-role aggregation** across 100+ persona files.

---

## Quick Start

```bash
# Single file
python nlp_evaluator.py --input Comparison_GoldStandard_accountant.csv

# Entire folder
python nlp_evaluator.py --input ./comparisons/ --output results/

# With filters
python nlp_evaluator.py --input ./comparisons/ --layer 16 --alpha 2.5

# Filter by multiple roles
python nlp_evaluator.py --input ./comparisons/ --role accountant teacher doctor
```

---

## 1. METRIC DESCRIPTIONS

### 1.1 Persona Identity Metrics

#### First-Person Rate (%)
| | |
|---|---|
| **What it measures** | How strongly the model embodies a persona by using self-referential language |
| **Calculation** | Count of (I, me, my, mine, myself) / total words × 100 |
| **Why it matters** | A persona that says "I think..." or "In my experience..." is adopting an identity. Generic AI responses avoid first-person. Higher = stronger persona embodiment. |
| **Interpretation** | 0-1%: Impersonal, generic AI response. 2-4%: Moderate persona presence. 5%+: Strong persona identity. |
| **Implementation** | Regex (optimal—closed class words, no ambiguity) |

#### Epistemic Markers (count)
| | |
|---|---|
| **What it measures** | Expressions of personal belief, opinion, or perspective |
| **Calculation** | Count of phrases: "I think", "I believe", "In my view", "In my opinion", "From my perspective", "I prefer", "I've found", "Personally", etc. |
| **Why it matters** | These phrases signal the model is expressing a viewpoint *as the persona*, not just stating facts. Key indicator of roleplay depth. |
| **Interpretation** | 0: No personal opinion expressed. 1-2: Some opinion markers. 3+: Strong personal voice. |
| **Implementation** | Regex (optimal—fixed phrases) |

#### Role Consistency Score (0-1)
| | |
|---|---|
| **What it measures** | Whether persona markers are consistent throughout the response or fade out |
| **Calculation** | Split response into thirds, compute first-person density in each, measure variance. Lower variance = higher score. |
| **Why it matters** | Detects if model starts in-character but drifts to generic AI tone mid-response. |
| **Interpretation** | 0.8-1.0: Consistent persona throughout. 0.5-0.8: Some drift. <0.5: Significant persona inconsistency. |
| **Implementation** | Regex |

---

### 1.2 Response Quality Metrics

#### Unique Bigram Ratio (0-1)
| | |
|---|---|
| **What it measures** | Text coherence and non-repetition |
| **Calculation** | (Number of unique consecutive word pairs) / (Total word pairs) |
| **Why it matters** | Detects degenerate, looping text where the model gets stuck repeating phrases. Critical for detecting broken outputs. |
| **Interpretation** | 0.8-1.0: Normal, coherent text. 0.5-0.8: Some repetition. <0.5: Severely repetitive/degenerate. |
| **Implementation** | Regex (optimal—simple sequence counting) |

#### Repetitive (%)
| | |
|---|---|
| **What it measures** | Percentage of responses flagged as degenerate loops |
| **Calculation** | % of responses where unique bigram ratio < 0.5 |
| **Why it matters** | A method that produces 90%+ repetitive responses is fundamentally broken. This is the clearest failure mode indicator. |
| **Interpretation** | 0%: No degeneration issues. 1-10%: Occasional issues. 50%+: Severe methodology problem. |
| **Implementation** | Derived from unique bigram ratio |

#### Degenerate Length (%)
| | |
|---|---|
| **What it measures** | Percentage of responses that are excessively long |
| **Calculation** | % of responses with word count > 500 |
| **Why it matters** | Runaway generation often indicates the model is stuck or confused. Normal responses to simple questions shouldn't be 1000+ words. |
| **Interpretation** | 0-10%: Normal. 10-30%: Verbose but possibly acceptable. 50%+: Likely degeneration issue. |
| **Implementation** | Simple word count threshold |

#### Word Count (avg)
| | |
|---|---|
| **What it measures** | Average response length |
| **Calculation** | Mean word count across responses |
| **Why it matters** | Context for other metrics. Very long responses dilute density metrics; very short may lack substance. |
| **Interpretation** | Compare relative to baseline for the same questions. |
| **Implementation** | Regex tokenization |

---

### 1.3 Style Metrics

#### Hedge Rate (%)
| | |
|---|---|
| **What it measures** | Use of uncertainty/hedging language |
| **Calculation** | Count of (perhaps, maybe, possibly, might, could, seems, likely, probably, generally, typically, etc.) / total words × 100 |
| **Why it matters** | Hedging can indicate appropriate epistemic humility or professional caution. Too much = uncertain; too little = overconfident. |
| **Interpretation** | Role-dependent. Compare to baseline. Therapists may hedge more than engineers. |
| **Implementation** | Regex word list |

#### Assertive Rate (%)
| | |
|---|---|
| **What it measures** | Use of confident/definitive language |
| **Calculation** | Count of (definitely, certainly, clearly, obviously, absolutely, must, always, never, etc.) / total words × 100 |
| **Why it matters** | Assertive language signals confidence. Should balance with hedging for appropriate tone. |
| **Interpretation** | Higher isn't always better—depends on role and context. |
| **Implementation** | Regex word list |

#### Modal Verb Rate (%)
| | |
|---|---|
| **What it measures** | Use of modal verbs (would, could, should, might, may, can, will, must) |
| **Calculation** | Count of modal verbs / total words × 100 |
| **Why it matters** | Modals indicate hypothetical reasoning, suggestions, or professional recommendations ("I would suggest...", "You might consider..."). Common in advisory personas. |
| **Interpretation** | Higher rates typical for consultative/advisory roles. |
| **Implementation** | spaCy POS tag "MD" (falls back to word list without spaCy) |

#### Questions per Response (avg)
| | |
|---|---|
| **What it measures** | How often the model asks questions |
| **Calculation** | Count of "?" per response, averaged |
| **Why it matters** | Questions can indicate engagement ("Does that help?") or Socratic style. Very high may indicate filler or deflection. |
| **Interpretation** | 0-0.5: Declarative style. 1-2: Conversational. 3+: Possibly excessive. |
| **Implementation** | Regex (count "?") |

#### Avg Sentence Length (words)
| | |
|---|---|
| **What it measures** | Sentence structure complexity |
| **Calculation** | Total words / number of sentences |
| **Why it matters** | Very long sentences may be hard to read; very short may seem choppy or incomplete. |
| **Interpretation** | 15-25 words/sentence is typical for readable prose. |
| **Implementation** | spaCy sentence segmentation (better for abbreviations like "Dr.", "U.S."), regex fallback |

---

### 1.4 Role Fidelity Metrics
<!-- 
#### Domain Vocab Rate (%)
| | |
|---|---|
| **What it measures** | Use of role-specific terminology (surface forms) |
| **Calculation** | Count of domain-specific words / total words × 100 |
| **Why it matters** | An accountant should use financial terms; a doctor should use medical terms. Indicates role knowledge expression. |
| **Limitation** | Currently uses hardcoded word lists for limited roles (accountant, doctor, lawyer, teacher, engineer, chef, therapist). Falls back to generic list for unknown roles. |
| **Interpretation** | Compare to baseline for same role. Baseline typically wins here. |
| **Implementation** | Regex word list matching |

#### Domain Vocab Lemmatized Rate (%)
| | |
|---|---|
| **What it measures** | Same as above but matching lemmatized (base) word forms |
| **Calculation** | spaCy lemmatizes words before matching (e.g., "auditing" → "audit", "investments" → "investment") |
| **Why it matters** | Better recall than surface matching—catches word variants without needing to list every form. |
| **Interpretation** | Should be ≥ surface rate. If equal, either spaCy disabled or words already in base form. |
| **Implementation** | spaCy lemmatization + word list matching | -->

#### AI Phrase Leakage (count)
| | |
|---|---|
| **What it measures** | Breaking character by revealing AI identity |
| **Calculation** | Count of phrases: "As an AI", "as a language model", "I don't have personal", "my training", "I was trained", "I'm designed to", etc. |
| **Why it matters** | A persona should stay in character. Saying "As an AI, I can't..." breaks immersion and indicates failed roleplay. |
| **Interpretation** | 0: Good—no character breaks. 0.01-0.05: Rare slips. 0.1+: Frequent character breaking. |
| **Implementation** | Regex (optimal—fixed phrases) |

---

### 1.5 spaCy-Only Metrics

#### Passive Voice Rate (%)
| | |
|---|---|
| **What it measures** | Use of passive voice constructions |
| **Calculation** | Passive voice markers detected via dependency parsing / sentences × 100 |
| **Why it matters** | Passive voice is more formal/distanced ("The report was prepared" vs "I prepared the report"). May indicate less personal engagement. |
| **Interpretation** | Higher = more formal/impersonal tone. Compare to baseline. |
| **Implementation** | spaCy dependency parsing (not available without spaCy) |

---

## 2. METRIC PRIORITY FOR YOUR EVALUATION

### Tier 1: Critical (Your Core Claims)
| Metric | Your Hypothesis | Expected Result |
|--------|-----------------|-----------------|
| **First-Person Rate** | Your method → stronger persona | Steered > Baseline > Assistant_Axis |
| **Epistemic Markers** | Your method → more "I think/believe" | Steered > Baseline > Assistant_Axis |
| **Unique Bigram Ratio** | Your method → coherent output | Steered ≈ Baseline >> Assistant_Axis |
| **Repetitive %** | Lu's method degenerates | Assistant_Axis >> Steered ≈ Baseline |

### Tier 2: Supporting Evidence
| Metric | What It Shows |
|--------|---------------|
| **Role Consistency** | Persona maintained throughout response |
| **AI Phrase Leakage** | Character not broken |
| **Modal Verb Rate** | Professional/advisory tone |

### Tier 3: Context (Compare to Baseline)
| Metric | What It Shows |
|--------|---------------|
<!-- | **Domain Vocab Rate** | Role expertise vocabulary | -->
| **Hedge/Assertive Rate** | Tone calibration |
| **Word Count** | Verbosity |

---

## 3. FILTER OPTIONS

Filter data before analysis by experimental parameters:

| Argument | Type | Example | Description |
|----------|------|---------|-------------|
| `--role` | string(s) | `--role accountant teacher` | Filter by role column |
| `--layer` | int(s) | `--layer 16 32` | Filter by layer column |
| `--sample-count` | int(s) | `--sample-count 50 100` | Filter by sample_count column |
| `--alpha` | float(s) | `--alpha 2.5 3.0` | Filter by alpha column |
| `--temperature` | float(s) | `--temperature 0.2 0.5` | Filter by temperature column |

Multiple values for same filter = OR logic. Different filters = AND logic.

---

## 4. STATISTICAL ANALYSIS

### Significance Thresholds
| Symbol | p-value | Interpretation |
|--------|---------|----------------|
| *** | p < 0.001 | Highly significant |
| ** | p < 0.01 | Very significant |
| * | p < 0.05 | Significant |
| (none) | p ≥ 0.05 | Not significant |

### Effect Size (Cohen's d)
| |d| | Interpretation |
|-----|----------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| > 0.8 | Large |

---

## 5. INTERPRETING RESULTS

### Your Method Wins If:
1. **Higher first-person rate** than Lu's → Stronger persona embodiment
2. **More epistemic markers** → More personal opinion expression  
3. **Higher unique bigram ratio** → Less degeneration
4. **Lower repetitive %** → Coherent outputs
5. **Zero AI phrase leakage** → Maintains character

### Watch Out For:
<!-- 1. **Lower domain vocab** than baseline → May sacrifice expertise vocabulary -->
2. **Higher degenerate length %** → May be verbose
3. **Very high question rate** → May be filler

---

## 6. COMMAND LINE REFERENCE

```bash
# Basic usage
python nlp_evaluator.py --input <file_or_folder>

# Full options
python nlp_evaluator.py \
    --input ./comparisons/ \
    --output ./results/ \
    --columns assistant_axis baseline steered \
    --layer 16 \
    --alpha 2.5 \
    --temperature 0.2 \
    --visualize

# Disable spaCy (faster, regex-only)
python nlp_evaluator.py --input ./comparisons/ --no-spacy

# Use larger spaCy model
python nlp_evaluator.py --input ./comparisons/ --spacy-model en_core_web_lg
```