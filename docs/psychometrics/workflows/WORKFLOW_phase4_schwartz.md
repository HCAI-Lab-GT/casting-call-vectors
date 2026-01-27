# Phase 4 Workflow: Schwartz Values Implementation

> **Status**: Ready for Implementation
>
> **Priority**: MEDIUM
>
> **Reference**: [PSYCHOMETRICS_DATA.md § Phase 4](../reference/PSYCHOMETRICS_DATA.md#phase-4-schwartz-values)
>
> **Last Updated**: 2025-01-25

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Why Schwartz Values for LM-VECTOR](#2-why-schwartz-values-for-lm-vector)
3. [Instrument Overview](#3-instrument-overview)
4. [Implementation Tasks](#4-implementation-tasks)
5. [Data Artifacts](#5-data-artifacts)
6. [Code Implementation](#6-code-implementation)
7. [Testing Strategy](#7-testing-strategy)
8. [Acceptance Criteria](#8-acceptance-criteria)
9. [Verification Commands](#9-verification-commands)

---

## 1. Executive Summary

### Goal
Implement a `SchwartzLoader` class that provides:
- Access to PVQ-21 questionnaire items (Portrait Values Questionnaire)
- Scoring of responses to produce value scores with **proper centering**
- Higher-order value dimension calculations
- Integration with the persona generation pipeline

### Scope
| In Scope | Out of Scope |
|----------|--------------|
| PVQ-21 (21 items, 10 values) | SVS-57 (Schwartz Value Survey, rating format) |
| Built-in item database (CC BY-NC-ND 3.0) | PVQ-40 (longer version) |
| Centered scoring (MRAT correction) | PVQ-RR (19 refined values version) |
| Higher-order dimensions (4 quadrants) | Normative percentile tables |
| 10 basic values structure | Cross-cultural validation |

### Key Differentiator
**Centered Scoring Required**: Unlike other personality instruments, Schwartz Values require centering scores by subtracting the individual's mean response (MRAT) to control for response style biases.

### Timeline Estimate
- **Data Preparation**: 1-2 hours (items from ESS documentation)
- **Loader Implementation**: 2-3 hours
- **Testing**: 2 hours
- **Documentation**: 1 hour

---

## 2. Why Schwartz Values for LM-VECTOR

### Beyond Personality: Measuring What Matters

While Big Five/HEXACO measure *how* someone behaves, Schwartz Values measure *why* — the underlying motivational goals that guide behavior.

| Framework | Measures | Example |
|-----------|----------|---------|
| Big Five | Personality traits | "I am organized" |
| HEXACO | Personality + ethics | "I avoid cheating" |
| O*NET Work Values | Job preferences | "I want job security" |
| **Schwartz Values** | Core life values | "Security is important to me" |

### Comparison with O*NET Work Values

We already have 6 Work Values from O*NET. Schwartz provides a deeper, more nuanced framework:

| O*NET Work Value | Nearest Schwartz Value(s) | Key Difference |
|------------------|---------------------------|----------------|
| Achievement | Achievement (AC) | Schwartz is broader (not just work) |
| Independence | Self-Direction (SE) | Schwartz includes creativity, exploration |
| Recognition | Power (PO), Achievement (AC) | Schwartz separates social status from competence |
| Relationships | Benevolence (BE) | Schwartz focuses on close others specifically |
| Support | Security (SC) | Schwartz is broader (national, personal, family) |
| Working Conditions | Security (SC), Hedonism (HE) | O*NET is work-specific |

### The Value Circumplex Model

Schwartz's theory provides a **circumplex structure** where:
- **Adjacent values** are compatible (e.g., Power and Achievement)
- **Opposite values** conflict (e.g., Self-Enhancement vs Self-Transcendence)

```
                    Openness to Change
                          ↑
                     Self-Direction
                    /             \
               Stimulation      Universalism
                  |                  |
              Hedonism           Benevolence
                  |                  |
             Achievement          Tradition
                    \             /
                       Power — Conformity — Security
                          ↓
                      Conservation
```

### Use Cases in LM-VECTOR

1. **Value-Aligned Personas**: Create personas with specific value priorities (e.g., high Benevolence + low Power)
2. **Conflict Scenarios**: Test model behavior when values conflict (Self-Enhancement vs Self-Transcendence)
3. **Cultural Variation**: Values vary across cultures; useful for diverse persona generation
4. **Ethical Decision-Making**: Values predict behavior in moral dilemmas

---

## 3. Instrument Overview

### PVQ-21 Structure

The Portrait Values Questionnaire (PVQ) presents **portrait descriptions** of people and asks respondents how similar each person is to themselves.

```
Schwartz Values (10 basic values, 4 higher-order dimensions)
│
├── Openness to Change
│   ├── SE: Self-Direction (2 items: 1, 11)
│   ├── ST: Stimulation (2 items: 6, 15)
│   └── HE: Hedonism* (2 items: 10, 21)
│
├── Self-Enhancement
│   ├── HE: Hedonism* (shared with Openness)
│   ├── AC: Achievement (2 items: 4, 13)
│   └── PO: Power (2 items: 2, 17)
│
├── Conservation
│   ├── SC: Security (2 items: 5, 14)
│   ├── CO: Conformity (2 items: 7, 16)
│   └── TR: Tradition (2 items: 9, 20)
│
└── Self-Transcendence
    ├── BE: Benevolence (2 items: 12, 18)
    └── UN: Universalism (3 items: 3, 8, 19)

*Hedonism is shared between Openness to Change and Self-Enhancement
```

### Item Distribution

| Value Code | Value Name | Items | Item IDs |
|------------|------------|-------|----------|
| SE | Self-Direction | 2 | 1, 11 |
| ST | Stimulation | 2 | 6, 15 |
| HE | Hedonism | 2 | 10, 21 |
| AC | Achievement | 2 | 4, 13 |
| PO | Power | 2 | 2, 17 |
| SC | Security | 2 | 5, 14 |
| CO | Conformity | 2 | 7, 16 |
| TR | Tradition | 2 | 9, 20 |
| BE | Benevolence | 2 | 12, 18 |
| UN | Universalism | 3 | 3, 8, 19 |
| **Total** | | **21** | |

### Response Scale

```
1 = Not like me at all
2 = Not like me
3 = A little like me
4 = Somewhat like me
5 = Like me
6 = Very much like me
```

### Scoring: The Critical Centering Step

**Raw scores are NOT interpretable!** Schwartz Values require **centered scores**:

```python
MRAT = mean(all 21 item responses)  # Individual's mean rating
centered_score[value] = raw_score[value] - MRAT
```

**Why centering?**
- Controls for **acquiescence bias** (some people rate everything high)
- Controls for **response style** differences
- Makes scores **comparable across individuals**

Example:
```
Person A: rates everything 5-6, raw Benevolence = 5.5
Person B: rates everything 2-3, raw Benevolence = 2.5

Without centering: A appears more benevolent than B
With centering: Both might have centered_BE = +0.5 (above their own mean)
```

---

## 4. Implementation Tasks

### Task 4.1: Create Data Directories

```bash
mkdir -p data/psychometrics/schwartz/{raw,items}
```

**Output Structure**:
```
data/psychometrics/schwartz/
├── items/
│   └── pvq_21.json             # Item database with scoring key
└── README.md                   # Data source documentation
```

---

### Task 4.2: Create Item Database JSON

The PVQ-21 items are from the European Social Survey (ESS) and are available under academic use terms.

**File: `data/psychometrics/schwartz/items/pvq_21.json`**

This file should contain:
- All 21 items with portrait text
- Value code for each item
- Response scale metadata
- Theoretical structure
- Citation information

See [Section 5.2](#52-item-database-schema) for the full schema.

---

### Task 4.3: Define Value Structure and Higher-Order Dimensions

The scoring key maps items to values and defines higher-order dimensions:

```python
SCHWARTZ_VALUE_STRUCTURE = {
    "SE": {
        "name": "Self-Direction",
        "definition": "Independent thought and action—choosing, creating, exploring",
        "items": [1, 11],
        "higher_order": "Openness to Change"
    },
    "ST": {
        "name": "Stimulation",
        "definition": "Excitement, novelty, and challenge in life",
        "items": [6, 15],
        "higher_order": "Openness to Change"
    },
    "HE": {
        "name": "Hedonism",
        "definition": "Pleasure or sensuous gratification for oneself",
        "items": [10, 21],
        "higher_order": ["Openness to Change", "Self-Enhancement"]  # Shared!
    },
    "AC": {
        "name": "Achievement",
        "definition": "Personal success through demonstrating competence",
        "items": [4, 13],
        "higher_order": "Self-Enhancement"
    },
    "PO": {
        "name": "Power",
        "definition": "Social status and prestige, control over resources",
        "items": [2, 17],
        "higher_order": "Self-Enhancement"
    },
    "SC": {
        "name": "Security",
        "definition": "Safety, harmony, and stability",
        "items": [5, 14],
        "higher_order": "Conservation"
    },
    "CO": {
        "name": "Conformity",
        "definition": "Restraint to avoid harming others or violating norms",
        "items": [7, 16],
        "higher_order": "Conservation"
    },
    "TR": {
        "name": "Tradition",
        "definition": "Respect for cultural and religious customs",
        "items": [9, 20],
        "higher_order": "Conservation"
    },
    "BE": {
        "name": "Benevolence",
        "definition": "Welfare of close others in everyday interaction",
        "items": [12, 18],
        "higher_order": "Self-Transcendence"
    },
    "UN": {
        "name": "Universalism",
        "definition": "Welfare of all people and nature",
        "items": [3, 8, 19],
        "higher_order": "Self-Transcendence"
    }
}

HIGHER_ORDER_DIMENSIONS = {
    "Openness to Change": ["SE", "ST", "HE"],
    "Self-Enhancement": ["HE", "AC", "PO"],
    "Conservation": ["SC", "CO", "TR"],
    "Self-Transcendence": ["BE", "UN"]
}
```

---

### Task 4.4: Document Data Sources

**File: `data/psychometrics/schwartz/README.md`**

```markdown
# Schwartz Portrait Values Questionnaire (PVQ-21)

## Source
European Social Survey (ESS) Human Values Module
https://www.europeansocialsurvey.org/

## License
CC BY-NC-ND 3.0 - Non-commercial academic use

## Citation
Schwartz, S. H. (2012). An overview of the Schwartz theory of basic values.
Online Readings in Psychology and Culture, 2(1).
https://doi.org/10.9707/2307-0919.1116

## Instrument
- **Name**: Portrait Values Questionnaire (PVQ-21)
- **Items**: 21
- **Values Measured**: 10 basic values
- **Response Scale**: 6-point similarity scale

## Critical Scoring Note
ALWAYS use centered scores for analysis:
- centered_score = raw_score - MRAT
- MRAT = individual's mean across all 21 items

See: Schwartz, S. H. (2003). Computing scores for the 10 human values.
```

---

## 5. Data Artifacts

### 5.1 Required Files

| Artifact | Location | Format |
|----------|----------|--------|
| Item database | `data/psychometrics/schwartz/items/pvq_21.json` | JSON |
| README | `data/psychometrics/schwartz/README.md` | Markdown |

### 5.2 Item Database Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["instrument", "version", "items", "value_structure", "response_scale"],
  "properties": {
    "instrument": {"type": "string"},
    "version": {"type": "string"},
    "source": {"type": "string"},
    "url": {"type": "string", "format": "uri"},
    "license": {"type": "string"},
    "citation": {"type": "string"},
    "response_scale": {
      "type": "object",
      "properties": {
        "type": {"const": "similarity"},
        "points": {"const": 6},
        "labels": {"type": "object"}
      }
    },
    "scoring_notes": {
      "type": "object",
      "properties": {
        "centering_required": {"type": "boolean"},
        "mrat_formula": {"type": "string"},
        "centered_formula": {"type": "string"}
      }
    },
    "value_structure": {
      "type": "object",
      "description": "10 basic values with item mappings"
    },
    "higher_order": {
      "type": "object",
      "description": "4 higher-order dimensions"
    },
    "circumplex_order": {
      "type": "array",
      "description": "Values in circumplex order"
    },
    "items": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["item_id", "text", "value_code", "value_name"],
        "properties": {
          "item_id": {"type": "integer", "minimum": 1, "maximum": 21},
          "text": {"type": "string", "minLength": 1},
          "value_code": {"type": "string", "pattern": "^[A-Z]{2}$"},
          "value_name": {"type": "string"}
        }
      }
    }
  }
}
```

---

## 6. Code Implementation

### 6.1 File: `src/pvx/data/schwartz_loader.py`

```python
"""
Schwartz Portrait Values Questionnaire (PVQ-21) Loader

Provides access to the PVQ-21 questionnaire items and scoring logic
for measuring Schwartz's 10 basic human values.

CRITICAL: Always use centered scores for analysis!

Usage:
    from pvx.data.schwartz_loader import SchwartzLoader

    loader = SchwartzLoader()
    items = loader.get_items()
    scores = loader.score_responses({1: 4, 2: 5, ...})  # Returns centered scores
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

# 10 Basic Values with metadata
SCHWARTZ_VALUES = {
    "SE": {
        "name": "Self-Direction",
        "definition": "Independent thought and action—choosing, creating, exploring",
        "items": [1, 11],
        "higher_order": "Openness to Change"
    },
    "ST": {
        "name": "Stimulation",
        "definition": "Excitement, novelty, and challenge in life",
        "items": [6, 15],
        "higher_order": "Openness to Change"
    },
    "HE": {
        "name": "Hedonism",
        "definition": "Pleasure or sensuous gratification for oneself",
        "items": [10, 21],
        "higher_order": "Openness to Change"  # Also Self-Enhancement
    },
    "AC": {
        "name": "Achievement",
        "definition": "Personal success through demonstrating competence",
        "items": [4, 13],
        "higher_order": "Self-Enhancement"
    },
    "PO": {
        "name": "Power",
        "definition": "Social status and prestige, control over resources",
        "items": [2, 17],
        "higher_order": "Self-Enhancement"
    },
    "SC": {
        "name": "Security",
        "definition": "Safety, harmony, and stability",
        "items": [5, 14],
        "higher_order": "Conservation"
    },
    "CO": {
        "name": "Conformity",
        "definition": "Restraint to avoid harming others or violating norms",
        "items": [7, 16],
        "higher_order": "Conservation"
    },
    "TR": {
        "name": "Tradition",
        "definition": "Respect for cultural and religious customs",
        "items": [9, 20],
        "higher_order": "Conservation"
    },
    "BE": {
        "name": "Benevolence",
        "definition": "Welfare of close others in everyday interaction",
        "items": [12, 18],
        "higher_order": "Self-Transcendence"
    },
    "UN": {
        "name": "Universalism",
        "definition": "Welfare of all people and nature",
        "items": [3, 8, 19],
        "higher_order": "Self-Transcendence"
    }
}

# Higher-order dimensions
HIGHER_ORDER_DIMENSIONS = {
    "Openness to Change": ["SE", "ST", "HE"],
    "Self-Enhancement": ["HE", "AC", "PO"],
    "Conservation": ["SC", "CO", "TR"],
    "Self-Transcendence": ["BE", "UN"]
}

# Circumplex order (adjacent values are compatible, opposite conflict)
CIRCUMPLEX_ORDER = ["SE", "ST", "HE", "AC", "PO", "SC", "CO", "TR", "BE", "UN"]


class SchwartzLoader:
    """Loader for Schwartz Portrait Values Questionnaire (PVQ-21)."""

    def __init__(self, data_dir: Path | str | None = None):
        """
        Initialize Schwartz Values loader.

        Args:
            data_dir: Path to psychometrics data directory.
                      Defaults to data/psychometrics/schwartz/
        """
        if data_dir is None:
            self.data_dir = Path("data/psychometrics/schwartz")
        else:
            self.data_dir = Path(data_dir)

        self._items_cache: dict | None = None

    def _load_items(self) -> dict:
        """Load item database from JSON file."""
        if self._items_cache is not None:
            return self._items_cache

        filepath = self.data_dir / "items" / "pvq_21.json"
        if not filepath.exists():
            raise FileNotFoundError(
                f"PVQ-21 items not found at {filepath}. "
                "Run the data preparation workflow first."
            )

        with open(filepath) as f:
            data = json.load(f)

        self._items_cache = data
        return data

    def get_items(self) -> list[dict]:
        """
        Get all questionnaire items.

        Returns:
            List of item dictionaries with keys:
            item_id, text, value_code, value_name
        """
        data = self._load_items()
        return data.get("items", [])

    def get_item_text(self, item_id: int) -> str:
        """Get the portrait text for a specific item (1-21)."""
        items = self.get_items()
        for item in items:
            if item["item_id"] == item_id:
                return item["text"]
        raise ValueError(f"Item {item_id} not found in PVQ-21")

    def get_items_by_value(self, value_code: str) -> list[dict]:
        """
        Get all items for a specific value.

        Args:
            value_code: Two-letter value code (e.g., "SE", "BE", "UN")

        Returns:
            List of item dictionaries for that value
        """
        if value_code not in SCHWARTZ_VALUES:
            raise ValueError(
                f"Unknown value code: {value_code}. "
                f"Valid codes: {list(SCHWARTZ_VALUES.keys())}"
            )
        items = self.get_items()
        return [item for item in items if item["value_code"] == value_code]

    def get_value_names(self) -> dict[str, str]:
        """Get mapping of value codes to full names."""
        return {code: info["name"] for code, info in SCHWARTZ_VALUES.items()}

    def get_value_definition(self, value_code: str) -> str:
        """Get the definition for a specific value."""
        if value_code not in SCHWARTZ_VALUES:
            raise ValueError(f"Unknown value code: {value_code}")
        return SCHWARTZ_VALUES[value_code]["definition"]

    def get_higher_order_dimension(self, value_code: str) -> str:
        """Get the higher-order dimension for a value."""
        if value_code not in SCHWARTZ_VALUES:
            raise ValueError(f"Unknown value code: {value_code}")
        return SCHWARTZ_VALUES[value_code]["higher_order"]

    def get_circumplex_order(self) -> list[str]:
        """Get values in circumplex order (for plotting)."""
        return CIRCUMPLEX_ORDER.copy()

    def score_responses(
        self,
        responses: dict[int, int],
        centered: bool = True
    ) -> dict:
        """
        Score PVQ-21 responses.

        IMPORTANT: By default, returns centered scores which are required
        for valid analysis. Set centered=False only for debugging.

        Args:
            responses: Dict mapping item_id (1-21) to response (1-6)
            centered: Whether to compute centered scores (STRONGLY RECOMMENDED)

        Returns:
            Dict with:
                - raw_scores: {value_code: mean_raw_score}
                - mrat: Mean Rating Across all items (individual's mean)
                - centered_scores: {value_code: centered_score} (if centered=True)
                - higher_order: {dimension: score} (if centered=True)
                - items_scored: number of items with responses
        """
        # Calculate raw scores per value
        raw_scores: dict[str, float] = {}
        all_responses: list[int] = []

        for value_code, value_info in SCHWARTZ_VALUES.items():
            item_ids = value_info["items"]
            scores = [responses[i] for i in item_ids if i in responses]

            if scores:
                raw_scores[value_code] = sum(scores) / len(scores)
                all_responses.extend(scores)

        # Calculate MRAT (Mean Rating Across all items)
        mrat = sum(all_responses) / len(all_responses) if all_responses else 0.0

        results = {
            "raw_scores": {k: round(v, 3) for k, v in raw_scores.items()},
            "mrat": round(mrat, 3),
            "items_scored": len(all_responses),
            "total_items": 21
        }

        # Calculate centered scores
        if centered:
            centered_scores = {}
            for value_code, raw_score in raw_scores.items():
                centered_scores[value_code] = round(raw_score - mrat, 3)

            results["centered_scores"] = centered_scores

            # Calculate higher-order dimension scores
            higher_order_scores = self._calculate_higher_order(centered_scores)
            results["higher_order"] = higher_order_scores

        return results

    def _calculate_higher_order(self, centered_scores: dict[str, float]) -> dict[str, float]:
        """Calculate higher-order dimension scores from centered value scores."""
        higher_order = {}

        for dimension, values in HIGHER_ORDER_DIMENSIONS.items():
            dim_scores = [centered_scores.get(v, 0.0) for v in values if v in centered_scores]
            if dim_scores:
                higher_order[dimension] = round(sum(dim_scores) / len(dim_scores), 3)

        return higher_order

    def get_value_profile(self, scores: dict, use_centered: bool = True) -> str:
        """
        Generate a text interpretation of value priorities.

        Args:
            scores: Output from score_responses()
            use_centered: Use centered scores (recommended)

        Returns:
            Text representation of value priorities from highest to lowest
        """
        score_key = "centered_scores" if use_centered else "raw_scores"
        value_scores = scores.get(score_key, {})

        if not value_scores:
            return "No scores available"

        # Sort by score (highest first)
        sorted_values = sorted(value_scores.items(), key=lambda x: x[1], reverse=True)
        names = self.get_value_names()

        lines = ["Value Priorities (highest to lowest):", ""]
        for rank, (code, score) in enumerate(sorted_values, 1):
            name = names.get(code, code)
            sign = "+" if score > 0 else ""
            lines.append(f"  {rank:2d}. {name:<15} ({code}): {sign}{score:.2f}")

        # Add interpretation guidance
        lines.extend([
            "",
            "Interpretation:",
            "  Positive scores = Above personal average (prioritized)",
            "  Negative scores = Below personal average (de-prioritized)",
            "  Scores near 0   = Average importance for this person"
        ])

        return "\n".join(lines)

    def compare_with_onet_work_values(
        self,
        schwartz_scores: dict[str, float],
        onet_work_values: dict[str, float]
    ) -> dict:
        """
        Compare Schwartz Values with O*NET Work Values.

        Args:
            schwartz_scores: Centered scores from score_responses()["centered_scores"]
            onet_work_values: Work value scores from ONETLoader.get_work_value_scores()

        Returns:
            Dict mapping O*NET work values to related Schwartz values with scores
        """
        # Mapping O*NET Work Values to nearest Schwartz Values
        mapping = {
            "achievement": ["AC"],
            "independence": ["SE"],
            "recognition": ["PO", "AC"],
            "relationships": ["BE"],
            "support": ["SC"],
            "working_conditions": ["SC", "HE"]
        }

        comparison = {}
        for onet_value, schwartz_values in mapping.items():
            if onet_value in onet_work_values:
                schwartz_data = [
                    {
                        "value": sv,
                        "name": SCHWARTZ_VALUES[sv]["name"],
                        "score": schwartz_scores.get(sv, 0.0)
                    }
                    for sv in schwartz_values
                    if sv in schwartz_scores
                ]
                comparison[onet_value] = {
                    "onet_score": onet_work_values[onet_value],
                    "related_schwartz_values": schwartz_data
                }

        return comparison

    def get_response_scale(self) -> dict:
        """Get the response scale metadata."""
        data = self._load_items()
        return data.get("response_scale", {
            "type": "similarity",
            "points": 6,
            "labels": {
                "1": "Not like me at all",
                "2": "Not like me",
                "3": "A little like me",
                "4": "Somewhat like me",
                "5": "Like me",
                "6": "Very much like me"
            }
        })

    def get_citation(self) -> str:
        """Get the citation for the instrument."""
        data = self._load_items()
        return data.get(
            "citation",
            "Schwartz, S. H. (2012). An overview of the Schwartz theory of "
            "basic values. Online Readings in Psychology and Culture, 2(1)."
        )

    def get_license(self) -> str:
        """Get the license information."""
        return "CC BY-NC-ND 3.0 - Non-commercial academic use"
```

### 6.2 Integration with `__init__.py`

**Update: `src/pvx/data/__init__.py`**

```python
from pvx.data.onet_loader import ONETLoader
from pvx.data.schwartz_loader import SchwartzLoader

__all__ = ["ONETLoader", "SchwartzLoader"]
```

---

## 7. Testing Strategy

### 7.1 Test File: `tests/unit/test_schwartz_loader.py`

```python
"""Unit tests for SchwartzLoader."""

import pytest
from pathlib import Path
import json

from pvx.data.schwartz_loader import (
    SchwartzLoader,
    SCHWARTZ_VALUES,
    HIGHER_ORDER_DIMENSIONS,
    CIRCUMPLEX_ORDER
)


class TestSchwartzLoader:
    """Tests for SchwartzLoader class."""

    @pytest.fixture
    def loader(self, tmp_path: Path) -> SchwartzLoader:
        """Create loader with test data."""
        items_dir = tmp_path / "items"
        items_dir.mkdir()

        # Create test items matching PVQ-21 structure
        test_items = []
        item_id = 1
        for value_code, value_info in SCHWARTZ_VALUES.items():
            for _ in value_info["items"]:
                test_items.append({
                    "item_id": item_id,
                    "text": f"Test portrait {item_id} for {value_code}",
                    "value_code": value_code,
                    "value_name": value_info["name"]
                })
                item_id += 1

        test_data = {
            "instrument": "PVQ-21",
            "version": "21-item",
            "license": "CC BY-NC-ND 3.0",
            "response_scale": {
                "type": "similarity",
                "points": 6,
                "labels": {"1": "Not like me at all", "6": "Very much like me"}
            },
            "items": test_items
        }

        with open(items_dir / "pvq_21.json", "w") as f:
            json.dump(test_data, f)

        return SchwartzLoader(data_dir=tmp_path)

    def test_get_items_count(self, loader: SchwartzLoader):
        """Test that we get all 21 items."""
        items = loader.get_items()
        assert len(items) == 21

    def test_get_items_structure(self, loader: SchwartzLoader):
        """Test item structure has required fields."""
        items = loader.get_items()
        required_fields = ["item_id", "text", "value_code", "value_name"]
        for item in items:
            for field in required_fields:
                assert field in item

    def test_get_item_text(self, loader: SchwartzLoader):
        """Test getting specific item text."""
        text = loader.get_item_text(1)
        assert "Test portrait 1" in text

    def test_get_item_text_not_found(self, loader: SchwartzLoader):
        """Test error for invalid item number."""
        with pytest.raises(ValueError, match="Item 99 not found"):
            loader.get_item_text(99)

    def test_get_items_by_value(self, loader: SchwartzLoader):
        """Test getting items for a value."""
        items = loader.get_items_by_value("UN")
        assert len(items) == 3  # Universalism has 3 items
        assert all(item["value_code"] == "UN" for item in items)

    def test_get_items_by_value_invalid(self, loader: SchwartzLoader):
        """Test error for invalid value code."""
        with pytest.raises(ValueError, match="Unknown value code"):
            loader.get_items_by_value("XX")

    def test_get_value_names(self, loader: SchwartzLoader):
        """Test value name mapping."""
        names = loader.get_value_names()
        assert names["SE"] == "Self-Direction"
        assert names["BE"] == "Benevolence"
        assert names["UN"] == "Universalism"
        assert len(names) == 10

    def test_get_value_definition(self, loader: SchwartzLoader):
        """Test getting value definition."""
        definition = loader.get_value_definition("BE")
        assert "welfare" in definition.lower()
        assert "close others" in definition.lower()

    def test_get_higher_order_dimension(self, loader: SchwartzLoader):
        """Test getting higher-order dimension for value."""
        assert loader.get_higher_order_dimension("SE") == "Openness to Change"
        assert loader.get_higher_order_dimension("BE") == "Self-Transcendence"
        assert loader.get_higher_order_dimension("SC") == "Conservation"

    def test_get_circumplex_order(self, loader: SchwartzLoader):
        """Test circumplex order."""
        order = loader.get_circumplex_order()
        assert len(order) == 10
        assert order == CIRCUMPLEX_ORDER

    def test_score_responses_centered(self, loader: SchwartzLoader):
        """Test scoring with centering (default)."""
        # All responses = 4, so MRAT = 4
        responses = {i: 4 for i in range(1, 22)}
        scores = loader.score_responses(responses)

        assert "raw_scores" in scores
        assert "mrat" in scores
        assert "centered_scores" in scores
        assert "higher_order" in scores
        assert scores["items_scored"] == 21
        assert scores["mrat"] == 4.0

        # All centered scores should be ~0 since everyone rated 4
        for value, centered in scores["centered_scores"].items():
            assert abs(centered) < 0.01, f"{value} should be ~0"

    def test_score_responses_without_centering(self, loader: SchwartzLoader):
        """Test scoring without centering."""
        responses = {i: 4 for i in range(1, 22)}
        scores = loader.score_responses(responses, centered=False)

        assert "raw_scores" in scores
        assert "centered_scores" not in scores
        assert "higher_order" not in scores

    def test_score_responses_varied(self, loader: SchwartzLoader):
        """Test scoring with varied responses."""
        # Give high scores to Benevolence items (12, 18) and low to Power (2, 17)
        responses = {i: 3 for i in range(1, 22)}  # Baseline
        responses[12] = 6  # BE item
        responses[18] = 6  # BE item
        responses[2] = 1   # PO item
        responses[17] = 1  # PO item

        scores = loader.score_responses(responses)

        # BE should have positive centered score
        assert scores["centered_scores"]["BE"] > 0

        # PO should have negative centered score
        assert scores["centered_scores"]["PO"] < 0

    def test_score_responses_partial(self, loader: SchwartzLoader):
        """Test scoring with partial responses."""
        responses = {1: 4, 2: 5, 3: 3}  # Only 3 items
        scores = loader.score_responses(responses)

        assert scores["items_scored"] == 3
        assert "SE" in scores["raw_scores"]  # Item 1 is SE
        assert "PO" in scores["raw_scores"]  # Item 2 is PO
        assert "UN" in scores["raw_scores"]  # Item 3 is UN

    def test_higher_order_calculation(self, loader: SchwartzLoader):
        """Test higher-order dimension calculation."""
        responses = {i: 4 for i in range(1, 22)}
        # Boost all Openness to Change values (SE: 1,11; ST: 6,15; HE: 10,21)
        for item_id in [1, 11, 6, 15, 10, 21]:
            responses[item_id] = 6

        scores = loader.score_responses(responses)

        # Openness to Change should be highest
        ho = scores["higher_order"]
        assert ho["Openness to Change"] > ho["Conservation"]
        assert ho["Openness to Change"] > ho["Self-Transcendence"]

    def test_get_value_profile(self, loader: SchwartzLoader):
        """Test value profile text generation."""
        responses = {i: 4 for i in range(1, 22)}
        responses[12] = 6  # Boost Benevolence
        responses[18] = 6

        scores = loader.score_responses(responses)
        profile = loader.get_value_profile(scores)

        assert "Benevolence" in profile
        assert "BE" in profile
        assert "Interpretation" in profile

    def test_get_license(self, loader: SchwartzLoader):
        """Test license returns correct value."""
        license_text = loader.get_license()
        assert "CC BY-NC-ND" in license_text


class TestSchwartzValueStructure:
    """Tests verifying Schwartz value structure correctness."""

    def test_all_values_have_items(self):
        """Verify each value has assigned items."""
        for code, info in SCHWARTZ_VALUES.items():
            assert "items" in info, f"Value {code} missing items"
            assert len(info["items"]) >= 2, f"Value {code} should have 2+ items"

    def test_universalism_has_three_items(self):
        """Verify Universalism (UN) has 3 items."""
        assert len(SCHWARTZ_VALUES["UN"]["items"]) == 3

    def test_total_items_is_21(self):
        """Verify total items across all values is 21."""
        total = sum(len(info["items"]) for info in SCHWARTZ_VALUES.values())
        assert total == 21

    def test_item_ids_are_valid(self):
        """Verify all item IDs are in range 1-21."""
        all_items = []
        for info in SCHWARTZ_VALUES.values():
            all_items.extend(info["items"])

        assert min(all_items) >= 1
        assert max(all_items) <= 21
        assert len(set(all_items)) == 21  # All unique

    def test_higher_order_dimensions(self):
        """Verify higher-order dimensions cover all values."""
        values_in_dims = set()
        for values in HIGHER_ORDER_DIMENSIONS.values():
            values_in_dims.update(values)

        # All values should appear (HE appears twice)
        for code in SCHWARTZ_VALUES:
            assert code in values_in_dims, f"Value {code} not in any dimension"

    def test_circumplex_contains_all_values(self):
        """Verify circumplex order has all 10 values."""
        assert len(CIRCUMPLEX_ORDER) == 10
        assert set(CIRCUMPLEX_ORDER) == set(SCHWARTZ_VALUES.keys())


class TestSchwartzCenteringLogic:
    """Tests specifically for centering logic correctness."""

    @pytest.fixture
    def simple_loader(self, tmp_path: Path) -> SchwartzLoader:
        """Create loader with minimal test data."""
        items_dir = tmp_path / "items"
        items_dir.mkdir()

        test_items = [
            {"item_id": i, "text": f"Item {i}", "value_code": "SE", "value_name": "Self-Direction"}
            for i in range(1, 22)
        ]

        test_data = {"instrument": "PVQ-21", "items": test_items}
        with open(items_dir / "pvq_21.json", "w") as f:
            json.dump(test_data, f)

        return SchwartzLoader(data_dir=tmp_path)

    def test_centering_with_uniform_responses(self, simple_loader: SchwartzLoader):
        """When all responses are same, centered scores should be 0."""
        for response_value in [1, 3, 4, 6]:
            responses = {i: response_value for i in range(1, 22)}
            scores = simple_loader.score_responses(responses)

            assert scores["mrat"] == response_value
            for centered in scores["centered_scores"].values():
                assert abs(centered) < 0.001

    def test_centering_math(self, simple_loader: SchwartzLoader):
        """Verify centering formula: centered = raw - MRAT."""
        responses = {i: 3 for i in range(1, 22)}
        responses[1] = 6  # Item 1 is SE

        scores = simple_loader.score_responses(responses)

        # MRAT should be slightly above 3
        assert scores["mrat"] > 3.0

        # SE raw should be higher than MRAT, so centered should be positive
        assert scores["raw_scores"]["SE"] > scores["mrat"]
        expected_centered = scores["raw_scores"]["SE"] - scores["mrat"]
        assert abs(scores["centered_scores"]["SE"] - expected_centered) < 0.001
```

### 7.2 Test Commands

```bash
# Run all Schwartz tests
uv run pytest tests/unit/test_schwartz_loader.py -v

# Run with coverage
uv run pytest tests/unit/test_schwartz_loader.py --cov=pvx.data.schwartz_loader --cov-report=term-missing

# Run specific test
uv run pytest tests/unit/test_schwartz_loader.py::TestSchwartzCenteringLogic -v
```

---

## 8. Acceptance Criteria

### 8.1 Data Artifacts

- [ ] `data/psychometrics/schwartz/items/pvq_21.json` contains all 21 items
- [ ] JSON file includes portrait text, value code, and value name for each item
- [ ] JSON file passes schema validation
- [ ] README documents data source, license, and centering requirement

### 8.2 Code Quality

- [ ] `SchwartzLoader` class implemented in `src/pvx/data/schwartz_loader.py`
- [ ] All public methods have docstrings
- [ ] Type hints on all parameters and return types
- [ ] No `ty` type checker errors

### 8.3 Functionality

- [ ] `get_items()` returns 21 items
- [ ] `score_responses()` computes MRAT correctly
- [ ] Centered scores sum to 0 for uniform responses
- [ ] Higher-order dimension scores computed from centered values
- [ ] `get_value_profile()` produces readable text output

### 8.4 Testing

- [ ] Unit tests cover all public methods
- [ ] Tests verify centering logic specifically
- [ ] Tests verify value structure (21 items, 10 values)
- [ ] All tests pass
- [ ] Test coverage > 80%

### 8.5 Integration

- [ ] Loader exported from `pvx.data` package
- [ ] Works alongside existing `ONETLoader`

---

## 9. Verification Commands

### Final Verification Script

```bash
#!/bin/bash
echo "=== Schwartz Values Phase 4 Verification ==="

echo -e "\n--- Data Files ---"
ls -la data/psychometrics/schwartz/
ls -la data/psychometrics/schwartz/items/

echo -e "\n--- Item Count Verification ---"
python3 -c "
import json
with open('data/psychometrics/schwartz/items/pvq_21.json') as f:
    data = json.load(f)
    print(f'PVQ-21: {len(data[\"items\"])} items')

    # Count by value
    values = {}
    for item in data['items']:
        v = item['value_code']
        values[v] = values.get(v, 0) + 1
    print(f'Items per value: {values}')
    print(f'Total values: {len(values)}')
"

echo -e "\n--- Loader Test ---"
python3 -c "
from pvx.data.schwartz_loader import SchwartzLoader

loader = SchwartzLoader()
items = loader.get_items()
print(f'Items loaded: {len(items)}')

# Test scoring with sample data
responses = {i: 4 for i in range(1, 22)}
responses[12] = 6  # Boost Benevolence
responses[18] = 6

scores = loader.score_responses(responses)
print(f'MRAT: {scores[\"mrat\"]}')
print(f'Centered scores: {scores[\"centered_scores\"]}')
print(f'Higher-order: {scores[\"higher_order\"]}')

# Show profile
print('\n' + loader.get_value_profile(scores))
"

echo -e "\n--- Unit Tests ---"
uv run pytest tests/unit/test_schwartz_loader.py -v --tb=short

echo -e "\n--- Type Check ---"
uv run ty check src/pvx/data/schwartz_loader.py

echo -e "\n=== Verification Complete ==="
```

---

## Related Documents

- [PSYCHOMETRICS_DATA.md § Phase 4](../reference/PSYCHOMETRICS_DATA.md#phase-4-schwartz-values) - Reference implementation
- [STATUS_psychometrics.md](./STATUS_psychometrics.md) - Progress tracking
- [WORKFLOW_phase2_hexaco.md](./WORKFLOW_phase2_hexaco.md) - Related workflow (HEXACO)
- [WORKFLOW_phase3_ipip_neo.md](./WORKFLOW_phase3_ipip_neo.md) - Related workflow (IPIP-NEO)
- [onet_loader.py](../../src/pvx/data/onet_loader.py) - Pattern reference

---

## Appendix A: Complete PVQ-21 Items

Source: European Social Survey (ESS) Human Values Module

| ID | Value | Portrait Text |
|----|-------|---------------|
| 1 | SE | Thinking up new ideas and being creative is important to him. He likes to do things in his own original way. |
| 2 | PO | It is important to him to be rich. He wants to have a lot of money and expensive things. |
| 3 | UN | He thinks it is important that every person in the world be treated equally. He believes everyone should have equal opportunities in life. |
| 4 | AC | It's important to him to show his abilities. He wants people to admire what he does. |
| 5 | SC | It is important to him to live in secure surroundings. He avoids anything that might endanger his safety. |
| 6 | ST | He likes surprises and is always looking for new things to do. He thinks it is important to do lots of different things in life. |
| 7 | CO | He believes that people should do what they're told. He thinks people should follow rules at all times, even when no-one is watching. |
| 8 | UN | It is important to him to listen to people who are different from him. Even when he disagrees with them, he still wants to understand them. |
| 9 | TR | It is important to him to be humble and modest. He tries not to draw attention to himself. |
| 10 | HE | Having a good time is important to him. He likes to "spoil" himself. |
| 11 | SE | It is important to him to make his own decisions about what he does. He likes to be free and not depend on others. |
| 12 | BE | It's very important to him to help the people around him. He wants to care for their well-being. |
| 13 | AC | Being very successful is important to him. He hopes people will recognise his achievements. |
| 14 | SC | It is important to him that the government ensures his safety against all threats. He wants the state to be strong so it can defend its citizens. |
| 15 | ST | He looks for adventures and likes to take risks. He wants to have an exciting life. |
| 16 | CO | It is important to him always to behave properly. He wants to avoid doing anything people would say is wrong. |
| 17 | PO | It is important to him to be in charge and tell others what to do. He wants people to do what he says. |
| 18 | BE | It is important to him to be loyal to his friends. He wants to devote himself to people close to him. |
| 19 | UN | He strongly believes that people should care for nature. Looking after the environment is important to him. |
| 20 | TR | Tradition is important to him. He tries to follow the customs handed down by his religion or his family. |
| 21 | HE | He seeks every chance he can to have fun. It is important to him to do things that give him pleasure. |

**Note**: Portrait texts use "he/him" pronouns as per ESS standard. Implementations may adapt for other pronouns.

---

## Appendix B: Scoring Key Summary

### Item-to-Value Mapping

| Value Code | Value Name | Item IDs | Item Count |
|------------|------------|----------|------------|
| SE | Self-Direction | 1, 11 | 2 |
| ST | Stimulation | 6, 15 | 2 |
| HE | Hedonism | 10, 21 | 2 |
| AC | Achievement | 4, 13 | 2 |
| PO | Power | 2, 17 | 2 |
| SC | Security | 5, 14 | 2 |
| CO | Conformity | 7, 16 | 2 |
| TR | Tradition | 9, 20 | 2 |
| BE | Benevolence | 12, 18 | 2 |
| UN | Universalism | 3, 8, 19 | 3 |

### Higher-Order Dimensions

| Dimension | Component Values | Motivational Focus |
|-----------|------------------|-------------------|
| Openness to Change | SE, ST, HE | Autonomy, novelty, pleasure |
| Self-Enhancement | HE, AC, PO | Personal success, status |
| Conservation | SC, CO, TR | Security, tradition, conformity |
| Self-Transcendence | BE, UN | Others' welfare, equality |

### Opposing Value Pairs

| Dimension 1 | Dimension 2 | Conflict |
|-------------|-------------|----------|
| Openness to Change | Conservation | Independence vs. order |
| Self-Enhancement | Self-Transcendence | Self-interest vs. others' welfare |
