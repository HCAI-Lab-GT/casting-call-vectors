# O*NET Psychometrics Mapping Workflow

## Purpose

This document provides a comprehensive workflow for leveraging **ALL O*NET data** to map occupational profiles to multiple psychometric frameworks including:

- **RIASEC** (Holland Codes) - Currently implemented
- **Big Five (OCEAN/FFM)** - Via Work Styles mapping
- **HEXACO** - Via Work Styles mapping
- **Values** (Schwartz-aligned) - Via Work Values

---

## Current State Analysis

### What ONETLoader Currently Parses

| Data Category | File | Status | Records |
|---------------|------|--------|---------|
| Occupations | `Occupation Data.txt` | ✅ Parsed | 1,016 |
| Interests (RIASEC) | `Interests.txt` | ✅ Parsed | 8,307 |
| Tasks | `Task Statements.txt` | ✅ Parsed | ~90,000 |

### What ONETLoader Does NOT Parse (Key Gaps)

| Data Category | File | Records | Psychometric Value |
|---------------|------|---------|-------------------|
| **Work Styles** | `Work Styles.txt` | 14,065 | **HIGH** - Maps to Big5/HEXACO |
| **Work Values** | `Work Values.txt` | 7,867 | **HIGH** - Maps to value theories |
| Abilities | `Abilities.txt` | 8.4M | Medium - Cognitive/physical capacities |
| Skills | `Skills.txt` | 5.6M | Medium - Competencies |
| Knowledge | `Knowledge.txt` | 5.5M | Medium - Domain knowledge |
| Work Context | `Work Context.txt` | 35M | Medium - Environment factors |
| Work Activities | `Work Activities.txt` | 8.6M | Medium - Activity descriptors |

---

## O*NET Work Styles → Big Five/HEXACO Mapping

The O*NET Work Styles (16 traits) map remarkably well to Big Five and HEXACO personality frameworks.

### Work Styles Structure

```
1.C Work Styles (16 traits in 7 groups)
├── 1.C.1 Achievement Orientation
│   ├── 1.C.1.a Achievement/Effort
│   ├── 1.C.1.b Persistence
│   └── 1.C.1.c Initiative
├── 1.C.2 Social Influence
│   └── 1.C.2.b Leadership
├── 1.C.3 Interpersonal Orientation
│   ├── 1.C.3.a Cooperation
│   ├── 1.C.3.b Concern for Others
│   └── 1.C.3.c Social Orientation
├── 1.C.4 Adjustment
│   ├── 1.C.4.a Self-Control
│   ├── 1.C.4.b Stress Tolerance
│   └── 1.C.4.c Adaptability/Flexibility
├── 1.C.5 Conscientiousness
│   ├── 1.C.5.a Dependability
│   ├── 1.C.5.b Attention to Detail
│   └── 1.C.5.c Integrity
├── 1.C.6 Independence
└── 1.C.7 Practical Intelligence
    ├── 1.C.7.a Innovation
    └── 1.C.7.b Analytical Thinking
```

### Big Five (OCEAN) Mapping

| Big Five Domain | O*NET Work Styles | Element IDs |
|-----------------|-------------------|-------------|
| **O**penness | Innovation, Analytical Thinking, Adaptability | 1.C.7.a, 1.C.7.b, 1.C.4.c |
| **C**onscientiousness | Dependability, Achievement/Effort, Persistence, Attention to Detail | 1.C.5.a, 1.C.1.a, 1.C.1.b, 1.C.5.b |
| **E**xtraversion | Leadership, Social Orientation, Initiative | 1.C.2.b, 1.C.3.c, 1.C.1.c |
| **A**greeableness | Cooperation, Concern for Others, Integrity | 1.C.3.a, 1.C.3.b, 1.C.5.c |
| **N**euroticism (inverse) | Stress Tolerance, Self-Control | 1.C.4.b, 1.C.4.a |

### HEXACO Mapping

| HEXACO Domain | O*NET Work Styles | Element IDs |
|---------------|-------------------|-------------|
| **H**onesty-Humility | Integrity | 1.C.5.c |
| **E**motionality | Stress Tolerance (inverse), Self-Control (inverse) | 1.C.4.b, 1.C.4.a |
| e**X**traversion | Leadership, Social Orientation | 1.C.2.b, 1.C.3.c |
| **A**greeableness | Cooperation, Concern for Others, Adaptability | 1.C.3.a, 1.C.3.b, 1.C.4.c |
| **C**onscientiousness | Dependability, Achievement/Effort, Persistence, Attention to Detail | 1.C.5.a, 1.C.1.a, 1.C.1.b, 1.C.5.b |
| **O**penness | Innovation, Analytical Thinking | 1.C.7.a, 1.C.7.b |

---

## O*NET Work Values → Value Theories Mapping

### Work Values Structure (6 values + 21 sub-needs)

```
1.B.2 Work Values
├── 1.B.2.a Achievement (Ability Utilization, Achievement)
├── 1.B.2.b Working Conditions (Activity, Independence, Variety, Compensation, Security, Working Conditions)
├── 1.B.2.c Recognition (Advancement, Authority, Recognition, Social Status)
├── 1.B.2.d Relationships (Co-workers, Social Service, Moral Values)
├── 1.B.2.e Support (Company Policies, Supervision: Human Relations, Supervision: Technical)
└── 1.B.2.f Independence (Creativity, Responsibility, Autonomy)
```

### Schwartz Values Alignment

| O*NET Work Value | Schwartz Values | Notes |
|------------------|-----------------|-------|
| Achievement | Achievement, Power | Self-enhancement |
| Independence | Self-Direction | Openness to change |
| Recognition | Power, Achievement | Self-enhancement |
| Relationships | Benevolence | Self-transcendence |
| Support | Security, Conformity | Conservation |
| Working Conditions | Security, Hedonism | Conservation/Openness |

---

## Workflow: Extend ONETLoader

### Step 1: Add Work Styles Parsing

```python
# Add to src/pvx/data/onet_loader.py

WORK_STYLE_ELEMENTS = {
    "1.C.1.a": "Achievement/Effort",
    "1.C.1.b": "Persistence",
    "1.C.1.c": "Initiative",
    "1.C.2.b": "Leadership",
    "1.C.3.a": "Cooperation",
    "1.C.3.b": "Concern for Others",
    "1.C.3.c": "Social Orientation",
    "1.C.4.a": "Self-Control",
    "1.C.4.b": "Stress Tolerance",
    "1.C.4.c": "Adaptability/Flexibility",
    "1.C.5.a": "Dependability",
    "1.C.5.b": "Attention to Detail",
    "1.C.5.c": "Integrity",
    "1.C.6": "Independence",
    "1.C.7.a": "Innovation",
    "1.C.7.b": "Analytical Thinking",
}

BIG_FIVE_MAPPING = {
    "O": ["1.C.7.a", "1.C.7.b", "1.C.4.c"],  # Openness
    "C": ["1.C.5.a", "1.C.1.a", "1.C.1.b", "1.C.5.b"],  # Conscientiousness
    "E": ["1.C.2.b", "1.C.3.c", "1.C.1.c"],  # Extraversion
    "A": ["1.C.3.a", "1.C.3.b", "1.C.5.c"],  # Agreeableness
    "N_inv": ["1.C.4.b", "1.C.4.a"],  # Neuroticism (inverse)
}

def load_work_styles(self) -> pd.DataFrame:
    """Load Work Styles data (16 personality-like traits)."""
    filepath = self.data_dir / "Work Styles.txt"
    df = pd.read_csv(filepath, sep="\t")
    df.columns = [
        "soc_code", "element_id", "element_name", "scale_id",
        "data_value", "n", "standard_error", "lower_ci",
        "upper_ci", "recommend_suppress", "date", "domain_source"
    ]
    return df

def get_work_style_scores(self) -> dict[str, dict[str, float]]:
    """Get Work Style scores (1-5 scale) for all occupations."""
    work_styles = self.load_work_styles()

    # Filter to IM scale (importance)
    im_data = work_styles[
        (work_styles["scale_id"] == "IM") &
        (work_styles["element_id"].isin(WORK_STYLE_ELEMENTS.keys()))
    ]

    result = {}
    for soc_code, group in im_data.groupby("soc_code"):
        scores = {}
        for _, row in group.iterrows():
            element = row["element_id"]
            name = WORK_STYLE_ELEMENTS.get(element, element)
            scores[name] = row["data_value"]
        result[soc_code] = scores

    return result

def get_big_five_scores(self) -> dict[str, dict[str, float]]:
    """Compute Big Five scores from Work Styles."""
    work_style_scores = self.get_work_style_scores()

    result = {}
    for soc_code, styles in work_style_scores.items():
        scores = {}
        for domain, elements in BIG_FIVE_MAPPING.items():
            domain_scores = []
            for elem_id in elements:
                name = WORK_STYLE_ELEMENTS.get(elem_id)
                if name and name in styles:
                    domain_scores.append(styles[name])
            if domain_scores:
                scores[domain] = sum(domain_scores) / len(domain_scores)
        result[soc_code] = scores

    return result
```

### Step 2: Add Work Values Parsing

```python
WORK_VALUE_ELEMENTS = {
    "1.B.2.a": "Achievement",
    "1.B.2.b": "Working Conditions",
    "1.B.2.c": "Recognition",
    "1.B.2.d": "Relationships",
    "1.B.2.e": "Support",
    "1.B.2.f": "Independence",
}

def load_work_values(self) -> pd.DataFrame:
    """Load Work Values data (6 values)."""
    filepath = self.data_dir / "Work Values.txt"
    df = pd.read_csv(filepath, sep="\t")
    df.columns = [
        "soc_code", "element_id", "element_name",
        "scale_id", "data_value", "date", "domain_source"
    ]
    return df

def get_work_value_scores(self) -> dict[str, dict[str, float]]:
    """Get Work Value scores (EX scale) for all occupations."""
    work_values = self.load_work_values()

    # Filter to EX scale (extent)
    ex_data = work_values[
        (work_values["scale_id"] == "EX") &
        (work_values["element_id"].isin(WORK_VALUE_ELEMENTS.keys()))
    ]

    result = {}
    for soc_code, group in ex_data.groupby("soc_code"):
        scores = {}
        for _, row in group.iterrows():
            element = row["element_id"]
            name = WORK_VALUE_ELEMENTS.get(element, element)
            scores[name] = row["data_value"]
        result[soc_code] = scores

    return result
```

### Step 3: Update Occupation Profile

```python
def get_occupation_profile(self, soc_code: str) -> dict:
    """Get complete profile for one occupation with ALL psychometrics."""
    # ... existing code ...

    # Add Work Styles
    work_style_scores = self.get_work_style_scores()
    work_styles = work_style_scores.get(soc_code, {})

    # Add Big Five derived scores
    big_five_scores = self.get_big_five_scores()
    big_five = big_five_scores.get(soc_code, {})

    # Add Work Values
    work_value_scores = self.get_work_value_scores()
    work_values = work_value_scores.get(soc_code, {})

    return {
        "soc_code": soc_code,
        "title": occ["title"],
        "description": occ["description"],
        # RIASEC (existing)
        "riasec": riasec,
        "riasec_primary": highpoints[0] if highpoints else None,
        "highpoint_codes": highpoints,
        # NEW: Work Styles
        "work_styles": work_styles,
        # NEW: Big Five (derived from Work Styles)
        "big_five": big_five,
        # NEW: Work Values
        "work_values": work_values,
        # Tasks
        "tasks": tasks,
    }
```

---

## Workflow: Update Persona Generator

After extending ONETLoader, update `VocationalPersonaGenerator` to include:

### Updated Persona JSON Schema

```json
{
  "instruction": [...],
  "eval_prompt": "...",
  "_metadata": {
    "soc_code": "15-1252.00",
    "title": "Software Developers",
    "riasec": {"R": 3.93, "I": 5.86, "A": 2.26, "S": 1.88, "E": 1.87, "C": 5.46},
    "riasec_primary": "I",
    "highpoint_codes": ["I", "C", "R"],
    "work_styles": {
      "Achievement/Effort": 4.2,
      "Persistence": 4.5,
      "Initiative": 4.3,
      "Analytical Thinking": 4.8,
      "Innovation": 4.1,
      "...": "..."
    },
    "big_five": {
      "O": 4.4,
      "C": 4.35,
      "E": 3.8,
      "A": 3.9,
      "N_inv": 4.2
    },
    "work_values": {
      "Achievement": 6.0,
      "Independence": 5.5,
      "Working Conditions": 5.2,
      "...": "..."
    }
  }
}
```

---

## Implementation Checklist

### Phase 1: Extend Data Loading

- [ ] Add `load_work_styles()` to ONETLoader
- [ ] Add `get_work_style_scores()` to ONETLoader
- [ ] Add `get_big_five_scores()` to ONETLoader (derived)
- [ ] Add `load_work_values()` to ONETLoader
- [ ] Add `get_work_value_scores()` to ONETLoader
- [ ] Update `get_occupation_profile()` to include new data
- [ ] Add validation tests for new loaders

### Phase 2: Update Persona Generation

- [ ] Update `VocationalPersonaGenerator` schema
- [ ] Regenerate personas with full psychometric data
- [ ] Validate persona JSON structure

### Phase 3: Analysis Extensions

- [ ] Add Big Five geometry analysis (like RIASEC)
- [ ] Add Work Styles correlation analysis
- [ ] Create cross-framework mapping visualizations

---

## Quick Validation Commands

```bash
# Test extended ONETLoader
uv run python -c "
from pvx.data.onet_loader import ONETLoader

loader = ONETLoader()

# Test Work Styles
print('=== Work Styles Sample ===')
ws = loader.get_work_style_scores()
sample = list(ws.items())[0]
print(f'{sample[0]}: {sample[1]}')

# Test Big Five
print('\\n=== Big Five Sample ===')
b5 = loader.get_big_five_scores()
sample = list(b5.items())[0]
print(f'{sample[0]}: {sample[1]}')

# Test Work Values
print('\\n=== Work Values Sample ===')
wv = loader.get_work_value_scores()
sample = list(wv.items())[0]
print(f'{sample[0]}: {sample[1]}')
"
```

---

## Data File Reference

### Files to Parse

| Priority | File | Size | Traits |
|----------|------|------|--------|
| **P0** | `Work Styles.txt` | 1.2M | 16 personality traits |
| **P0** | `Work Values.txt` | 495k | 6 values |
| P1 | `Abilities.txt` | 8.4M | 52 abilities |
| P1 | `Skills.txt` | 5.6M | 35 skills |
| P2 | `Knowledge.txt` | 5.5M | 33 knowledge areas |
| P2 | `Work Activities.txt` | 8.6M | 41 activities |
| P3 | `Work Context.txt` | 35M | 57 context factors |

### Element ID Patterns

| Pattern | Category | Example |
|---------|----------|---------|
| `1.A.*` | Abilities | `1.A.1.a.1` Oral Comprehension |
| `1.B.1.*` | Interests (RIASEC) | `1.B.1.a` Realistic |
| `1.B.2.*` | Work Values | `1.B.2.a` Achievement |
| `1.C.*` | Work Styles | `1.C.1.a` Achievement/Effort |
| `2.A.*` | Basic Skills | `2.A.1.a` Reading Comprehension |
| `2.B.*` | Cross-Functional Skills | `2.B.1.a` Social Perceptiveness |
| `2.C.*` | Knowledge | `2.C.1.a` Administration |
| `4.A.*` | Work Activities | GWAs and DWAs |
| `4.C.*` | Work Context | Environmental factors |

---

## Related Documents

- [PSYCHOMETRICS_DATA.md](./PSYCHOMETRICS_DATA.md) - Full psychometric instruments guide
- [PHASE1_IMPLEMENTATION_PLAN.md](./PHASE1_IMPLEMENTATION_PLAN.md) - Extraction pipeline architecture
- [experiments/phase1_riasec/README.md](../../experiments/phase1_riasec/README.md) - RIASEC experiment details
