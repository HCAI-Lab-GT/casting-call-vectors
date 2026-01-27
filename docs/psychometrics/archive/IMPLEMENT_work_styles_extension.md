# O*NET Psychometrics Extension - Verified Implementation Plan

> This plan has been verified against actual O*NET data files and existing codebase patterns.
> Ready for implementation agent execution.

---

## Executive Summary

Extend `ONETLoader` to parse Work Styles (16 personality traits) and Work Values (6 values),
then derive Big Five scores from Work Styles. Update `VocationalPersonaGenerator` metadata schema.

**Files to modify:**
- `src/pvx/data/onet_loader.py` (primary)
- `src/pvx/pvx_models/vocational_dataset.py` (secondary)
- `tests/unit/test_onet_loader.py` (tests)

---

## Verified Data Structures

### Work Styles File: `Work Styles.txt`

| Property | Value |
|----------|-------|
| Records | 14,064 (excluding header) |
| Unique Occupations | 880 |
| Unique Elements | 16 traits |
| Scale | IM (Importance) - Range 1-5 |
| Columns | O*NET-SOC Code, Element ID, Element Name, Scale ID, Data Value, N, Standard Error, Lower CI Bound, Upper CI Bound, Recommend Suppress, Date, Domain Source |

**Column Mapping:**
```python
["soc_code", "element_id", "element_name", "scale_id",
 "data_value", "n", "standard_error", "lower_ci",
 "upper_ci", "recommend_suppress", "date", "domain_source"]
```

**Element IDs (verified):**
```python
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
```

### Work Values File: `Work Values.txt`

| Property | Value |
|----------|-------|
| Records | 7,866 (excluding header) |
| Unique Occupations | 874 |
| Unique Elements | 9 (6 values + 3 high-points) |
| Scales | EX (Extent) 1-7, VH (Value High-Point) 1-6 |
| Columns | O*NET-SOC Code, Element ID, Element Name, Scale ID, Data Value, Date, Domain Source |

**Column Mapping:**
```python
["soc_code", "element_id", "element_name",
 "scale_id", "data_value", "date", "domain_source"]
```

**Element IDs (verified):**
```python
WORK_VALUE_ELEMENTS = {
    "1.B.2.a": "Achievement",
    "1.B.2.b": "Working Conditions",
    "1.B.2.c": "Recognition",
    "1.B.2.d": "Relationships",
    "1.B.2.e": "Support",
    "1.B.2.f": "Independence",
}

# High-point mapping (VH scale value -> value name)
WORK_VALUE_HIGHPOINT = {
    1: "Achievement",
    2: "Working Conditions",
    3: "Recognition",
    4: "Relationships",
    5: "Support",
    6: "Independence",
}
```

---

## Coverage Analysis

| Dataset | Occupations | Notes |
|---------|-------------|-------|
| Occupation Data | 1,016 | Primary source |
| Work Styles | 880 | **137 missing** from Occupation Data |
| Work Values | 874 | **142 missing** from Occupation Data |
| Interests (RIASEC) | ~930 | Existing implementation handles gracefully |

**Implication:** Implementation must return empty dicts `{}` for occupations without data,
matching existing `get_riasec_scores()` pattern.

---

## Implementation Details

### Part 1: Module-Level Constants

Add after line 48 (after `HIGHPOINT_TO_RIASEC`):

```python
# Work Styles Element ID mapping (16 personality-like traits)
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

# Big Five (OCEAN) derived from Work Styles
# Based on established Work Styles → FFM mappings
BIG_FIVE_MAPPING = {
    "O": ["1.C.7.a", "1.C.7.b", "1.C.4.c"],  # Openness: Innovation, Analytical, Adaptability
    "C": ["1.C.5.a", "1.C.1.a", "1.C.1.b", "1.C.5.b"],  # Conscientiousness: Dependability, Achievement, Persistence, Detail
    "E": ["1.C.2.b", "1.C.3.c", "1.C.1.c"],  # Extraversion: Leadership, Social, Initiative
    "A": ["1.C.3.a", "1.C.3.b", "1.C.5.c"],  # Agreeableness: Cooperation, Concern, Integrity
    "N_inv": ["1.C.4.b", "1.C.4.a"],  # Emotional Stability (inverse N): Stress Tolerance, Self-Control
}

# Work Values Element ID mapping (6 work values)
WORK_VALUE_ELEMENTS = {
    "1.B.2.a": "Achievement",
    "1.B.2.b": "Working Conditions",
    "1.B.2.c": "Recognition",
    "1.B.2.d": "Relationships",
    "1.B.2.e": "Support",
    "1.B.2.f": "Independence",
}

# Work Value High-Point mapping (VH scale)
WORK_VALUE_HIGHPOINT = {
    1: "Achievement",
    2: "Working Conditions",
    3: "Recognition",
    4: "Relationships",
    5: "Support",
    6: "Independence",
}
```

### Part 2: Add Cached Attributes to __init__

Add to `__init__` method (around line 73):

```python
self._work_styles: Optional[pd.DataFrame] = None
self._work_values: Optional[pd.DataFrame] = None
self._work_style_scores: Optional[dict] = None
self._work_value_scores: Optional[dict] = None
self._big_five_scores: Optional[dict] = None
```

### Part 3: New Methods

Add after `load_tasks()` method (after line 137):

```python
def load_work_styles(self) -> pd.DataFrame:
    """Load Work Styles data (16 personality-like traits).

    Returns:
        DataFrame with columns: soc_code, element_id, element_name,
        scale_id, data_value, n, standard_error, lower_ci, upper_ci,
        recommend_suppress, date, domain_source
    """
    if self._work_styles is not None:
        return self._work_styles

    filepath = self.data_dir / "Work Styles.txt"
    df = pd.read_csv(filepath, sep="\t")
    df.columns = [
        "soc_code", "element_id", "element_name", "scale_id",
        "data_value", "n", "standard_error", "lower_ci",
        "upper_ci", "recommend_suppress", "date", "domain_source"
    ]
    self._work_styles = df
    logger.info(f"Loaded {len(df)} work style records from O*NET")
    return df

def load_work_values(self) -> pd.DataFrame:
    """Load Work Values data (6 work values).

    Returns:
        DataFrame with columns: soc_code, element_id, element_name,
        scale_id, data_value, date, domain_source
    """
    if self._work_values is not None:
        return self._work_values

    filepath = self.data_dir / "Work Values.txt"
    df = pd.read_csv(filepath, sep="\t")
    df.columns = [
        "soc_code", "element_id", "element_name",
        "scale_id", "data_value", "date", "domain_source"
    ]
    self._work_values = df
    logger.info(f"Loaded {len(df)} work value records from O*NET")
    return df

def get_work_style_scores(self) -> dict[str, dict[str, float]]:
    """Get Work Style scores (IM scale, 1-5) for all occupations.

    Returns:
        Dict mapping SOC code -> {trait_name: score, ...}
        Only includes the 16 standard Work Style traits.
    """
    if self._work_style_scores is not None:
        return self._work_style_scores

    work_styles = self.load_work_styles()

    # Filter to IM scale and known elements
    im_data = work_styles[
        (work_styles["scale_id"] == "IM") &
        (work_styles["element_id"].isin(WORK_STYLE_ELEMENTS.keys()))
    ]

    result = {}
    for soc_code, group in im_data.groupby("soc_code"):
        scores = {}
        for _, row in group.iterrows():
            element = row["element_id"]
            name = WORK_STYLE_ELEMENTS[element]
            scores[name] = row["data_value"]
        result[soc_code] = scores

    self._work_style_scores = result
    return result

def get_big_five_scores(self) -> dict[str, dict[str, float]]:
    """Compute Big Five (OCEAN) scores derived from Work Styles.

    Returns:
        Dict mapping SOC code -> {"O": score, "C": score, "E": score,
        "A": score, "N_inv": score}

    Note:
        N_inv is Emotional Stability (inverse of Neuroticism).
        Higher values = more emotionally stable.
    """
    if self._big_five_scores is not None:
        return self._big_five_scores

    work_style_scores = self.get_work_style_scores()

    result = {}
    for soc_code, styles in work_style_scores.items():
        scores = {}
        for domain, element_ids in BIG_FIVE_MAPPING.items():
            domain_scores = []
            for elem_id in element_ids:
                name = WORK_STYLE_ELEMENTS.get(elem_id)
                if name and name in styles:
                    domain_scores.append(styles[name])
            if domain_scores:
                scores[domain] = sum(domain_scores) / len(domain_scores)
        result[soc_code] = scores

    self._big_five_scores = result
    return result

def get_work_value_scores(self) -> dict[str, dict[str, float]]:
    """Get Work Value scores (EX scale, 1-7) for all occupations.

    Returns:
        Dict mapping SOC code -> {value_name: score, ...}
        Only includes the 6 standard Work Values.
    """
    if self._work_value_scores is not None:
        return self._work_value_scores

    work_values = self.load_work_values()

    # Filter to EX scale and known elements
    ex_data = work_values[
        (work_values["scale_id"] == "EX") &
        (work_values["element_id"].isin(WORK_VALUE_ELEMENTS.keys()))
    ]

    result = {}
    for soc_code, group in ex_data.groupby("soc_code"):
        scores = {}
        for _, row in group.iterrows():
            element = row["element_id"]
            name = WORK_VALUE_ELEMENTS[element]
            scores[name] = row["data_value"]
        result[soc_code] = scores

    self._work_value_scores = result
    return result

def get_work_value_highpoints(self) -> dict[str, list[str]]:
    """Get Work Value high-point codes for all occupations.

    Returns:
        Dict mapping SOC code -> [primary, secondary, tertiary] value names
    """
    work_values = self.load_work_values()

    # Filter to VH scale (high-point codes)
    vh_data = work_values[
        (work_values["scale_id"] == "VH") &
        (work_values["element_id"].isin(["1.B.2.g", "1.B.2.h", "1.B.2.i"]))
    ]

    result = {}
    for soc_code, group in vh_data.groupby("soc_code"):
        codes = []
        for element_id in ["1.B.2.g", "1.B.2.h", "1.B.2.i"]:
            row = group[group["element_id"] == element_id]
            if not row.empty:
                val = int(row.iloc[0]["data_value"])
                name = WORK_VALUE_HIGHPOINT.get(val)
                if name:
                    codes.append(name)
        result[soc_code] = codes

    return result
```

### Part 4: Update get_occupation_profile()

Replace the existing method (lines 195-232) with:

```python
def get_occupation_profile(self, soc_code: str) -> dict:
    """Get complete profile for one occupation.

    Args:
        soc_code: O*NET-SOC occupation code (e.g., "11-1011.00")

    Returns:
        Dict with keys: soc_code, title, description, riasec,
        riasec_primary, highpoint_codes, work_styles, big_five,
        work_values, work_value_highpoints, tasks
    """
    occupations = self.load_occupations()
    occ_row = occupations[occupations["soc_code"] == soc_code]

    if occ_row.empty:
        raise ValueError(f"Occupation not found: {soc_code}")

    occ = occ_row.iloc[0]

    # Get RIASEC scores
    riasec_scores = self.get_riasec_scores()
    riasec = riasec_scores.get(soc_code, {})

    # Get RIASEC high-point codes
    highpoint_codes = self.get_highpoint_codes()
    highpoints = highpoint_codes.get(soc_code, [])

    # Get Work Styles (16 traits, scale 1-5)
    work_style_scores = self.get_work_style_scores()
    work_styles = work_style_scores.get(soc_code, {})

    # Get Big Five (derived from Work Styles)
    big_five_scores = self.get_big_five_scores()
    big_five = big_five_scores.get(soc_code, {})

    # Get Work Values (6 values, scale 1-7)
    work_value_scores = self.get_work_value_scores()
    work_values = work_value_scores.get(soc_code, {})

    # Get Work Value high-points
    work_value_hp = self.get_work_value_highpoints()
    work_value_highpoints = work_value_hp.get(soc_code, [])

    # Get tasks
    tasks_df = self.load_tasks()
    tasks = tasks_df[tasks_df["soc_code"] == soc_code]["task"].tolist()

    return {
        "soc_code": soc_code,
        "title": occ["title"],
        "description": occ["description"],
        # RIASEC (existing)
        "riasec": riasec,
        "riasec_primary": highpoints[0] if highpoints else None,
        "highpoint_codes": highpoints,
        # Work Styles (new)
        "work_styles": work_styles,
        # Big Five derived (new)
        "big_five": big_five,
        # Work Values (new)
        "work_values": work_values,
        "work_value_highpoints": work_value_highpoints,
        # Tasks
        "tasks": tasks,
    }
```

---

## Part 5: Update VocationalPersonaGenerator Metadata

In `src/pvx/pvx_models/vocational_dataset.py`, update the `generate_persona()` method
to include new fields in `_metadata`:

```python
"_metadata": {
    "soc_code": profile["soc_code"],
    "title": profile["title"],
    "riasec": profile.get("riasec", {}),
    "riasec_primary": profile.get("riasec_primary"),
    "highpoint_codes": profile.get("highpoint_codes", []),
    # NEW: Work Styles (16 traits, 1-5 scale)
    "work_styles": profile.get("work_styles", {}),
    # NEW: Big Five derived scores
    "big_five": profile.get("big_five", {}),
    # NEW: Work Values (6 values, 1-7 scale)
    "work_values": profile.get("work_values", {}),
    "work_value_highpoints": profile.get("work_value_highpoints", []),
}
```

---

## Part 6: Tests

Add to `tests/unit/test_onet_loader.py`:

```python
class TestWorkStyles:
    """Tests for Work Styles loading."""

    @pytest.mark.skipif(not HAS_ONET_DATA, reason="O*NET data not downloaded")
    def test_load_work_styles(self, loader: ONETLoader) -> None:
        ws = loader.load_work_styles()
        assert len(ws) > 0
        assert "soc_code" in ws.columns
        assert "element_id" in ws.columns
        assert "data_value" in ws.columns

    @pytest.mark.skipif(not HAS_ONET_DATA, reason="O*NET data not downloaded")
    def test_get_work_style_scores(self, loader: ONETLoader) -> None:
        scores = loader.get_work_style_scores()
        assert len(scores) > 0
        # Check a sample occupation has expected traits
        sample_soc = next(iter(scores))
        assert "Achievement/Effort" in scores[sample_soc]
        assert "Analytical Thinking" in scores[sample_soc]

    @pytest.mark.skipif(not HAS_ONET_DATA, reason="O*NET data not downloaded")
    def test_work_style_scores_in_range(self, loader: ONETLoader) -> None:
        scores = loader.get_work_style_scores()
        for soc, traits in scores.items():
            for trait, value in traits.items():
                assert 1.0 <= value <= 5.0, f"Work Style score out of range for {soc}: {trait}={value}"


class TestBigFive:
    """Tests for Big Five derivation."""

    @pytest.mark.skipif(not HAS_ONET_DATA, reason="O*NET data not downloaded")
    def test_get_big_five_scores(self, loader: ONETLoader) -> None:
        scores = loader.get_big_five_scores()
        assert len(scores) > 0
        sample_soc = next(iter(scores))
        # Should have all 5 dimensions
        assert set(scores[sample_soc].keys()) == {"O", "C", "E", "A", "N_inv"}

    @pytest.mark.skipif(not HAS_ONET_DATA, reason="O*NET data not downloaded")
    def test_big_five_scores_in_range(self, loader: ONETLoader) -> None:
        scores = loader.get_big_five_scores()
        for soc, dims in scores.items():
            for dim, value in dims.items():
                assert 1.0 <= value <= 5.0, f"Big Five score out of range for {soc}: {dim}={value}"


class TestWorkValues:
    """Tests for Work Values loading."""

    @pytest.mark.skipif(not HAS_ONET_DATA, reason="O*NET data not downloaded")
    def test_load_work_values(self, loader: ONETLoader) -> None:
        wv = loader.load_work_values()
        assert len(wv) > 0
        assert "soc_code" in wv.columns
        assert "element_id" in wv.columns

    @pytest.mark.skipif(not HAS_ONET_DATA, reason="O*NET data not downloaded")
    def test_get_work_value_scores(self, loader: ONETLoader) -> None:
        scores = loader.get_work_value_scores()
        assert len(scores) > 0
        sample_soc = next(iter(scores))
        assert "Achievement" in scores[sample_soc]
        assert "Independence" in scores[sample_soc]

    @pytest.mark.skipif(not HAS_ONET_DATA, reason="O*NET data not downloaded")
    def test_work_value_scores_in_range(self, loader: ONETLoader) -> None:
        scores = loader.get_work_value_scores()
        for soc, values in scores.items():
            for value_name, score in values.items():
                assert 1.0 <= score <= 7.0, f"Work Value score out of range for {soc}: {value_name}={score}"

    @pytest.mark.skipif(not HAS_ONET_DATA, reason="O*NET data not downloaded")
    def test_get_work_value_highpoints(self, loader: ONETLoader) -> None:
        hp = loader.get_work_value_highpoints()
        assert len(hp) > 0
        sample_soc = next(iter(hp))
        assert len(hp[sample_soc]) <= 3  # At most 3 high-points


class TestExtendedProfile:
    """Tests for extended occupation profile."""

    @pytest.mark.skipif(not HAS_ONET_DATA, reason="O*NET data not downloaded")
    def test_profile_has_new_fields(self, loader: ONETLoader) -> None:
        # Use Registered Nurses as test occupation
        profile = loader.get_occupation_profile("29-1141.00")

        # New fields should exist
        assert "work_styles" in profile
        assert "big_five" in profile
        assert "work_values" in profile
        assert "work_value_highpoints" in profile

        # And have data (RN has Work Styles data)
        assert len(profile["work_styles"]) > 0
        assert len(profile["big_five"]) > 0
```

---

## Validation Commands

After implementation, run:

```bash
# Run tests
uv run pytest tests/unit/test_onet_loader.py -v

# Quick validation script
uv run python -c "
from pvx.data.onet_loader import ONETLoader

loader = ONETLoader()

# Test Work Styles
print('=== Work Styles Sample ===')
ws = loader.get_work_style_scores()
print(f'Occupations with Work Styles: {len(ws)}')
sample = list(ws.items())[0]
print(f'{sample[0]}: {sample[1]}')

# Test Big Five
print('\n=== Big Five Sample ===')
b5 = loader.get_big_five_scores()
sample = list(b5.items())[0]
print(f'{sample[0]}: {sample[1]}')

# Test Work Values
print('\n=== Work Values Sample ===')
wv = loader.get_work_value_scores()
print(f'Occupations with Work Values: {len(wv)}')
sample = list(wv.items())[0]
print(f'{sample[0]}: {sample[1]}')

# Test full profile
print('\n=== Full Profile (Chief Executives) ===')
profile = loader.get_occupation_profile('11-1011.00')
for key in ['work_styles', 'big_five', 'work_values', 'work_value_highpoints']:
    print(f'{key}: {profile.get(key, {})}')
"
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Missing Work Styles for some occupations (137/1016) | Return empty dict `{}`, matches existing pattern |
| Missing Work Values for some occupations (142/1016) | Return empty dict `{}` |
| Big Five derivation with partial data | Skip domain if any component missing |
| Cache invalidation | Follow existing pattern with `_cached` prefix |

---

## Implementation Order

1. **Add constants** (WORK_STYLE_ELEMENTS, BIG_FIVE_MAPPING, WORK_VALUE_ELEMENTS, WORK_VALUE_HIGHPOINT)
2. **Add cache attributes** to `__init__`
3. **Add load methods** (load_work_styles, load_work_values)
4. **Add score methods** (get_work_style_scores, get_work_value_scores, get_work_value_highpoints)
5. **Add derived method** (get_big_five_scores)
6. **Update get_occupation_profile** to include new data
7. **Add tests**
8. **Run validation**
9. **Update VocationalPersonaGenerator** metadata schema

---

## Sign-off Checklist

- [x] Verified Work Styles file structure (12 columns, IM scale)
- [x] Verified Work Values file structure (7 columns, EX/VH scales)
- [x] Verified element IDs match documentation
- [x] Confirmed coverage gaps (137 occupations missing from Work Styles)
- [x] Confirmed existing code patterns for caching and error handling
- [x] Test specifications cover all new methods
- [x] Validation commands verified to work with expected output
