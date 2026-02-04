# Code Review: occupational_network Branch

**Date:** 2026-02-03  
**Reviewer:** Automated code review (5 parallel agents)  
**Branch:** `occupational_network`  
**Base:** `main`  
**Diff Stats:** +30,725 lines (after LFS migration from +1,071,666)

---

## Branch Summary

This branch implements Phase 1 O*NET psychometric integration for generating occupational personas based on vocational psychology theory.

Key additions:

- O*NET data pipeline (`onet_loader.py`) parses occupational database
- Vocational source (`vocational.py`) generates 150 vocational personas
- Extraction framework with question bank, activation extraction, and full pipeline
- Evaluation system with LLM-as-judge and RIASEC judge
- Infrastructure for multi-model runs and SLURM cluster execution
- Analysis tools for persona geometry and cross-model comparison

---

## Critical Issues (Score >= 80)

### Issue 1: String Escaping Bug in Prompt Formatting

| Field | Value |
|-------|-------|
| Score | 85/100 |
| Severity | Critical - breaks hf_local backend |
| File | `src/pvx/judges/llm_as_judge.py` |
| Lines | 331-337 |

#### Problem

The `_messages_to_prompt` method uses literal `\\n` (backslash-n as text) instead of `\n` (actual newline character).

#### Current Code (Buggy)

```python
system_block = "\\n".join(system_parts)
user_block = "\\n\\n".join(user_parts)
prompt = ""
if system_block:
    prompt += f"{system_block}\\n\\n"
prompt += user_block
prompt += "\\n\\nAssistant:"
```

#### Fix Required

```python
system_block = "\n".join(system_parts)
user_block = "\n\n".join(user_parts)
prompt = ""
if system_block:
    prompt += f"{system_block}\n\n"
prompt += user_block
prompt += "\n\nAssistant:"
```

#### Impact

When using the `hf_local` backend, prompts contain literal backslash-n text instead of actual line breaks. This breaks prompt formatting and will degrade model performance.

#### How to Verify

```bash
# Check the file
grep -n '\\\\n' src/pvx/judges/llm_as_judge.py
```

---

## Near-Threshold Issues (75-79)

### Issue 2: Missing OpenAI API Error Handling

| Field | Value |
|-------|-------|
| Score | 78/100 |
| Severity | Medium - causes crashes on API failures |
| File | `src/pvx/pvx_models/vocational_dataset.py` |
| Lines | 123-141 |

#### Problem

The `generate_system_prompts()` method makes an OpenAI API call but only catches `json.JSONDecodeError`. Network failures, rate limiting, or authentication errors will crash the script instead of using the fallback method.

#### Current Code

```python
response = self.client.chat.completions.create(
    model=self.model,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
)
content = response.choices[0].message.content.strip()

try:
    prompts = json.loads(content)
    if isinstance(prompts, list) and len(prompts) == 5:
        return prompts
except json.JSONDecodeError:
    pass
```

#### Suggested Fix

```python
try:
    response = self.client.chat.completions.create(
        model=self.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    content = response.choices[0].message.content.strip()
    prompts = json.loads(content)
    if isinstance(prompts, list) and len(prompts) == 5:
        return prompts
except (json.JSONDecodeError, Exception) as e:
    logger.warning(f"API call failed for {profile['title']}: {e}, using fallback")

return self._generate_fallback_prompts(profile)
```

---

### Issue 3: File Length Violations (AGENTS.md)

| Field | Value |
|-------|-------|
| Score | 75/100 |
| Severity | Style/Architecture guideline violation |
| Guideline | AGENTS.md line 179: "Files should never exceed 150 lines" |

#### Files Exceeding Limit

| File | Lines | Over By |
|------|-------|---------|
| `src/pvx/analysis/viz.py` | 732 | 487% |
| `src/pvx/data/onet_loader.py` | 603 | 402% |
| `src/pvx/extraction/pipeline.py` | 536 | 357% |
| `src/pvx/analysis/geometry.py` | 439 | 293% |
| `src/pvx/extraction/activations.py` | 373 | 249% |
| `experiments/phase1/run_riasec_analysis.py` | 350 | 233% |
| `src/pvx/analysis/riasec.py` | 320 | 213% |
| `experiments/phase1/validate_pipeline.py` | 315 | 210% |
| `src/pvx/analysis/comparison.py` | 302 | 201% |
| `experiments/phase1/validate_riasec.py` | 286 | 191% |
| `src/pvx/config/run_config.py` | 287 | 191% |
| `src/pvx/outputs.py` | 253 | 169% |
| `src/pvx/extraction/questions.py` | 235 | 157% |

#### Recommendation

Consider refactoring large files into smaller modules with clear separation of concerns. For example:

- `viz.py` could split into `viz_plots.py`, `viz_heatmaps.py`, `viz_utils.py`
- `onet_loader.py` could split into `onet_parser.py`, `onet_models.py`, `onet_utils.py`
- `pipeline.py` could split into `pipeline_core.py`, `pipeline_io.py`, `pipeline_validation.py`

---

### Issue 4: Accumulator None Pattern Risk

| Field | Value |
|-------|-------|
| Score | 75/100 |
| Severity | Low - code smell, maintenance risk |
| File | `src/pvx/extraction/pipeline.py` |
| Lines | 263-269, 339-364 |

#### Problem

Accumulators are initialized as `None` and only assigned inside a loop after the first valid pair passes filtering. If no pairs pass, accumulators remain `None`.

#### Current Pattern

```python
# Lines 263-269: Initialize as None
persona_prompt_last_sum = None
persona_response_avg_sum = None
persona_all_layers_sum = None
baseline_prompt_last_sum = None
baseline_response_avg_sum = None
baseline_all_layers_sum = None

# ... later in loop ...
if persona_prompt_last_sum is None:
    persona_prompt_last_sum = torch.zeros_like(...)
```

#### Mitigation

The code has a guard at line 339 (`if valid_count == 0`) that returns early with zero vectors. This handles the case correctly, but the pattern is fragile and could break if modified.

#### Recommendation

Consider initializing accumulators with a shape placeholder or using a more explicit pattern that doesn't rely on None checks inside loops.

---

## Filtered Issues (Score < 75)

These issues were identified but scored below the threshold:

| Issue | Score | Reason Filtered |
|-------|-------|-----------------|
| Print statements in CLI scripts | 25 | Standard pattern for script output in `__main__` blocks |
| Epsilon placement in cosine similarity | 25 | Mathematically valid but practically unlikely to cause issues |

---

## Review Methodology

Five parallel agents reviewed the branch:

1. **CLAUDE.md Compliance Agent** - Checked adherence to project guidelines
2. **Bug Scanner Agent** - Shallow scan for obvious bugs in changed lines
3. **Git History Agent** - Analyzed historical context and patterns
4. **PR History Agent** - Checked previous PR comments for recurring issues
5. **Code Comments Agent** - Verified changes comply with documented guidance

Each issue was then scored by a separate agent using this rubric:

- **0**: False positive
- **25**: Might be real, stylistic, not in CLAUDE.md
- **50**: Real but nitpick, not important
- **75**: Very likely real, explicitly in CLAUDE.md
- **100**: Definitely real, will happen frequently

---

## Action Items

### Must Fix (Before PR)

- [x] Fix string escaping in `llm_as_judge.py` lines 331-337 (FIXED 2026-02-03)
- [x] Fix string escaping in `persona_dataset.py` lines 575-589 (FIXED 2026-02-03 - same bug)

### Should Fix (Recommended)

- [x] Add API error handling in `vocational_dataset.py` (FIXED 2026-02-03)

### Consider (Future Work)

- [ ] Refactor files exceeding 150-line limit
- [ ] Refactor accumulator initialization pattern

---

## Commands for Verification

```bash
# Check for the string escaping bug
grep -n '\\\\n' src/pvx/judges/llm_as_judge.py

# Count lines in flagged files
wc -l src/pvx/analysis/viz.py src/pvx/data/onet_loader.py src/pvx/extraction/pipeline.py

# View the problematic method
sed -n '320,340p' src/pvx/judges/llm_as_judge.py
```
