#!/usr/bin/env python3
"""
LM-VECTOR Phase 1 — Linear Project Setup Script
=================================================
Creates the full project structure in Linear via GraphQL API.

Usage:
  1. Get your API key: Linear > Settings > Account > Security & Access > API
  2. Get your team ID: Run this script with --discover flag first
  3. Run dry-run:   python setup_linear.py --api-key lin_api_XXX --team-id UUID --dry-run
  4. Run for real:  python setup_linear.py --api-key lin_api_XXX --team-id UUID

The script will:
  - Create 3 stream labels (Data & Personas, Pipeline & Infra, Instruments & Eval)
  - Create helper labels (Critical Path, Decision Gate, Delegatable)
  - Create 4 weekly cycles
  - Create 1 project (LM-VECTOR Phase 1)
  - Create 28 issues with descriptions, labels, and cycle assignments
"""

import argparse
import json
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

API_URL = "https://api.linear.app/graphql"

# ─────────────────────────────────────────────────────────────────────────────
# All 28 tasks from the v3 plan
# ─────────────────────────────────────────────────────────────────────────────

TASKS = [
    # ── Stream 1: Data & Personas ──────────────────────────────────────────
    {
        "id": "S1-01", "title": "O*NET Data Download & Validation",
        "stream": "Data & Personas", "size": "S", "weeks": [1], "days": 2,
        "assignee": "Person B", "critical": True, "delegatable": True, "gate": None,
        "goal": "Produce clean JSON with all 1,016 O*NET occupations and metadata.",
        "acceptance": [
            "JSON has 1,016 entries with soc_code, title, description, 6 RIASEC scores",
            "10 spot-checks pass against O*NET website",
            "Schema documented",
        ],
        "deps": [], "inputs": "O*NET 29.0 database, pvx ONETLoader",
        "outputs": "occupations.json, schema.md",
    },
    {
        "id": "S1-02", "title": "RIASEC Stratification Analysis",
        "stream": "Data & Personas", "size": "XS", "weeks": [1], "days": 0.5,
        "assignee": "Person B", "critical": False, "delegatable": True, "gate": None,
        "goal": "Understand RIASEC distribution and propose stratified pilot set of 50 occupations (~8-9 per RIASEC category).",
        "acceptance": [
            "Counts per primary RIASEC code",
            "Ranked occupation list per category",
            "Pilot set of 50 with ~8-9 per category",
        ],
        "deps": ["S1-01"], "inputs": "occupations.json",
        "outputs": "riasec_stratification.md, pilot_occupations.json",
    },
    {
        "id": "S1-03", "title": "Persona Prompt Template Design",
        "stream": "Data & Personas", "size": "M", "weeks": [1, 2], "days": 4,
        "assignee": "Glenn + Person B", "critical": True, "delegatable": False, "gate": None,
        "goal": "Create 3-5 system prompt templates. No psychometric language, no trait labels, no demographics.",
        "acceptance": [
            "≥3 template variants",
            "Automated blocklist scanner passing",
            "Each tested on 5 diverse occupations",
            "Glenn approved",
        ],
        "deps": [], "inputs": "pvx VocationalPersonaGenerator, QC criteria",
        "outputs": "prompt_templates.py, blocklist_scanner.py",
    },
    {
        "id": "S1-04", "title": "Large-Scale Prompt Generation",
        "stream": "Data & Personas", "size": "M", "weeks": [2], "days": 3,
        "assignee": "Person B", "critical": True, "delegatable": True, "gate": None,
        "goal": "Generate full library: 1,016 occupations × 3-5 variants = 3,000-5,000 prompts.",
        "acceptance": [
            "One JSON per occupation with all variants",
            "All pass blocklist scan",
            "Manifest documenting any augmented/flagged prompts",
        ],
        "deps": ["S1-01", "S1-03"], "inputs": "occupations.json, templates",
        "outputs": "prompts/ directory, generation_manifest.csv",
    },
    {
        "id": "S1-05", "title": "Automated QC Filtering",
        "stream": "Data & Personas", "size": "S", "weeks": [2, 3], "days": 1,
        "assignee": "Person B", "critical": True, "delegatable": True, "gate": None,
        "goal": "Final QC pass over all generated prompts.",
        "acceptance": [
            "QC report with pass rate and failure reasons",
            "Filtered prompt set ready",
        ],
        "deps": ["S1-04"], "inputs": "prompts/ directory",
        "outputs": "prompts_filtered/, qc_report.md",
    },
    {
        "id": "S1-06", "title": "Non-Occupational Role Library",
        "stream": "Data & Personas", "size": "S", "weeks": [2], "days": 2,
        "assignee": "Person C", "critical": False, "delegatable": True, "gate": None,
        "goal": "Assemble ~515 character roles and traits from assistant-axis as additional personas.",
        "acceptance": [
            "~515 non-occupational persona JSONs",
            "Same schema as occupational",
            "Category tags present",
        ],
        "deps": ["S2-05"], "inputs": "assistant-axis role/trait lists",
        "outputs": "nonoccupational_prompts/ directory",
    },
    {
        "id": "S1-07", "title": "Leakage Audit",
        "stream": "Data & Personas", "size": "S", "weeks": [2], "days": 1,
        "assignee": "Person D", "critical": False, "delegatable": True, "gate": None,
        "goal": "Verify zero overlap between elicitation battery and psychometric instruments.",
        "acceptance": [
            "Audit report with methodology",
            "Item overlap matrix",
            "Zero items above threshold or documented replacements",
        ],
        "deps": ["S3-01", "S3-02", "S3-03", "S3-05"],
        "inputs": "elicitation_battery.json, all instruments",
        "outputs": "leakage_audit_report.md",
    },

    # ── Stream 2: Pipeline & Infrastructure ────────────────────────────────
    {
        "id": "S2-01", "title": "Model Download & Environment Setup",
        "stream": "Pipeline & Infra", "size": "M", "weeks": [1], "days": 3,
        "assignee": "Person C", "critical": True, "delegatable": True, "gate": None,
        "goal": "Download OLMo3, set up CUDA, verify generation works.",
        "acceptance": [
            "Model loads and generates coherent text",
            "Model card documented",
            "Smoke test passes",
            "Hardware requirements documented",
        ],
        "deps": [], "inputs": "HuggingFace model hub",
        "outputs": "model environment, model_card_olmo3.md",
    },
    {
        "id": "S2-02", "title": "Activation Extraction Code Refinement",
        "stream": "Pipeline & Infra", "size": "M", "weeks": [1, 2], "days": 5,
        "assignee": "Glenn", "critical": True, "delegatable": False, "gate": None,
        "goal": "Refine pvx extraction: 3 token-position modes (prompt_last, response_avg, prompt_avg), batching, selective layers.",
        "acceptance": [
            "Shape verification tests pass for all 3 modes",
            "Batch processing matches single-item results",
            "Selective layer extraction works",
            ".pt output format correct",
        ],
        "deps": ["S2-01"], "inputs": "pvx persona_model.py, assistant-axis reference",
        "outputs": "extract_activations.py, test_extraction.py",
    },
    {
        "id": "S2-03", "title": "Steering Code Validation",
        "stream": "Pipeline & Infra", "size": "S", "weeks": [2], "days": 2,
        "assignee": "Glenn", "critical": False, "delegatable": False, "gate": None,
        "goal": "Verify existing steering code works with synthetic vectors; choose implementation.",
        "acceptance": [
            "Steered ≠ unsteered qualitatively",
            "Layer parameter works",
            "Coefficient parameter works",
            "Implementation chosen and documented",
        ],
        "deps": ["S2-02"], "inputs": "pvx persona_model.py, persona_vectors activation_steer.py",
        "outputs": "steering.py, steering_test_results.md",
    },
    {
        "id": "S2-04", "title": "Judge Pipeline Implementation",
        "stream": "Pipeline & Infra", "size": "M", "weeks": [2], "days": 4,
        "assignee": "Person C", "critical": False, "delegatable": True, "gate": None,
        "goal": "Build 0-3 scale LLM judge with async batch scoring (0=refused, 1=broke character, 2=partial, 3=fully embodied).",
        "acceptance": [
            "Judge returns 0-3 scores",
            "≥80% agreement on 20 calibration cases",
            "Batch scorer handles 1,000+ responses",
            "Rate limiting and retries work",
        ],
        "deps": [], "inputs": "pvx LLMJudge, assistant-axis judge.py",
        "outputs": "judge.py, batch_scorer.py",
    },
    {
        "id": "S2-05", "title": "Data Format Standardization",
        "stream": "Pipeline & Infra", "size": "S", "weeks": [1], "days": 1,
        "assignee": "Glenn", "critical": False, "delegatable": False, "gate": None,
        "goal": "Define unified JSON schema for persona definitions across all three codebases.",
        "acceptance": [
            "JSON Schema documented",
            "Validator catches malformed files",
            "Converters from pvx/assistant-axis formats tested",
        ],
        "deps": [], "inputs": "Three codebase formats",
        "outputs": "persona_schema.json, validate_persona.py, format_migration.py",
    },
    {
        "id": "S2-06", "title": "Baseline Response Generation",
        "stream": "Pipeline & Infra", "size": "S", "weeks": [3], "days": 0.5,
        "assignee": "Person C", "critical": True, "delegatable": True, "gate": None,
        "goal": "Generate default model responses to elicitation battery (text only). 3 variants × 250 Qs = 750 generations.",
        "acceptance": [
            "JSONL has 750 entries",
            "20 random responses look normal (spot-check)",
        ],
        "deps": ["S3-05", "S2-01"], "inputs": "elicitation_battery.json, model",
        "outputs": "baseline_responses.jsonl",
    },
    {
        "id": "S2-07", "title": "Baseline Activation Extraction",
        "stream": "Pipeline & Infra", "size": "S", "weeks": [3], "days": 0.5,
        "assignee": "Person C", "critical": True, "delegatable": True, "gate": None,
        "goal": "Second pass: extract hidden states from baseline responses. All layers, all 3 token modes.",
        "acceptance": [
            "Shapes match model config",
            "No NaN/zeros (spot-check 10 tensors)",
            "2,250 tensor files stored",
        ],
        "deps": ["S2-06", "S2-02"], "inputs": "baseline_responses.jsonl, model, extraction code",
        "outputs": "baseline_activations/",
    },
    {
        "id": "S2-08", "title": "Pilot Persona Response Generation (50 occupations)",
        "stream": "Pipeline & Infra", "size": "M", "weeks": [3], "days": 2,
        "assignee": "Person C", "critical": True, "delegatable": True, "gate": None,
        "goal": "Generate text for 50 pilot personas × 3 variants × 250 Qs = 37,500 responses. Compute: 40-75 GPU-hours.",
        "acceptance": [
            "50 persona dirs with correct response counts",
            "Spot-check: nurse ≠ developer qualitatively",
            "No truncated responses",
        ],
        "deps": ["S1-05", "S3-05", "S2-01"],
        "inputs": "pilot prompts, elicitation battery, model",
        "outputs": "pilot_responses/",
    },
    {
        "id": "S2-09", "title": "Pilot Activation Extraction",
        "stream": "Pipeline & Infra", "size": "L", "weeks": [3], "days": 4,
        "assignee": "Person C", "critical": True, "delegatable": True, "gate": None,
        "goal": "Extract activations from all 37,500 pilot responses. All layers. Most compute-intensive task: 100-200 GPU-hours.",
        "acceptance": [
            "Shapes match model config",
            "No NaN values",
            "File counts correct",
        ],
        "deps": ["S2-08", "S2-02"],
        "inputs": "pilot_responses/, model, extraction code",
        "outputs": "pilot_activations/",
    },
    {
        "id": "S2-10", "title": "Pilot Adherence Scoring → Gate 1",
        "stream": "Pipeline & Infra", "size": "S", "weeks": [3], "days": 0.5,
        "assignee": "Person C", "critical": True, "delegatable": True, "gate": 1,
        "goal": "Score all 37,500 pilot responses with LLM judge. Feed Decision Gate 1.",
        "acceptance": [
            "All responses scored 0-3",
            "Adherence report with per-persona stats produced",
            "Gate 1 evaluated: >60% of personas achieve score-3 on >50% of questions",
        ],
        "deps": ["S2-08", "S2-04"],
        "inputs": "pilot_responses/, judge.py",
        "outputs": "adherence_scores_pilot.csv, adherence_report.md",
    },
    {
        "id": "S2-11", "title": "Persona Vector Computation (Pilot)",
        "stream": "Pipeline & Infra", "size": "S", "weeks": [3, 4], "days": 1,
        "assignee": "Glenn", "critical": True, "delegatable": False, "gate": None,
        "goal": "Compute persona vectors for 50 personas using score-3 filtered responses. All 3 token modes → 150 vectors.",
        "acceptance": [
            "150 .pt files (50 personas × 3 modes)",
            "Shapes verified [num_layers, hidden_dim]",
            "Response-to-score-to-activation bookkeeping spot-checked",
        ],
        "deps": ["S2-09", "S2-07", "S2-10"],
        "inputs": "pilot_activations/, baseline_activations/, adherence_scores",
        "outputs": "persona_vectors_pilot/",
    },
    {
        "id": "S2-12", "title": "Token Position Experiment",
        "stream": "Pipeline & Infra", "size": "S", "weeks": [4], "days": 0.5,
        "assignee": "Glenn", "critical": True, "delegatable": False, "gate": None,
        "goal": "Compare prompt_last vs response_avg vs prompt_avg using RIASEC clustering quality and k-NN classification (k=5).",
        "acceptance": [
            "Report comparing 3 modes with metrics",
            "Winner identified with rationale",
            "Decision documented",
        ],
        "deps": ["S2-11"],
        "inputs": "persona_vectors_pilot/ (all modes), riasec_mapping",
        "outputs": "token_position_report.md",
    },
    {
        "id": "S2-13", "title": "Layer Selection Experiment",
        "stream": "Pipeline & Infra", "size": "S", "weeks": [4], "days": 0.5,
        "assignee": "Glenn", "critical": True, "delegatable": False, "gate": None,
        "goal": "Identify optimal extraction/steering layers via per-layer RIASEC clustering metrics.",
        "acceptance": [
            "Layer quality plot (within-vs-between ratio vs depth)",
            "Optimal layer identified",
            "Layer range within 90% of best documented",
        ],
        "deps": ["S2-12"],
        "inputs": "persona_vectors_pilot/ (best mode, all layers)",
        "outputs": "layer_selection_report.md, optimal_layers.json",
    },
    {
        "id": "S2-14", "title": "Steering Validation (Real Vectors) → Gate 2",
        "stream": "Pipeline & Infra", "size": "M", "weeks": [4], "days": 3,
        "assignee": "Glenn", "critical": True, "delegatable": False, "gate": 2,
        "goal": "Validate steering with real vectors: 3 conditions × 10 personas × coefficient sweep 0.5-5.0.",
        "acceptance": [
            "Condition comparison table (baseline vs prompt-only vs steered)",
            "Coefficient sweep results with coherence check",
            "Optimal coefficient found",
            "Gate 2 evaluated: steered > baseline for >70% of test personas",
        ],
        "deps": ["S2-11", "S2-13", "S2-03"],
        "inputs": "persona_vectors, steering.py, optimal_layers",
        "outputs": "steering_validation_report.md",
    },

    # ── Stream 3: Instruments & Evaluation ─────────────────────────────────
    {
        "id": "S3-01", "title": "IPIP-NEO-120 Assembly & Scoring",
        "stream": "Instruments & Eval", "size": "S", "weeks": [1, 2], "days": 3,
        "assignee": "Person D", "critical": False, "delegatable": True, "gate": None,
        "goal": "Assemble 120-item Big Five inventory with scoring functions (5 domains, 30 facets).",
        "acceptance": [
            "JSON with 120 annotated items (text, domain, facet, keying)",
            "Scoring correct on published test case",
            "Reverse-scoring verified",
        ],
        "deps": [], "inputs": "ipip.ori.org",
        "outputs": "ipip_neo_120.json, score_ipip_neo.py",
    },
    {
        "id": "S3-02", "title": "HEXACO-100 Assembly & Scoring",
        "stream": "Instruments & Eval", "size": "S", "weeks": [1, 2], "days": 3,
        "assignee": "Person D", "critical": False, "delegatable": True, "gate": None,
        "goal": "Assemble 100-item HEXACO inventory with scoring (6 domains, 25 facets).",
        "acceptance": [
            "JSON with 100 annotated items",
            "Scoring correct on published test case",
            "All 25 facets producing scores",
        ],
        "deps": [], "inputs": "HEXACO website",
        "outputs": "hexaco_100.json, score_hexaco.py",
    },
    {
        "id": "S3-03", "title": "O*NET Interest Profiler Assembly",
        "stream": "Instruments & Eval", "size": "S", "weeks": [1], "days": 2,
        "assignee": "Person D", "critical": False, "delegatable": True, "gate": None,
        "goal": "Assemble 60-item RIASEC inventory with scoring (6 dimensions × 10 items).",
        "acceptance": [
            "JSON with 60 items",
            "6 RIASEC dimension scores computed",
            "Validated against published test case",
        ],
        "deps": [], "inputs": "O*NET Resource Center",
        "outputs": "interest_profiler.json, score_riasec.py",
    },
    {
        "id": "S3-04", "title": "Likert Response Parsing",
        "stream": "Instruments & Eval", "size": "S", "weeks": [2], "days": 3,
        "assignee": "Person D", "critical": False, "delegatable": True, "gate": None,
        "goal": "Build parser for LLM Likert responses + decide optimal prompt format for item administration.",
        "acceptance": [
            "Prompt format chosen with rationale",
            "Regex handles ≥80% of responses",
            "LLM fallback brings total to ≥95%",
            "Failure cases flagged for review",
        ],
        "deps": ["S2-01"], "inputs": "Model, sample items",
        "outputs": "likert_parser.py, item_prompt_format.md",
    },
    {
        "id": "S3-05", "title": "Elicitation Battery Design",
        "stream": "Instruments & Eval", "size": "M", "weeks": [1, 2], "days": 4,
        "assignee": "Person D", "critical": False, "delegatable": True, "gate": None,
        "goal": "Design ~250 role-agnostic questions for vector extraction. Zero overlap with psychometric items.",
        "acceptance": [
            "~250 questions in JSON, categorized by topic",
            "Glenn reviewed sample of 20",
            "No psychometric wording (verified by S1-07)",
        ],
        "deps": [], "inputs": "assistant-axis question list (reference)",
        "outputs": "elicitation_battery.json",
    },
    {
        "id": "S3-06", "title": "Baseline Psychometric Administration",
        "stream": "Instruments & Eval", "size": "S", "weeks": [4], "days": 1,
        "assignee": "Person D", "critical": False, "delegatable": True, "gate": None,
        "goal": "Administer 280 psychometric items to default model (no persona, no steering). Establish baseline scores.",
        "acceptance": [
            "All 280 items administered and parsed",
            "Parse rate ≥95%",
            "Stability across repetitions documented",
            "Scores: Big Five (5+30), HEXACO (6+25), RIASEC (6)",
        ],
        "deps": ["S3-01", "S3-02", "S3-03", "S3-04", "S2-01"],
        "inputs": "Instruments, parser, model",
        "outputs": "baseline_psychometric_scores.json",
    },
    {
        "id": "S3-07", "title": "Steered Psychometric Administration (Pilot)",
        "stream": "Instruments & Eval", "size": "M", "weeks": [4], "days": 2,
        "assignee": "Person D", "critical": True, "delegatable": True, "gate": None,
        "goal": "Administer 280 items to each of 50 steered personas. 14,000 total generations. Compute deltas vs baseline.",
        "acceptance": [
            "14,000 items administered and parsed",
            "Parse rate ≥95%",
            "Delta table: steered - baseline for every persona × domain × facet",
        ],
        "deps": ["S2-14", "S3-06", "S2-11"],
        "inputs": "persona_vectors, steering params, instruments",
        "outputs": "steered_psychometric_scores.csv, psychometric_deltas.csv",
    },
    {
        "id": "S3-08", "title": "H1 Statistical Validation → Gate 3",
        "stream": "Instruments & Eval", "size": "M", "weeks": [4], "days": 3,
        "assignee": "Glenn", "critical": True, "delegatable": False, "gate": 3,
        "goal": "Formally test H1: non-trivial, stable, occupation-dependent psychometric shifts.",
        "acceptance": [
            "Test 1: Bootstrap CIs on mean absolute deltas (CI excludes zero)",
            "Test 2: Permutation baseline p<0.05 (shifts better than shuffled labels)",
            "Test 3: Correlation between O*NET profiles and steered deltas",
            "Test 4: Facet-level analysis with Benjamini-Hochberg FDR correction",
            "H1 verdict documented and reviewed",
        ],
        "deps": ["S3-07"],
        "inputs": "psychometric_deltas.csv, occupations.json",
        "outputs": "h1_report.md",
    },
]

# Priority mapping: 0=none, 1=urgent, 2=high, 3=medium, 4=low
def get_priority(task):
    if task["gate"]:
        return 1  # urgent
    if task["critical"]:
        return 2  # high
    return 3  # medium


# ─────────────────────────────────────────────────────────────────────────────
# GraphQL helpers
# ─────────────────────────────────────────────────────────────────────────────

def gql(api_key, query, variables=None):
    """Execute a GraphQL query against Linear API."""
    body = {"query": query}
    if variables:
        body["variables"] = variables
    data = json.dumps(body).encode("utf-8")
    req = Request(API_URL, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": api_key,
    })
    try:
        with urlopen(req) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if "errors" in result:
                print(f"  ⚠ GraphQL errors: {result['errors']}", file=sys.stderr)
            return result
    except HTTPError as e:
        body = e.read().decode("utf-8")
        print(f"  ✗ HTTP {e.code}: {body}", file=sys.stderr)
        sys.exit(1)


def discover(api_key):
    """Discover workspace info: teams, projects, labels, cycles, workflow states."""
    print("\n🔍 Discovering workspace...\n")

    # Teams
    r = gql(api_key, "{ teams { nodes { id name } } }")
    teams = r["data"]["teams"]["nodes"]
    print("Teams:")
    for t in teams:
        print(f"  {t['name']:30s}  {t['id']}")

    # Workflow states (for the first team)
    if teams:
        tid = teams[0]["id"]
        r = gql(api_key, """
            query($tid: String!) {
                team(id: $tid) {
                    states { nodes { id name type } }
                }
            }
        """, {"tid": tid})
        states = r["data"]["team"]["states"]["nodes"]
        print(f"\nWorkflow states for '{teams[0]['name']}':")
        for s in states:
            print(f"  {s['name']:20s}  type={s['type']:12s}  {s['id']}")

    # Existing projects
    r = gql(api_key, "{ projects { nodes { id name } } }")
    projects = r["data"]["projects"]["nodes"]
    print(f"\nProjects ({len(projects)}):")
    for p in projects:
        print(f"  {p['name']:30s}  {p['id']}")

    # Existing labels
    r = gql(api_key, "{ issueLabels { nodes { id name color } } }")
    labels = r["data"]["issueLabels"]["nodes"]
    print(f"\nLabels ({len(labels)}):")
    for l in labels:
        print(f"  {l['name']:30s}  {l['color']}  {l['id']}")

    # Existing cycles
    r = gql(api_key, """{ cycles { nodes { id name number startsAt endsAt } } }""")
    cycles = r["data"]["cycles"]["nodes"]
    print(f"\nCycles ({len(cycles)}):")
    for c in cycles:
        print(f"  #{c.get('number','?')}: {c.get('name','(unnamed)'):20s}  {c.get('startsAt','?')} → {c.get('endsAt','?')}  {c['id']}")

    # Members
    r = gql(api_key, "{ users { nodes { id name email } } }")
    users = r["data"]["users"]["nodes"]
    print(f"\nMembers ({len(users)}):")
    for u in users:
        print(f"  {u['name']:25s}  {u.get('email',''):30s}  {u['id']}")

    print("\n" + "="*70)
    print("Copy your team ID from above and use it with --team-id")
    print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# Creation functions
# ─────────────────────────────────────────────────────────────────────────────

LABEL_DEFS = [
    # Stream labels
    {"name": "Stream 1: Data & Personas",     "color": "#2DD4BF"},
    {"name": "Stream 2: Pipeline & Infra",    "color": "#818CF8"},
    {"name": "Stream 3: Instruments & Eval",  "color": "#FB923C"},
    # Meta labels
    {"name": "Critical Path",                 "color": "#EF4444"},
    {"name": "Decision Gate",                 "color": "#FBBF24"},
    {"name": "Delegatable",                   "color": "#22C55E"},
    # Size labels
    {"name": "Size: XS",                      "color": "#94A3B8"},
    {"name": "Size: S",                       "color": "#94A3B8"},
    {"name": "Size: M",                       "color": "#64748B"},
    {"name": "Size: L",                       "color": "#475569"},
]

STREAM_TO_LABEL = {
    "Data & Personas":   "Stream 1: Data & Personas",
    "Pipeline & Infra":  "Stream 2: Pipeline & Infra",
    "Instruments & Eval":"Stream 3: Instruments & Eval",
}

# Week start dates (Mon-Fri cycles, starting Feb 10, 2026)
CYCLES = [
    {"name": "Week 1: Setup & Foundation",    "start": "2026-02-10", "end": "2026-02-14"},
    {"name": "Week 2: Build & Generate",      "start": "2026-02-17", "end": "2026-02-21"},
    {"name": "Week 3: Pilot Run & Gate 1",    "start": "2026-02-24", "end": "2026-02-28"},
    {"name": "Week 4: Experiments & H1",      "start": "2026-03-03", "end": "2026-03-07"},
]


def format_description(task):
    """Build a rich Markdown description for a Linear issue."""
    lines = []

    lines.append(f"**Goal:** {task['goal']}")
    lines.append("")

    # Metadata block
    meta = []
    meta.append(f"**Stream:** {task['stream']}")
    meta.append(f"**Size:** {task['size']} · **Est. days:** {task['days']}")
    meta.append(f"**Assignee:** {task['assignee']}")
    if task["critical"]:
        meta.append("**🔴 CRITICAL PATH**")
    if task["gate"]:
        gate_info = {
            1: ("Do personas work?", ">60% of personas achieve score-3 on >50% of questions"),
            2: ("Does steering work?", ">70% improved adherence, coefficient exists without coherence degradation"),
            3: ("Occupation-dependent shifts?", "≥2 of 3 instruments show significant occupation-dependent shifts (permutation p<0.05)"),
        }
        q, threshold = gate_info[task["gate"]]
        meta.append(f"**⭐ DECISION GATE {task['gate']}:** {q}")
        meta.append(f"**Pass threshold:** {threshold}")
    lines.extend(meta)
    lines.append("")

    # Acceptance criteria
    lines.append("### Acceptance Criteria")
    for a in task["acceptance"]:
        lines.append(f"- [ ] {a}")
    lines.append("")

    # Dependencies
    if task["deps"]:
        lines.append(f"### Dependencies")
        lines.append(f"Blocked by: {', '.join(task['deps'])}")
        lines.append("")

    # I/O
    lines.append("### Inputs")
    lines.append(task["inputs"])
    lines.append("")
    lines.append("### Outputs")
    lines.append(task["outputs"])

    return "\n".join(lines)


def create_all(api_key, team_id, dry_run=False, skip_labels=False,
               skip_project=False, skip_cycles=False, project_id=None,
               reuse_labels=False):
    """Create the full Linear project structure."""
    label_ids = {}  # name -> id
    cycle_ids = {}  # week_num -> id

    # ── 1. Labels ─────────────────────────────────────────────────────────
    if reuse_labels or skip_labels:
        print("\n📌 Fetching existing labels...")
        r = gql(api_key, "{ issueLabels { nodes { id name } } }")
        for l in r["data"]["issueLabels"]["nodes"]:
            label_ids[l["name"]] = l["id"]
        matched = [n for n in label_ids if any(n == d["name"] for d in LABEL_DEFS)]
        print(f"  Found {len(matched)} matching labels")
    else:
        print("\n📌 Creating labels...")
        for ldef in LABEL_DEFS:
            if dry_run:
                print(f"  [DRY RUN] Would create label: {ldef['name']} ({ldef['color']})")
                label_ids[ldef["name"]] = "dry-run-id"
            else:
                r = gql(api_key, """
                    mutation($input: IssueLabelCreateInput!) {
                        issueLabelCreate(input: $input) {
                            success
                            issueLabel { id name }
                        }
                    }
                """, {"input": {"name": ldef["name"], "color": ldef["color"], "teamId": team_id}})
                if r.get("data") and r["data"].get("issueLabelCreate") and r["data"]["issueLabelCreate"]["success"]:
                    lid = r["data"]["issueLabelCreate"]["issueLabel"]["id"]
                    label_ids[ldef["name"]] = lid
                    print(f"  ✓ {ldef['name']}  →  {lid}")
                else:
                    print(f"  ✗ Failed: {ldef['name']}")
                time.sleep(0.2)

    # ── 2. Create project ─────────────────────────────────────────────────
    proj_name = "LM-VECTOR Phase 1"
    proj_desc = (
        "Map persona-conditioned behavioral variation in open-weight LMs as directions "
        "in activation space, validate with psychometrics, assess cross-model consistency. "
        "COLM deadline Mar 31, 2026."
    )
    if project_id:
        print(f"\n📁 Using existing project: {project_id}")
    elif skip_project:
        print("\n📁 Skipping project creation (--skip-project)")
        project_id = None
    elif dry_run:
        print(f"  [DRY RUN] Would create project: {proj_name}")
        project_id = "dry-run-project-id"
    else:
        r = gql(api_key, """
            mutation($input: ProjectCreateInput!) {
                projectCreate(input: $input) {
                    success
                    project { id name }
                }
            }
        """, {"input": {
            "name": proj_name,
            "description": proj_desc,
            "teamIds": [team_id],
            "targetDate": "2026-03-31",
        }})
        if r.get("data") and r["data"].get("projectCreate") and r["data"]["projectCreate"]["success"]:
            project_id = r["data"]["projectCreate"]["project"]["id"]
            print(f"  ✓ {proj_name}  →  {project_id}")
        else:
            print(f"  ✗ Failed to create project. Errors: {r.get('errors', 'unknown')}")
            sys.exit(1)

    # ── 3. Create cycles ──────────────────────────────────────────────────
    if skip_cycles:
        print("\n🔄 Skipping cycle creation (--skip-cycles)")
    else:
        print("\n🔄 Creating weekly cycles...")
        for i, cdef in enumerate(CYCLES):
            week_num = i + 1
            if dry_run:
                print(f"  [DRY RUN] Would create cycle: {cdef['name']} ({cdef['start']} → {cdef['end']})")
                cycle_ids[week_num] = "dry-run-cycle-id"
            else:
                r = gql(api_key, """
                    mutation($input: CycleCreateInput!) {
                        cycleCreate(input: $input) {
                            success
                            cycle { id name number }
                        }
                    }
                """, {"input": {
                    "teamId": team_id,
                    "name": cdef["name"],
                    "startsAt": cdef["start"],
                    "endsAt": cdef["end"],
                }})
                if r.get("data") and r["data"].get("cycleCreate") and r["data"]["cycleCreate"]["success"]:
                    cid = r["data"]["cycleCreate"]["cycle"]["id"]
                    cycle_ids[week_num] = cid
                    print(f"  ✓ {cdef['name']}  →  {cid}")
                else:
                    print(f"  ✗ Failed: {cdef['name']}")
                time.sleep(0.2)

    # ── 4. Create issues ──────────────────────────────────────────────────
    print(f"\n📋 Creating {len(TASKS)} issues...")
    created = {}  # task_id -> linear_id

    for task in TASKS:
        title = f"[{task['id']}] {task['title']}"
        desc = format_description(task)

        # Assemble label IDs
        task_label_ids = []
        stream_label = STREAM_TO_LABEL.get(task["stream"])
        if stream_label and stream_label in label_ids:
            task_label_ids.append(label_ids[stream_label])
        if task["critical"] and "Critical Path" in label_ids:
            task_label_ids.append(label_ids["Critical Path"])
        if task["gate"] and "Decision Gate" in label_ids:
            task_label_ids.append(label_ids["Decision Gate"])
        if task["delegatable"] and "Delegatable" in label_ids:
            task_label_ids.append(label_ids["Delegatable"])
        size_label = f"Size: {task['size']}"
        if size_label in label_ids:
            task_label_ids.append(label_ids[size_label])

        # Pick cycle (first week the task appears in)
        first_week = task["weeks"][0]
        cycle_id = cycle_ids.get(first_week)

        # Due date = Friday of last week
        last_week = task["weeks"][-1]
        due_dates = {1: "2026-02-14", 2: "2026-02-21", 3: "2026-02-28", 4: "2026-03-07"}
        due = due_dates.get(last_week, "2026-03-07")

        if dry_run:
            labels_str = ", ".join([k for k, v in label_ids.items() if v in task_label_ids]) if not dry_run else task["stream"]
            print(f"  [DRY RUN] {task['id']:6s}  {task['title'][:50]:50s}  W{first_week}  P{get_priority(task)}  due={due}")
        else:
            input_data = {
                "title": title,
                "description": desc,
                "teamId": team_id,
                "priority": get_priority(task),
                "dueDate": due,
            }
            if project_id:
                input_data["projectId"] = project_id
            if task_label_ids:
                input_data["labelIds"] = task_label_ids
            if cycle_id:
                input_data["cycleId"] = cycle_id

            r = gql(api_key, """
                mutation($input: IssueCreateInput!) {
                    issueCreate(input: $input) {
                        success
                        issue { id identifier title }
                    }
                }
            """, {"input": input_data})

            if r.get("data") and r["data"].get("issueCreate") and r["data"]["issueCreate"]["success"]:
                issue = r["data"]["issueCreate"]["issue"]
                created[task["id"]] = issue["id"]
                print(f"  ✓ {issue['identifier']:10s}  {task['id']:6s}  {task['title'][:50]}")
            else:
                print(f"  ✗ Failed: {task['id']} {task['title']}")
            time.sleep(0.3)  # rate limit

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*70)
    if dry_run:
        print("DRY RUN COMPLETE — nothing was created.")
        print(f"Would create: {len(LABEL_DEFS)} labels, 1 project, {len(CYCLES)} cycles, {len(TASKS)} issues")
    else:
        print(f"✅ Setup complete!")
        print(f"   Labels:  {len(label_ids)}")
        print(f"   Project: {project_id}")
        print(f"   Cycles:  {len(cycle_ids)}")
        print(f"   Issues:  {len(created)}")
    print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# CSV export (backup / fallback)
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(path="lm-vector-linear-import.csv"):
    """Export tasks as CSV compatible with Linear's bulk import."""
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Title", "Description", "Priority", "Labels", "Due Date"])
        for task in TASKS:
            title = f"[{task['id']}] {task['title']}"
            desc = format_description(task)
            pri_map = {1: "Urgent", 2: "High", 3: "Medium", 4: "Low"}
            pri = pri_map.get(get_priority(task), "Medium")
            labels = [STREAM_TO_LABEL[task["stream"]]]
            if task["critical"]:
                labels.append("Critical Path")
            if task["gate"]:
                labels.append("Decision Gate")
            if task["delegatable"]:
                labels.append("Delegatable")
            labels.append(f"Size: {task['size']}")
            due_dates = {1: "2026-02-14", 2: "2026-02-21", 3: "2026-02-28", 4: "2026-03-07"}
            due = due_dates.get(task["weeks"][-1], "2026-03-07")
            w.writerow([title, desc, pri, "; ".join(labels), due])
    print(f"✓ CSV exported to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Set up LM-VECTOR Phase 1 project in Linear",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Discover your workspace (find team ID)
  python setup_linear.py --api-key lin_api_XXX --discover

  # Step 2: Preview what will be created
  python setup_linear.py --api-key lin_api_XXX --team-id UUID --dry-run

  # Step 3: Create everything
  python setup_linear.py --api-key lin_api_XXX --team-id UUID

  # Backup: Export CSV for manual import
  python setup_linear.py --csv
        """
    )
    parser.add_argument("--api-key", help="Linear API key (lin_api_...)")
    parser.add_argument("--team-id", help="Linear team UUID")
    parser.add_argument("--discover", action="store_true", help="List workspace info")
    parser.add_argument("--dry-run", action="store_true", help="Preview without creating")
    parser.add_argument("--csv", action="store_true", help="Export CSV instead of using API")
    parser.add_argument("--skip-labels", action="store_true", help="Skip label creation (if already created)")
    parser.add_argument("--skip-project", action="store_true", help="Skip project creation")
    parser.add_argument("--skip-cycles", action="store_true", help="Skip cycle creation")
    parser.add_argument("--project-id", help="Existing project UUID to attach issues to")
    parser.add_argument("--reuse-labels", action="store_true", help="Fetch existing labels by name instead of creating")

    args = parser.parse_args()

    if args.csv:
        export_csv()
        return

    if not args.api_key:
        parser.error("--api-key is required (get it from Linear > Settings > Account > Security & Access > API)")

    if args.discover:
        discover(args.api_key)
        return

    if not args.team_id:
        parser.error("--team-id is required (run with --discover first to find it)")

    create_all(args.api_key, args.team_id, dry_run=args.dry_run,
               skip_labels=args.skip_labels, skip_project=args.skip_project,
               skip_cycles=args.skip_cycles, project_id=args.project_id,
               reuse_labels=args.reuse_labels)


if __name__ == "__main__":
    main()
