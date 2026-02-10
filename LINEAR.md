# LINEAR.md — LM-VECTOR Project Management Context

## Purpose

This document primes a Claude Code agent to manage the LM-VECTOR Phase 1 research project in Linear. It contains workspace configuration, API patterns, project structure, and common operations. Read this file before performing any Linear operations.

---

## 1. Workspace Configuration

| Field | Value |
|-------|-------|
| Workspace | LM-VECTOR |
| Team ID | `4c93e045-94e9-49b9-a9b8-f68affadea8e` |
| API endpoint | `https://api.linear.app/graphql` |
| Auth header | `Authorization: <API_KEY>` (key stored in env var `LINEAR_API_KEY`) |
| Project name | LM-VECTOR Phase 1 |
| COLM deadline | March 31, 2026 |

**API key location:** The API key is a personal key starting with `lin_api_`. Store it in your environment:
```bash
export LINEAR_API_KEY="lin_api_XXXXX"
```

The key is passed as `Authorization: $LINEAR_API_KEY` — no "Bearer" prefix needed for personal API keys.

---

## 2. Linear GraphQL API Patterns

Linear uses a single GraphQL endpoint. All operations are POST requests to `https://api.linear.app/graphql`.

### Base request pattern (Python)

```python
import json
from urllib.request import Request, urlopen

def gql(query, variables=None):
    """Execute a Linear GraphQL query."""
    body = {"query": query}
    if variables:
        body["variables"] = variables
    req = Request("https://api.linear.app/graphql",
                  data=json.dumps(body).encode("utf-8"),
                  headers={
                      "Content-Type": "application/json",
                      "Authorization": os.environ["LINEAR_API_KEY"],
                  })
    with urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))
```

### Base request pattern (curl)

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: $LINEAR_API_KEY" \
  --data '{"query": "{ viewer { id name } }"}' \
  https://api.linear.app/graphql
```

### Key API facts

- GraphQL only — no REST endpoints
- Rate limit: ~1,500 requests/hour for API keys. Add 200-300ms delays between batch mutations
- Project descriptions: max 255 characters
- Issue descriptions: Markdown supported, including `- [ ]` checkboxes
- Labels: Use `issueUpdate` with full `labelIds` array to modify labels (no `issueAddLabel` mutation exists)
- Cycles: team-scoped, issues can belong to one cycle at a time
- Due dates: ISO 8601 format `YYYY-MM-DD` (called `TimelessDate` in schema)
- Priorities: 0=none, 1=urgent, 2=high, 3=medium, 4=low

---

## 3. Common Queries

### List all issues in the project

```graphql
{
  issues(filter: { project: { name: { eq: "LM-VECTOR Phase 1" } } }) {
    nodes {
      id
      identifier
      title
      state { name type }
      priority
      assignee { name }
      labels { nodes { name } }
      dueDate
      cycle { name }
    }
  }
}
```

### Get a specific issue by identifier (e.g., "TEAM-42")

```graphql
{
  issues(filter: { identifier: { eq: "LMV-42" } }) {
    nodes { id title description state { name } }
  }
}
```

### Get team workflow states

```graphql
query($tid: String!) {
  team(id: $tid) {
    states { nodes { id name type } }
  }
}
```

Typical states: Backlog, Todo, In Progress, Done, Cancelled. The `type` field is one of: `backlog`, `unstarted`, `started`, `completed`, `canceled`.

### Get all labels

```graphql
{
  issueLabels { nodes { id name color } }
}
```

### Get all cycles

```graphql
{
  cycles { nodes { id name number startsAt endsAt } }
}
```

### Get project info

```graphql
{
  projects(filter: { name: { eq: "LM-VECTOR Phase 1" } }) {
    nodes { id name description targetDate }
  }
}
```

---

## 4. Common Mutations

### Update issue status

```graphql
mutation($id: String!, $stateId: String!) {
  issueUpdate(id: $id, input: { stateId: $stateId }) {
    success
    issue { id title state { name } }
  }
}
```

### Update issue priority

```graphql
mutation($id: String!, $priority: Int!) {
  issueUpdate(id: $id, input: { priority: $priority }) {
    success
    issue { id title priority }
  }
}
```

### Assign an issue

```graphql
mutation($id: String!, $assigneeId: String!) {
  issueUpdate(id: $id, input: { assigneeId: $assigneeId }) {
    success
    issue { id title assignee { name } }
  }
}
```

### Add a comment

```graphql
mutation($issueId: String!, $body: String!) {
  commentCreate(input: { issueId: $issueId, body: $body }) {
    success
    comment { id body }
  }
}
```

### Move issue to a cycle

```graphql
mutation($id: String!, $cycleId: String!) {
  issueUpdate(id: $id, input: { cycleId: $cycleId }) {
    success
    issue { id cycle { name } }
  }
}
```

### Update labels (IMPORTANT: must include ALL label IDs, not just new ones)

```graphql
# Step 1: Get current labels
{
  issue(id: "ISSUE_UUID") {
    labels { nodes { id name } }
  }
}

# Step 2: Update with full array (existing + new)
mutation($id: String!, $labelIds: [String!]!) {
  issueUpdate(id: $id, input: { labelIds: $labelIds }) {
    success
    issue { labels { nodes { name } } }
  }
}
```

### Create a new issue

```graphql
mutation($input: IssueCreateInput!) {
  issueCreate(input: $input) {
    success
    issue { id identifier title }
  }
}
```

Input fields: `title`, `description`, `teamId`, `projectId`, `priority`, `labelIds`, `cycleId`, `dueDate`, `assigneeId`, `stateId`.

---

## 5. Project Structure

### Labels (already created)

| Label | Color | Purpose |
|-------|-------|---------|
| Stream 1: Data & Personas | `#2DD4BF` | Work stream |
| Stream 2: Pipeline & Infra | `#818CF8` | Work stream |
| Stream 3: Instruments & Eval | `#FB923C` | Work stream |
| Critical Path | `#EF4444` | Tasks on the critical path |
| Decision Gate | `#FBBF24` | Tasks that evaluate a go/no-go gate |
| Delegatable | `#22C55E` | Tasks that can be delegated to team members |
| Size: XS / S / M / L | `#94A3B8` | T-shirt sizing |

### Cycles (already created)

| Cycle | Dates | Focus |
|-------|-------|-------|
| Week 1: Setup & Foundation | Feb 10-14, 2026 | Environment, data download, instrument assembly |
| Week 2: Build & Generate | Feb 17-21, 2026 | Prompt generation, code refinement, judge pipeline |
| Week 3: Pilot Run & Gate 1 | Feb 24-28, 2026 | 50-persona pilot, adherence scoring, vectors |
| Week 4: Experiments & H1 | Mar 3-7, 2026 | Token/layer experiments, steering validation, H1 test |

### Decision Gates

| Gate | Week | Question | Pass Threshold | Evaluated By |
|------|------|----------|---------------|--------------|
| Gate 1 | 3 | Do personas work? | >60% of personas achieve score-3 on >50% of questions | S2-10 |
| Gate 2 | 4 | Does steering work? | >70% improved adherence, viable coefficient found | S2-14 |
| Gate 3 (H1) | 4 | Occupation-dependent shifts? | ≥2 of 3 instruments show significant occupation-dependent shifts (permutation p<0.05) | S3-08 |

### Team

| Person | Role | Focus |
|--------|------|-------|
| Glenn | Lead | Methodology, analysis, writing. Non-delegatable tasks. |
| Person B | Data pipeline | O*NET data, persona prompt generation, QC |
| Person C | Infrastructure | Model setup, compute ops, generation runs, judge pipeline |
| Person D | Psychometrics | Instrument assembly, scoring code, Likert parsing, evaluation |

---

## 6. All 29 Issues (Task Reference)

Issues are named `[S{stream}-{number}] {title}` in Linear.

### Stream 1: Data & Personas

| ID | Title | Weeks | Days | Assignee | Critical | Gate |
|----|-------|-------|------|----------|----------|------|
| S1-01 | O*NET Data Download & Validation | 1 | 2 | B | ✓ | — |
| S1-02 | RIASEC Stratification Analysis | 1 | 0.5 | B | — | — |
| S1-03 | Persona Prompt Template Design | 1-2 | 4 | Glenn+B | ✓ | — |
| S1-04 | Large-Scale Prompt Generation | 2 | 3 | B | ✓ | — |
| S1-05 | Automated QC Filtering | 2-3 | 1 | B | ✓ | — |
| S1-06 | Non-Occupational Role Library | 2 | 2 | C | — | — |
| S1-07 | Leakage Audit | 2 | 1 | D | — | — |

### Stream 2: Pipeline & Infrastructure

| ID | Title | Weeks | Days | Assignee | Critical | Gate |
|----|-------|-------|------|----------|----------|------|
| S2-01 | Model Download & Environment Setup | 1 | 3 | C | ✓ | — |
| S2-02 | Activation Extraction Code Refinement | 1-2 | 5 | Glenn | ✓ | — |
| S2-03 | Steering Code Validation | 2 | 2 | Glenn | — | — |
| S2-04 | Judge Pipeline Implementation | 2 | 4 | C | — | — |
| S2-05 | Data Format Standardization | 1 | 1 | Glenn | — | — |
| S2-06 | Baseline Response Generation | 3 | 0.5 | C | ✓ | — |
| S2-07 | Baseline Activation Extraction | 3 | 0.5 | C | ✓ | — |
| S2-08 | Pilot Persona Response Generation | 3 | 2 | C | ✓ | — |
| S2-09 | Pilot Activation Extraction | 3 | 4 | C | ✓ | — |
| S2-10 | Pilot Adherence Scoring | 3 | 0.5 | C | ✓ | Gate 1 |
| S2-11 | Persona Vector Computation (Pilot) | 3-4 | 1 | Glenn | ✓ | — |
| S2-12 | Token Position Experiment | 4 | 0.5 | Glenn | ✓ | — |
| S2-13 | Layer Selection Experiment | 4 | 0.5 | Glenn | ✓ | — |
| S2-14 | Steering Validation (Real Vectors) | 4 | 3 | Glenn | ✓ | Gate 2 |

### Stream 3: Instruments & Evaluation

| ID | Title | Weeks | Days | Assignee | Critical | Gate |
|----|-------|-------|------|----------|----------|------|
| S3-01 | IPIP-NEO-120 Assembly & Scoring | 1-2 | 3 | D | — | — |
| S3-02 | HEXACO-100 Assembly & Scoring | 1-2 | 3 | D | — | — |
| S3-03 | O*NET Interest Profiler Assembly | 1 | 2 | D | — | — |
| S3-04 | Likert Response Parsing | 2 | 3 | D | — | — |
| S3-05 | Elicitation Battery Design | 1-2 | 4 | D | — | — |
| S3-06 | Baseline Psychometric Administration | 4 | 1 | D | — | — |
| S3-07 | Steered Psychometric Administration | 4 | 2 | D | ✓ | — |
| S3-08 | H1 Statistical Validation | 4 | 3 | Glenn | ✓ | Gate 3 |

### Critical Path (sequential dependency chain)

```
S2-01 → S2-02 → S2-06 → S2-07 → S2-08 → S2-09 → S2-10 [Gate 1]
                                                    ↓
S2-10 → S2-11 → S2-12 → S2-13 → S2-14 [Gate 2] → S3-07 → S3-08 [Gate 3/H1]
```

### Key Dependencies (non-obvious)

- S1-05 (filtered prompts) and S3-05 (elicitation battery) must finish before S2-08 (pilot generation)
- S1-07 (leakage audit) depends on ALL instruments (S3-01, S3-02, S3-03) AND elicitation battery (S3-05)
- S3-07 (steered psychometrics) needs BOTH S2-14 (steering validated) AND S3-06 (baseline psychometrics)

---

## 7. Common Operations You Might Ask For

Below are the kinds of tasks Glenn is likely to ask for. Use the API patterns above to implement them.

### Status updates
- "Move S2-01 to In Progress"
- "Mark S1-01 as done"
- "What's still in Backlog for Week 1?"

### Reporting
- "Show me all critical path tasks and their status"
- "What's blocked right now?"
- "Give me a weekly status summary"
- "Which tasks are overdue?"

### Bulk operations
- "Move all Week 1 tasks to Done"
- "Add a comment to all Gate tasks with updated thresholds"
- "Reassign Person B's tasks to Person C"

### View building
- "Create a view showing only critical path items"
- "Show me tasks grouped by assignee and week"

### Modifications
- "Add a new task for [something] in Week 3"
- "Change the due date on S2-09 to Feb 27"
- "Update the description of S3-08 with new acceptance criteria"
- "Add a 'Blocked' label to S2-08"

### Dependency tracking
- "What depends on S2-02?"
- "Can we start S2-08 yet? Check if its dependencies are done."
- "What's the longest unfinished chain to Gate 3?"

---

## 8. Views in Linear

Linear views are saved filters. Useful views for this project:

1. **Critical Path** — Filter: label = "Critical Path", sort by due date
2. **This Week** — Filter: cycle = current week's cycle
3. **My Tasks (Glenn)** — Filter: assignee = Glenn
4. **Decision Gates** — Filter: label = "Decision Gate"
5. **Blocked / At Risk** — Filter: status = In Progress AND due date < today

Views can be created via the UI (Linear sidebar → Views → New View) or via the API. For this project, UI creation is fine since you only need 4-5 views.

---

## 9. Slack Integration Notes

Linear's Slack integration supports:
- **Notifications:** Issue status changes post to a configured Slack channel
- **Issue creation from Slack:** Type `/linear create` in Slack
- **Link previews:** Pasting a Linear issue URL unfurls it in Slack

Setup: Linear Settings → Integrations → Slack → Connect.

Recommended channel setup:
- `#lm-vector` — general project discussion
- Linear notifications configured to post status changes here

---

## 10. GitHub Integration Notes

Linear's GitHub integration (already connected per Glenn) supports:
- Linking PRs to issues: Include `LMV-42` in PR title or description
- Auto-closing issues when linked PRs merge (configurable)
- Branch name suggestions from issue identifiers

---

## 11. Important Caveats

1. **No `issueAddLabel` mutation exists.** To add a label, you must first GET current labels, then PUT the full array including the new one.
2. **Project description max 255 chars.** Use issue descriptions for detailed content.
3. **Cycles are team-scoped**, not project-scoped. An issue can be in one cycle at a time.
4. **Linear doesn't have subtasks in the traditional sense.** Use parent-child issue relationships (`parentId` field) if needed, but for this project flat issues with labels work fine.
5. **GraphQL errors return HTTP 200.** Always check `response["errors"]` in addition to status code.
6. **Rate limiting:** ~1,500 req/hr for API keys. For batch operations, add 200-300ms sleeps between mutations.
