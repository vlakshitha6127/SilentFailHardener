# SilentFail Hardener

SilentFail Hardener is an offline-first reliability and side-effect verification
system for tool-using refund agents. It compares what an agent reports with
what actually changed in a deterministic ground-truth world, detects silent
failures, and verifies typed hardening wrappers against explicit contracts.

This repository was prepared for the Nebius x NVIDIA Global AI Hackathon.
The local implementation includes an optional Nebius-compatible client, but
the published offline evidence does **not** verify a live Nebius or NVIDIA
endpoint, model behavior, latency, or throughput.

## What it verifies

The system models refund operations with realistic local faults, including:

- timeout after a mutation;
- empty HTTP-200 responses with applied or unapplied effects;
- duplicate webhook-like responses;
- retry and idempotency behavior;
- legitimate repeated or partial refunds;
- email side effects and email faults;
- excessive, ineligible, or incorrect refund requests.

Each episode records both the agent-visible response and the ground-truth
ledger. Detectors keep distinct signals separate:

- duplicate mutation;
- excess mutation;
- incorrect refund amount;
- ineligible mutation;
- silent failure and false success;
- missed action;
- incorrect completion;
- legitimate completion and over-suppression.

The deterministic verifier evaluates typed candidates such as idempotency,
read-back verification, retry policy, and combined wrappers. Negative controls
are included to demonstrate that refusing all actions or blindly deduplicating
legitimate operations is not an acceptable hardening strategy.

## Architecture

```text
Scenario and contract fixtures
          |
          v
Deterministic local agent + isolated world state
          |
          v
Fault-injecting tool registry + causal ledger
          |
          v
Failure detectors and contract validation
          |
          v
Typed hardening wrappers and verifier guards
          |
          v
Structural split validation and statistical reporting
          |
          v
Offline paired results and reproducible evidence
```

The optional `src/silentfail/nebius.py` integration is separate from the
offline harness. The offline harness rejects live model clients and runs only
the deterministic local agent.

## Setup

The project uses Python 3.11, Pydantic v2, and pytest.

### Install

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux or macOS
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Environment configuration

Copy `.env.example` to `.env` only when local configuration is needed. Keep
`.env` private; it is ignored by Git. The deterministic local path does not
require credentials:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux or macOS
cp .env.example .env
```

The optional remote client reads `NEBIUS_API_KEY` when explicitly configured.
No live call is required for the tests or the offline validation harness.

### Run tests

From the repository root:

```bash
py -m pytest -q
```

On systems where `py` is unavailable:

```bash
python -m pytest -q
```

### Run the dashboard locally

```bash
uvicorn src.silentfail.dashboard:app --reload --port 8000
```

The dashboard presents the local ledger timeline, detector metrics, candidate
comparison, and proof data. It should not be interpreted as evidence of live
provider behavior.

## Bounded offline validation

The Phase 4 design uses a proposed cohort of **65 deterministic scenarios**
across 13 classes:

- 39 training scenarios: three per class;
- 26 held-out scenarios: two per class.

This allocation is explicitly a local design decision, not a historical
approval or a live-pilot authorization. Per-class samples are small and do
not support per-class significance claims or acceptance thresholds.

The persisted Phase 4.32 local simulation reported:

| Configuration | Correct | Contract-valid | Contract-invalid |
|---|---:|---:|---:|
| Baseline | 40/65 | 40 | 25 |
| Hardened (`patch_retry_policy`) | 65/65 | 65 | 0 |

The split results reconciled to 24/39 baseline and 39/39 hardened on training
scenarios, and 16/26 baseline and 26/26 hardened on held-out scenarios.
There were no inconclusive episodes.

These are descriptive deterministic-simulation results only. The hardened
100% result applies only to this simulated cohort and configuration; it is not
a real-world reliability guarantee and does not reproduce the historical live
pilot.

The paired validator found 40 eligible pairs, all correct-to-unchanged. The
other 25 pairs were invalid because the baseline contract observation was
invalid. There were zero eligible discordant pairs, so McNemar's exact test
was undefined. The run therefore does not establish a statistically valid
paired efficacy or improvement claim.

The three runtime contract mismatches corrected before the final local run
were:

1. Read-back-confirmed `same_operation_retry` does not require an unsafe
   second request after a committed timeout.
2. Hardened `excess_mutation` records prevention when idempotency/read-back
   safely short-circuits the unsafe second mutation.
3. `incorrect_refund_amount` records the deterministic agent's documented
   full-balance fallback while keeping requested amount, executed amount,
   refund correctness, and detector outcome separate.

The detailed design and handoff boundary are documented in
[`docs/offline-validation-design.md`](docs/offline-validation-design.md).

## Evidence and publication boundary

The repository preserves the original five-scenario evidence and its
provenance checks. Generated evaluation outputs are intentionally excluded
from default publication by `.gitignore`; they may be published only after
an explicit review confirms that they are curated reproducibility evidence.

The following are not part of this publication:

- Copilot session-state outputs and temporary evaluation directories;
- `offline-validation-v2.json`, which has not been created;
- any credentials or local `.env` files;
- claims about historical live-pilot results that are not present and verified.

Live-pilot readiness is **NO-GO**. Before a live pilot can be considered, the
project needs explicit authorization, a verified external launcher and
customer-message transport path, preserved historical-pilot evidence or an
approved replacement evidence plan, and separately approved acceptance
criteria.

## Project structure

```text
src/silentfail/
  agent.py                 Deterministic agent and optional model adapter
  contract_validation.py   Contract, episode, pair, and report validation
  dashboard.py             Local FastAPI dashboard
  detectors.py             Side-effect and completion detectors
  ledger.py                Ground-truth state and causal ledger
  nebius.py                Optional Nebius-compatible client and gate
  offline_harness.py       Offline-only execution and manifest gate
  orchestrator.py          Original end-to-end pipeline coordinator
  patches.py               Typed hardening wrappers
  runner.py                Episode execution and aggregation
  scenarios.py             Scenario and cohort fixtures
  structural_validation.py Split-independence and leakage checks
  tools.py                 Local tools and fault injector

tests/
  test_correctness_and_guards.py
  test_offline_harness.py
  test_pipeline_artifact_sync.py
  test_pre_manifest_gate.py
  test_statistics_reporting.py
  test_structural_validation.py
  test_silentfail.py
  test_model_proposals.py
  test_p1_generality.py
  test_generated_regression.py
```

## License

See [LICENSE](LICENSE).
