# Offline validation cohort design

This document records a proposed design decision for the future
`offline-validation-v2.json` cohort. It is not a historical approval record
and does not authorize cohort generation or evaluation.

## Proposed allocation

The proposed cohort contains 65 scenarios across 13 scenario classes:

- three training scenarios per class (39 total);
- two held-out scenarios per class (26 total).

The allocation is implemented by the read-only cohort builder as a deterministic
3:2 repetition split. Each scenario receives a class- and repetition-specific
ID and a deterministic seed. The held-out fixtures also vary selected message
and initial-state dimensions so that the split is not created by cosmetic IDs
alone.

No earlier repository document authoritatively specifies this allocation. The
README describes an older 50-training / 30-held-out, nine-category workflow,
so this 39/26 allocation must be treated as a new proposed design decision.

Three held-out examples per class would improve per-class coverage, but would
require 39 held-out scenarios and a larger cohort. Two held-out examples per
class provide minimal repeated coverage while preserving representation of
every class in both splits. Per-class estimates remain highly uncertain at
this size; no per-class acceptance threshold or statistical significance
claim is implied.

## Readiness boundary

This proposal remains subject to the mandatory pre-manifest gate. The gate
must validate exact scenario and contract inputs, structural split
independence, contract compatibility, and complete statistical reports before
any v2 manifest can be written.

## Runtime contract decisions

The proposed contracts record observable local-agent behavior rather than
requiring unsupported outcomes:

- `same_operation_retry` permits one request when read-back confirms that a
  timeout-after-success already committed the refund. A second request remains
  permitted only when the effect is unresolved, and must reuse the same
  operation identity.
- `excess_mutation` requires the baseline duplicate mutation but treats the
  second scheduled fault as legitimately unobserved under hardening when
  idempotency/read-back short-circuits the unsafe retry. Prevention is not
  represented as an observed excess mutation.
- `incorrect_refund_amount` records the deterministic agent's documented
  full-balance fallback for an excessive explicit request. The detector still
  reports that the executed amount differed from the customer's requested
  amount; the refund ledger and correctness contract remain separate signals.

These decisions are local simulation semantics, not claims about live-agent
behavior or approval of a live pilot.

## Phase 4 offline validation closeout

The bounded Phase 4 offline validation work is complete. The Phase 4.32 run
executed the proposed deterministic local cohort in `local_simulation` mode
with 65 paired scenarios: 39 training and 26 held-out scenarios across the
13 classes above. This run did not create
`offline-validation-v2.json`, call external services, or invoke model APIs.

### Observed results

The persisted Phase 4.32 results report:

- baseline: 40/65 correct; 40 contract-valid and 25 contract-invalid
  episodes;
- hardened (`patch_retry_policy`): 65/65 correct; 65 contract-valid and zero
  contract-invalid episodes;
- zero inconclusive episodes in either configuration;
- exact split reconciliation: baseline 24/39 and 16/26 correct for train and
  held-out scenarios respectively, versus 39/39 and 26/26 for hardened.

These are descriptive results from a deterministic local simulation. They do
not reproduce the historical live pilot and do not establish a real-world
reliability guarantee. The 100% hardened result applies only to this simulated
cohort and configuration.

The paired validator counted 40 eligible pairs overall (24 train and 16
held-out), all as correct-to-unchanged. The remaining 25 pairs (15 train and
10 held-out) were invalid because the baseline contract observation was
invalid. There were no eligible failed-to-corrected, failed-to-still-failed,
or correct-to-regressed transitions. Consequently, there are zero eligible
discordant pairs and McNemar's exact test is undefined; the run cannot support
a statistically valid paired efficacy or improvement claim.

### Corrected runtime mismatches

Phase 4.32 corrected three runtime/contract mismatches identified by the
initial controlled run:

1. `same_operation_retry` now accepts one request when read-back confirms a
   timeout-after-success committed the refund. A retry remains bounded and
   must preserve the same operation identity when the effect is unresolved.
2. `excess_mutation` records prevention under hardening rather than requiring
   the unsafe second mutation or its scheduled fault to be observed. The
   hardened wrapper may safely short-circuit after read-back/idempotency
   confirmation.
3. `incorrect_refund_amount` records the deterministic agent's documented
   full-balance fallback for an excessive request. Requested amount, executed
   amount, refund correctness, and detector outcome remain separate signals.

These are implementation decisions for the local simulator's observable
semantics. They are not verified assumptions about external launchers,
transport layers, or live-agent behavior.

### Limitations and handoff boundary

The following limitations remain explicit:

- the deterministic offline simulation does not reproduce the historical live
  pilot;
- real external launcher and customer-message transport behavior remains
  unvalidated;
- historical pilot evidence remains outstanding and was not regenerated or
  inferred from the offline run;
- no acceptance threshold, approval, or live-pilot authorization is implied;
- `offline-validation-v2.json` has not been created;
- live-pilot readiness is **NO-GO**.

Before a live pilot can be considered, the project needs explicit
authorization for that phase, a verified external launcher and transport
path, preserved historical-pilot evidence or an approved replacement
evidence plan, and separately defined acceptance criteria. None of those
items is established by this closeout.
