# FFCA development pitfalls (incident-driven, May 2026)

This document exists because the same classes of bugs hit the FFCA case-study
pipeline more than once. Each entry is a real bug we shipped, the cost it
incurred, and the preventive practice (or assertion) that catches it next
time.

If you change anything in `case_studies/`, `rulebook/`, or how FFCA reports
are produced or consumed, read the relevant section here first.

---

## 1. Concurrent shards writing to the same CSV

**Incident.** During the May 25–26 HPC runs of
`compound_flooding_extra_validation.py`, all 7 shards wrote to
`extra_validation_runs/section_<X>.csv` simultaneously. Only the last shard
to finish left its slice in the CSV; every other shard's results were
silently overwritten on disk. Detected post-hoc when §F looked like it had
2 rows instead of 40+. Bit us again on §H (May 26) because the H driver
script inherited the same writer.

**Cost.** Two re-runs of partial sections; one analysis (§F) was almost
performed against incomplete data; we had to write a JSON-rebuild snippet
to recover.

**Why the per-job results were fine.** Each `run_train_job` writes its own
`runs/section_<X>/<exp>/<variant>/results.json` atomically, so the
canonical per-job results survived. The bug was only in the per-section
CSV aggregation.

**Preventive practice (now in code).**

- `compound_flooding_extra_validation.py::run_train_section` takes a
  `shard` argument and writes
  `section_<X>_shard_<k>_of_<N>.csv` when sharded, not
  `section_<X>.csv`.
- `compound_flooding_extra_validation.py::rebuild_section_csv_from_jsons`
  rebuilds the canonical `section_<X>.csv` from the per-job JSONs (the
  authoritative source). It runs automatically in `--finalize-only`.
- Same pattern reproduced in `case_studies/section_FH_redo.py`.

**Rule.** *Never have multiple processes write to the same canonical
filesystem path concurrently.* When sharding, give each shard a unique
suffix. Always finalize from per-job JSONs, not from the per-shard CSVs.

---

## 2. "Matched X" filters that don't actually match X

**Incident.** §F (Nonlinearity dimension validation) selected a high-N
subset and a "matched-Impact, low-N" subset, claiming `matched_impact`.
The actual code computed an Impact range tolerance proportional to the
high-N's *own* Impact range — and because the high-N subset spans a
12× range (rain/gwl features are heterogeneous), the tolerance was wide
enough that ~every feature in the model qualified. Resulting subsets
differed by **36× in mean Impact**. The §F LR/MLP gap result published
in the paper was confounded; we retracted it (commit `2cb6c91`).

**Cost.** Bad result in the published paper for ~24 hours; one retraction
commit; reviewer trust risk.

**Preventive practice.** `case_studies/experiment_utils.py` exposes
`assert_subset_distributions_match()`. Any selection code that names a
property it intends to match MUST call this assertion before
returning the subsets, and the default `max_mean_ratio` is 2.0× — any
real "matched" selection comfortably passes that, and a §F-style
confound trips it instantly.

**Rule.** *If your code or paper says "matched on X", the resulting
distributions on X must be empirically close.* Compute the ratio; assert
it; refuse to proceed if it isn't.

---

## 3. Empty control / null sets returned silently

**Incident.** §H (Hessian-pair synergy) was designed to compare 5
high-Hessian pairs against 5 Impact-matched, low-Hessian control pairs.
The control selector's matching constraint conflicted with the
Catalyst-exclusion constraint (most features in the matched Impact band
were already used by the Hessian pairs), and the selector returned
`control_pairs = []` without raising. The experiment ran 15 hessian-pair
trainings, produced super-additivity numbers, and the analysis script
silently computed statistics on Hessian pairs alone without realising no
control distribution existed.

**Cost.** Re-run with a corrected selector (`section_FH_redo.py` uses
random non-Catalyst pairs); paper retraction of §H (commit `afd4fc8`).

**Preventive practice.** `experiment_utils.py::assert_nonempty()` is a
one-liner that any selection function MUST call before returning a
control/null/baseline set. The function's docstring names §H as the
incident.

**Rule.** *A null distribution that is silently empty is worse than no
test at all.* Always assert the controls/nulls are populated at
selection time, not at analysis time.

---

## 4. Hardcoded paths to directories that get renamed

**Incident.** `tests/test_evaluator.py::test_real_flooding_report_processes_without_error`
hardcoded a path under `compound_flooding/FFCA_resutls_before_prunning/`.
Phase G (May 20) regenerated all FFCA reports under
`FFCA_resutls_before_prunning_ensemble/` and the old directory was
implicitly deprecated. The test silently switched to `pytest.skip()`
because the file no longer existed.

**Cost.** Six weeks of green CI runs that quietly weren't testing the
end-to-end smoke. When the path was fixed, the second bug below (§5)
surfaced immediately.

**Preventive practice.**

- Canonical post-Phase-G FFCA report root:
  `/Users/hnaja002/Documents/projects/compound_flooding/FFCA_resutls_before_prunning_ensemble/`.
  The pre-Phase-G `FFCA_resutls_before_prunning/` is **contaminated** —
  do not read from it for any new test, narration, or analysis.
- Whenever a canonical artifact directory is renamed, the change must
  be accompanied by `grep -rn 'OLD_PATH' .` and an update of every match
  found in tests + scripts + docs.

**Rule.** *Prefer hard fail over silent skip when a fixture goes
missing in a dev environment.* If a test is environmental-fixture
dependent, mark it explicitly (a fixture, an env var) so a missing
artifact is loud, not silent.

---

## 5. Rule renames / splits without updating callers

**Incident.** Phase E (May 19) split the rulebook's
`trust_instability_high` rule into two axis-aware variants:
`trust_instability_high_epoch_axis` (epoch axis) and
`trust_multi_modal_seeds` (seed axis). The flooding smoke test still
referenced the old name. The test was already silently skipping (see §4),
so the rename audit caught nothing at the time; the stale assertion only
surfaced when §4 was fixed.

**Cost.** One additional test edit (`8120bef`) when bug §4 was fixed.

**Preventive practice.** When renaming or splitting any rule:

1. `grep -rn '<old_rule_id>' tests/ case_studies/ docs/` and update
   every match.
2. Add a deprecation marker in `rulebook/ffca_rules.yaml` (the rule's
   `paper_ref` is a good carrier) if the rename is structural — e.g.
   `paper_ref: "renamed from trust_instability_high (Phase E)"`.
3. Bump the rulebook `version:` field. v0.6.0 was the first rulebook
   version after the prose refinements; the rule split happened earlier
   and was not version-bumped, which made the audit harder.

**Rule.** *Rule IDs are public API for the LLM narration layer and for
all downstream tests.* Treat them like Python function names — no
silent removals or renames.

---

## 6. Feature ordering between training and FFCA analysis

**Incident.** Phase G surfaced that
`run_ffca_pruned.build_feature_matrix` was sorting feature channels
alphabetically while original training used `order_input_arrays`
(channels in `input_specifications` order, lag-0 interleaved within
each channel block between -1 and +1). Every compound-flooding FFCA
report shipped before Phase G was reading the wrong column-to-feature
mapping. RMSE sanity check `3.37 cm ≈ paper 3.40 cm` on 24h-gate
confirmed the fix when the ordering was corrected.

**Cost.** Full report regeneration; full retraining of variant_C/D
experiments; rewrite of paper §7; the original 100%-recall headline
correlation in Phase 2 had to be retracted.

**Preventive practice.**

- `case_studies/compound_flooding_extra_validation.py::build_full_feature_matrix`
  is the *canonical* feature-matrix builder. Use it (or directly mirror
  `order_input_arrays`) for any new code that touches the
  feature-matrix column order. Never assume alphabetical sort.
- For any new domain integration of FFCA, the first sanity check is:
  reload the original trained model and run it on a held-out batch; the
  RMSE must match the original training's reported number to within
  ~0.05 of the original. If not, suspect feature ordering before
  suspecting the model.

**Rule.** *The column order of the feature matrix that goes into FFCA
must exactly match the column order the model was trained on.* No
exceptions. Validate by RMSE round-trip before producing any FFCA
report on a new model.

---

## 7. Pre-flight checks before any HPC run

Combining the lessons above, every new HPC experiment script should
include these checks at startup (most are one-liners):

- [ ] **Feature-ordering RMSE round-trip:** load one original model,
      run it on a held-out batch with the new code path, assert the
      RMSE matches the model's reported number within tolerance. (§6)
- [ ] **Subset-match assertion:** every selection that names a matched
      property calls `assert_subset_distributions_match()`. (§2)
- [ ] **Non-empty controls:** every control / null set is checked via
      `assert_nonempty()` before any training job is queued. (§3)
- [ ] **Per-shard CSV path:** the section CSV writer uses the
      `_shard_<k>_of_<N>` suffix when `--shard` is set. (§1)
- [ ] **Canonical report root:** all FFCA report reads point at
      `FFCA_resutls_before_prunning_ensemble/`, never the pre-Phase-G
      directory. (§4)
- [ ] **Rule-ID grep:** if any rulebook rule was renamed since the
      script's last revision, grep this script for the old ID. (§5)

---

## Incident timeline

| Date | Bug | Where | Commit / artifact |
|---|---|---|---|
| 2026-05-19 | Phase E: seed-vs-epoch rule misapplication | rulebook + FFCA pkg | rule split + ensemble mode |
| 2026-05-20 | Phase G: feature-ordering mismatch | `run_ffca_pruned.build_feature_matrix` | report + variant regeneration |
| 2026-05-25 | Sharded CSV overwrite | §A-G CSV writer | per-job JSON rebuild |
| 2026-05-26 | Sharded CSV overwrite (again) | §H CSV writer | per-job JSON rebuild |
| 2026-05-26 | §F "matched Impact" filter too loose | `build_section_f_jobs` | retract paper §F, `2cb6c91` |
| 2026-05-26 | §H control selector returns empty | `pick_hessian_pairs` controls | retract paper §H, `afd4fc8` |
| 2026-05-26 | Flooding smoke test stale path | `tests/test_evaluator.py` | `8120bef` |
| 2026-05-26 | Flooding smoke test stale rule ID | `tests/test_evaluator.py` | `8120bef` |

When you fix the next one of these, add a row.
