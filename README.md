# Opinion Network Formation — DPCN Assignment 1

Constructing and analysing a network of student opinions from a 60-item survey
on Technology, Education, Society & Ethics, and Environment.

**Team:** nerds/ammy — Aishita Marri (2023113027), Abdul Rahman Mujtaba (2023102024), Anmol Sharma (2024101148) · **Report:** [`report/report.pdf`](report/report.pdf)

---

## Result in one paragraph

96 students answered 60 Likert statements. Raw agreement similarity is
**degenerate** — 94% of all student pairs score above 0.70 — because the survey
carries a heavy agreement bias (grand mean **+1.14** on a −2…+2 scale). Removing
item popularity and individual response style by **double centring**, then
keeping only pairs whose correlation beats a **permutation null** (τ = 0.205),
yields a connected 87-node, 304-edge network. It is a **small world**
(clustering ×2.87 vs random, σ = 2.74) with **8 detected
communities** (Q = 0.430, z = 9.5 against a degree- and weight-preserving null).
At the settings we used we found eight communities rather than a clear two-group
partition: many small like-minded groups that agree
on values and divide on policy — hardest on **compulsory attendance** (E03) and
**AI regulation** (T12).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/run_all.py          # ~10 s, regenerates every figure and table
```

## Pipeline

| Stage | Script | Produces |
|---|---|---|
| 1. Prepare | `src/prepare_data.py` | numeric matrix, item metadata, cleaning report |
| 2. Build | `src/build_network.py` | similarity matrices, null model, `.graphml` networks |
| 3. Analyse | `src/analyze.py` | metrics, centralities, communities, discriminating items |
| 4. Visualise | `src/visualize.py` | the nine report figures |

`src/config.py` holds every tunable choice (coding scheme, thresholds, seed) so
that no constant is buried in the analysis code.

## Layout

```
data/raw/            the survey export, untouched
data/processed/      coded matrix, item metadata, cleaning report
src/                 the four pipeline stages + config and plot style
outputs/figures/     fig1–fig9 (the report's figures)
outputs/tables/      every table and metric as CSV
outputs/network/     .graphml networks (open in Gephi/Cytoscape)
report/              the submitted report
```

## Method notes

- **Likert coding** is symmetric (−2…+2) so that zero means neutral and the
  centring steps are meaningful.
- **Missing data**: 9 respondents abandoned the survey in whole topic blocks and
  are excluded (<50% completeness); the remaining 1.19% of cells take the item
  median.
- **Double centring** removes item popularity (column z-score) and acquiescence
  bias (row centring) — without it, every student looks similar to every other.
- **The threshold is calibrated, not chosen.** Each item is independently
  shuffled 500 times to build a null correlation distribution; τ is its 95th
  percentile. Connectivity persists up to τ ≈ 0.238, so the selected threshold
  retains a connected network with no isolates.
- **Communities are checked**, not just reported: modularity is compared with
  double-edge-swap randomisations, which preserve each degree *exactly* and carry
  the observed weight multiset, so the null is scored with the same weighted
  Louvain as the observed graph (z = 9.5); the partition is also checked for
  stability over 100 restarts (within-community co-assignment 0.71 vs 0.07).
  The clustering and path-length baselines are looser — Erdős–Rényi graphs
  matched on *n* and *m* only.

## Reproducibility

Every random draw is seeded from `config.RANDOM_SEED = 42`. Re-running
`src/run_all.py` reproduces all reported values exactly; the only variation
across runs is floating-point noise below 1e-15 in `eigenvector_centrality_numpy`
(LAPACK), which affects no reported digit.
