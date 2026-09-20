"""Central configuration for the Opinion Network Formation pipeline.

Every tunable choice lives here so that the report can cite exact values and
the whole study can be reproduced with a single `python src/run_all.py`.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "Survey_Results_UC.csv"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
FIGS = OUT / "figures"
TABLES = OUT / "tables"
NETWORK = OUT / "network"
REPORT = ROOT / "report"

for _p in (PROCESSED, FIGS, TABLES, NETWORK, REPORT):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- team
TEAM_NAME = "nerds/ammy"
GITHUB_URL = "https://github.com/AishitaM/opinion-network-assignment"
MEMBERS = [
    "Aishita Marri (2023113027)",
    "Abdul Rahman Mujtaba (2023102024)",
    "Anmol Sharma (2024101148)",
]

# ---------------------------------------------------------------- coding
# Symmetric interval coding of the 5-point Likert scale. Zero is placed on
# "Neutral" so that sign carries meaning (agreement vs disagreement) and the
# scale is symmetric, which matters for the centring steps downstream.
LIKERT = {
    "Strongly Disagree": -2,
    "Disagree": -1,
    "Neutral": 0,
    "Agree": 1,
    "Strongly Agree": 2,
}
# Values treated as "no usable opinion recorded".
MISSING_TOKENS = {"", "No Comments", "NA", "N/A", None}

TOPICS = {
    "T": "Technology & AI",
    "E": "Education",
    "S": "Society & Ethics",
    "V": "Environment",
}
TOPIC_COLORS = {"T": "#3b6ea5", "E": "#c2703d", "S": "#4f8f5b", "V": "#8a5fa8"}

# ---------------------------------------------------------------- cleaning
# A respondent must have answered at least this fraction of the 60 items to be
# admitted as a node. The survey shows clean block-wise drop-out, so this
# cleanly separates finishers from abandoners.
MIN_COMPLETENESS = 0.50

# ---------------------------------------------------------------- network
# Edge weight = Pearson correlation between double-centred opinion profiles.
# The retention threshold is calibrated against a permutation null model
# rather than being picked by hand.
NULL_PERMUTATIONS = 500
NULL_PERCENTILE = 95.0       # keep edges stronger than 95% of null edges
FDR_ALPHA = 0.05             # Benjamini-Hochberg level for the significance view
RANDOM_SEED = 42

# Sensitivity sweep reported in the paper.
THRESHOLD_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
