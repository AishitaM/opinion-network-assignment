"""Step 1 - read the raw survey export and turn it into a clean numeric matrix.

Outputs
-------
data/processed/responses_numeric.csv   respondents x 60 items, coded -2..+2
data/processed/items.csv               item code, topic, full statement text
data/processed/cleaning_report.json    every number quoted in the report
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

import config as C


def load_raw() -> pd.DataFrame:
    """Read the CSV. utf-8-sig strips the BOM that prefixes the id column."""
    return pd.read_csv(C.RAW_CSV, encoding="utf-8-sig", dtype=str)


def parse_items(columns) -> pd.DataFrame:
    """Split 'T01. Artificial Intelligence will...' into code / topic / text."""
    rows = []
    for col in columns:
        m = re.match(r"^([TESV])(\d{2})\.\s*(.+)$", col.strip())
        if m:
            letter, num, text = m.groups()
            rows.append(
                {
                    "column": col,
                    "code": f"{letter}{num}",
                    "topic": letter,
                    "topic_name": C.TOPICS[letter],
                    "statement": text.strip(),
                }
            )
    return pd.DataFrame(rows)


def encode(raw: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Map Likert labels to integers; everything unrecognised becomes NaN."""
    id_col = [c for c in raw.columns if c.lower().startswith("id")][0]
    mat = raw[items["column"]].copy()
    mat.columns = items["code"].tolist()

    def code_cell(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        s = str(v).strip()
        if s in C.MISSING_TOKENS:
            return np.nan
        return C.LIKERT.get(s, np.nan)

    coded = mat.map(code_cell).astype("Float64")
    coded.index = pd.Index(raw[id_col].astype(str).str.strip(), name="respondent")
    return coded.astype(float)


def main() -> dict:
    raw = load_raw()
    items = parse_items(raw.columns)
    coded = encode(raw, items)

    n_items = coded.shape[1]
    completeness = coded.notna().sum(axis=1) / n_items

    # --- respondent filter: block-wise abandoners are removed as nodes ------
    keep = completeness >= C.MIN_COMPLETENESS
    dropped = completeness[~keep].sort_values()
    clean = coded.loc[keep].copy()

    # --- residual imputation ------------------------------------------------
    # After the filter only a handful of scattered cells remain empty. They are
    # filled with the item's median among respondents who did answer it, which
    # is the neutral choice for an ordinal scale and cannot invent an opinion
    # more extreme than the item already attracts.
    residual_missing = int(clean.isna().sum().sum())
    item_median = clean.median(axis=0, skipna=True)
    imputed_mask = clean.isna()
    clean = clean.fillna(item_median).round().astype(int)

    # --- descriptive statistics used verbatim in the report -----------------
    label_counts = (
        raw[items["column"]].stack(future_stack=True).value_counts(dropna=False).to_dict()
    )
    label_counts = {("(blank)" if pd.isna(k) else str(k)): int(v) for k, v in label_counts.items()}

    item_stats = pd.DataFrame(
        {
            "code": clean.columns,
            "topic": [c[0] for c in clean.columns],
            "mean": clean.mean().values,
            "std": clean.std(ddof=1).values,
            "pct_agree": (clean >= 1).mean().values * 100,
            "pct_disagree": (clean <= -1).mean().values * 100,
            "pct_neutral": (clean == 0).mean().values * 100,
        }
    ).merge(items[["code", "statement", "topic_name"]], on="code")
    item_stats["polarisation"] = item_stats["std"]

    # Completeness of every raw respondent, and the raw answered/not-answered
    # mask - both are plotted directly rather than reconstructed downstream.
    completeness.rename("completeness").to_frame().assign(
        retained=keep, n_answered=coded.notna().sum(axis=1)
    ).to_csv(C.PROCESSED / "respondent_completeness.csv")
    coded.notna().astype(int).to_csv(C.PROCESSED / "answered_mask.csv")

    clean.to_csv(C.PROCESSED / "responses_numeric.csv")
    items.to_csv(C.PROCESSED / "items.csv", index=False)
    item_stats.to_csv(C.TABLES / "item_statistics.csv", index=False)

    report = {
        "raw_respondents": int(raw.shape[0]),
        "n_items": int(n_items),
        "items_per_topic": items["topic"].value_counts().to_dict(),
        "raw_cells": int(raw.shape[0] * n_items),
        "raw_label_counts": label_counts,
        "missing_cells_raw": int(coded.isna().sum().sum()),
        "missing_rate_raw": float(coded.isna().sum().sum() / (raw.shape[0] * n_items)),
        "min_completeness": C.MIN_COMPLETENESS,
        "dropped_respondents": int((~keep).sum()),
        "dropped_detail": {str(k): round(float(v), 3) for k, v in dropped.items()},
        "retained_respondents": int(keep.sum()),
        "residual_missing_cells": residual_missing,
        "residual_missing_rate": float(residual_missing / (keep.sum() * n_items)),
        "imputed_cells_per_item_max": int(imputed_mask.sum().max()),
        "grand_mean": float(clean.to_numpy().mean()),
        "acquiescence_row_mean_min": float(clean.mean(axis=1).min()),
        "acquiescence_row_mean_max": float(clean.mean(axis=1).max()),
        "acquiescence_row_mean_std": float(clean.mean(axis=1).std(ddof=1)),
        "most_consensual": item_stats.nsmallest(5, "std")[["code", "statement", "mean", "std"]]
        .to_dict("records"),
        "most_polarised": item_stats.nlargest(5, "std")[["code", "statement", "mean", "std"]]
        .to_dict("records"),
    }
    (C.PROCESSED / "cleaning_report.json").write_text(json.dumps(report, indent=2))

    print(f"[prepare] raw respondents      : {report['raw_respondents']}")
    print(f"[prepare] items                : {n_items} ({report['items_per_topic']})")
    print(f"[prepare] raw missing rate     : {report['missing_rate_raw']:.1%}")
    print(f"[prepare] dropped respondents  : {report['dropped_respondents']} -> {list(report['dropped_detail'])}")
    print(f"[prepare] retained (nodes)     : {report['retained_respondents']}")
    print(f"[prepare] residual imputed     : {residual_missing} cells "
          f"({report['residual_missing_rate']:.2%})")
    print(f"[prepare] grand mean response  : {report['grand_mean']:+.3f} (agreement bias)")
    return report


if __name__ == "__main__":
    main()
