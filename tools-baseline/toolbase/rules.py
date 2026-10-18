"""Platon's three documented filter modes, re-derived from one `--characterize` run.

Platon ships three filter modes. `sensitivity` and `specificity` are pure
thresholds on the replicon distribution score (RDS); `accuracy`, the default,
combines those thresholds with a disjunction of plasmid hallmarks. Because
`--characterize` reports RDS and every hallmark column for *all* contigs, the
three modes can be evaluated from a single run instead of three, and the RDS
itself becomes a continuous score for threshold-free metrics (AUROC/AUPRC).

The cutoffs and the hallmark disjunction below are transcribed from the Platon
documentation (https://gensoft.pasteur.fr/docs/platon/1.7/):

  sensitivity   RDS < -7.9 -> chromosome, everything else plasmid   (95% sensitivity)
  specificity   RDS >  0.7 -> plasmid, everything else chromosome   (99.9% specificity)
  accuracy      RDS < -7.9 -> chromosome
                RDS >  0.7 -> plasmid
                otherwise plasmid if the contig can be circularised, OR has an
                incompatibility group sequence, OR has a replication or
                mobilization HMM hit, OR has an oriT hit, OR has RDS > 0.1 with a
                RefSeq plasmid hit and no rRNA hit
                otherwise chromosome

`server/13_smoke_tools.sh` checks this transcription against a real
`platon --mode accuracy` run on two shards and fails the pipeline if the labels
disagree, so the saving above is never taken on trust.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import parse

SENSITIVITY_CUTOFF = -7.9
SPECIFICITY_CUTOFF = 0.7
ACCURACY_RDS_FLOOR = 0.1

MODES = ("sensitivity", "accuracy", "specificity")

# Columns each mode needs on top of PLATON_REQUIRED.
MODE_COLUMNS = {
    "sensitivity": (),
    "specificity": (),
    "accuracy": ("circularity", "inc_types", "n_replication", "n_mobilization",
                 "n_orit", "n_rrna", "n_plasmid_hits"),
}


def missing_for_mode(df: pd.DataFrame, mode: str) -> list[str]:
    return [c for c in MODE_COLUMNS[mode] if c not in df.columns]


def _hallmarks(df: pd.DataFrame) -> np.ndarray:
    """The accuracy mode's disjunction of plasmid-specific evidence."""
    n = len(df)
    hit = np.zeros(n, dtype=bool)
    if "circularity" in df.columns:
        hit |= parse.is_circular(df["circularity"])
    if "inc_types" in df.columns:
        hit |= parse.is_present(df["inc_types"])
    for col in ("n_replication", "n_mobilization", "n_orit"):
        if col in df.columns:
            hit |= df[col].to_numpy() > 0
    # The last clause is a conjunction: a weak RDS plus a database hit counts only
    # when no rRNA gene is present, because rRNA is chromosomal evidence.
    if all(c in df.columns for c in ("n_plasmid_hits", "n_rrna")):
        hit |= ((df["rds"].to_numpy() > ACCURACY_RDS_FLOOR)
                & (df["n_plasmid_hits"].to_numpy() > 0)
                & (df["n_rrna"].to_numpy() == 0))
    return hit


def platon_labels(df: pd.DataFrame, mode: str) -> np.ndarray:
    """1 = plasmid, 0 = chromosome, for one filter mode. NaN RDS -> chromosome.

    A NaN RDS means Platon found no marker protein hits at all on that contig,
    which its own sensitivity rule treats as chromosomal.
    """
    if mode not in MODES:
        raise ValueError(f"unknown Platon mode {mode!r}; expected one of {MODES}")
    missing = missing_for_mode(df, mode)
    if missing:
        raise parse.SchemaError(
            f"Platon mode '{mode}' needs column(s) {missing}, which the parsed TSV "
            f"does not carry (it has {sorted(df.columns)}). Re-run Platon with "
            f"--characterize so every hallmark column is reported for every contig."
        )
    rds = pd.to_numeric(df["rds"], errors="coerce").to_numpy()
    below = rds < SENSITIVITY_CUTOFF
    above = rds > SPECIFICITY_CUTOFF
    nan = np.isnan(rds)

    if mode == "sensitivity":
        return np.where(nan | below, 0, 1).astype(np.int8)
    if mode == "specificity":
        return np.where(above, 1, 0).astype(np.int8)
    label = np.where(above, 1, 0)
    middle = ~below & ~above & ~nan
    label = np.where(middle & _hallmarks(df), 1, label)
    return np.where(nan, 0, label).astype(np.int8)


def mobsuite_labels(df: pd.DataFrame) -> np.ndarray:
    """1 where mob_recon's molecule_type says Plasmid, 0 for Chromosome."""
    text = df["molecule_type"].fillna("").astype(str).str.strip().str.lower()
    unknown = sorted(set(text) - {"plasmid", "chromosome", ""})
    if unknown:
        raise ValueError(
            f"mob_recon reported unexpected molecule_type value(s) {unknown}; "
            "the documented values are Plasmid and Chromosome. Check the installed "
            "version before trusting these labels."
        )
    return (text == "plasmid").to_numpy().astype(np.int8)


def mobsuite_score(df: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """A ranking heuristic, NOT a calibrated probability.

    mob_recon returns an assignment, not a probability, so there is nothing to
    plot a real ROC curve from. To still place it on the same axes as the other
    baselines, contigs are ranked by their hard label first and, inside each
    class, by closeness to the nearest plasmid database sketch
    (1 - mash_neighbor_distance). AUROC/AUPRC computed from this are dominated by
    the hard label and must be reported as such - score_tools.py flags every
    entry built this way with score_is_hard_label.
    """
    if "mash_distance" in df.columns:
        closeness = 1.0 - df["mash_distance"].to_numpy(dtype=float)
        closeness = np.clip(np.nan_to_num(closeness, nan=0.0), 0.0, 1.0)
    else:
        closeness = np.zeros(len(df))
    # Mapping label into the integer part and closeness into the fraction keeps the
    # label strictly dominant: every plasmid lands in [0.5, 1.0] and every
    # chromosome below 0.5, so thresholding the score at 0.5 reproduces the hard
    # label exactly while the ordering inside each class stays informative.
    return (labels.astype(float) + 0.999 * closeness) / 2.0
