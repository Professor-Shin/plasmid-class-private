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


# ---------------------------------------------------------------------- PLASMe --
# PLASMe's three preset modes, transcribed from the `-m/--mode` branch at the
# bottom of PLASMe.py (v1.1) as (identity, coverage, probability):
#
#   high-sensitivity  0.7, 0.1, 0.5
#   balance           0.9, 0.9, 0.5      == the CLI defaults
#   high-precision    0.9, 0.9, 0.9
#
# and the decision itself, from plasme_output():
#
#   plasmid if (identity >= i AND coverage >= c)        - the BLASTn branch
#           or (transformer_score > p)                  - the Transformer branch
#   chromosome otherwise
#
# Note `>` and not `>=` on the probability, and that a contig the Transformer did
# not score carries -1, which can never pass any threshold in [0, 1]. Because all
# three presets are thresholds over the same three numbers, one PLASMe run yields
# all three modes offline - the same saving --characterize buys for Platon, and
# 13_smoke_tools.sh checks this re-derivation against PLASMe's own _report.csv
# before any of it is trusted.
PLASME_MODES: dict[str, tuple[float, float, float]] = {
    "high_sensitivity": (0.7, 0.1, 0.5),
    "balance":          (0.9, 0.9, 0.5),
    "high_precision":   (0.9, 0.9, 0.9),
}

# transformer_score for a contig classified by BLASTn alone.
PLASME_NO_TRANSFORMER = -1.0


def plasme_labels(df: pd.DataFrame, identity: float, coverage: float,
                  probability: float) -> np.ndarray:
    """1 = plasmid, 0 = chromosome, for one (identity, coverage, probability) mode."""
    missing = [c for c in ("identity", "coverage", "transformer_score")
               if c not in df.columns]
    if missing:
        raise parse.SchemaError(
            f"PLASMe's decision rule needs column(s) {missing}, which the parsed "
            f"candidate table does not carry (it has {sorted(df.columns)}). The "
            "scored file must be <temp>/PLASMe_candidate.csv, not the _report.csv "
            "PLASMe writes beside its output FASTA.")
    ident = pd.to_numeric(df["identity"], errors="coerce").to_numpy()
    cov = pd.to_numeric(df["coverage"], errors="coerce").to_numpy()
    prob = pd.to_numeric(df["transformer_score"], errors="coerce").to_numpy()
    blastn = (ident >= identity) & (cov >= coverage)
    transformer = prob > probability
    return np.where(blastn | transformer, 1, 0).astype(np.int8)


def plasme_modes(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: plasme_labels(df, *thr) for name, thr in PLASME_MODES.items()}


# --------------------------------------------------------------- PlasmidHunter --
def plasmidhunter_labels(df: pd.DataFrame) -> np.ndarray:
    """1 = plasmid, 0 = chromosome, straight from PlasmidHunter's own argmax.

    The tool writes both the class and the two class probabilities, and its class
    is the argmax of them. Recomputing it here rather than trusting the column
    would hide a disagreement; comparing them instead turns one into a check on
    the other.
    """
    pred = pd.to_numeric(df["pred"], errors="coerce").to_numpy()
    unknown = sorted(set(pred[~np.isnan(pred)]) - {0.0, 1.0})
    if unknown:
        raise ValueError(
            f"PlasmidHunter reported unexpected class value(s) {unknown}; the "
            "documented values are 0 (chromosome) and 1 (plasmid). Check the "
            "installed version before trusting these labels.")
    labels = np.nan_to_num(pred, nan=0.0).astype(np.int8)
    if {"proba_plasmid", "proba_chromosome"} <= set(df.columns):
        p1 = pd.to_numeric(df["proba_plasmid"], errors="coerce").to_numpy()
        p0 = pd.to_numeric(df["proba_chromosome"], errors="coerce").to_numpy()
        # np.argmax breaks a tie towards the FIRST class, so the equivalent test
        # is a strict comparison, not a threshold at 0.5.
        argmax = (p1 > p0).astype(np.int8)
        disagree = (~np.isnan(p1)) & (~np.isnan(p0)) & (argmax != labels)
        if disagree.any():
            raise ValueError(
                f"{int(disagree.sum())} PlasmidHunter row(s) have a class that is "
                "not the argmax of their own probabilities. The output schema has "
                "changed; re-read PlasmidHunter.predict() before scoring this run.")
    return labels
