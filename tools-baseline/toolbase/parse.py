"""Readers for Platon's TSV and MOB-recon's contig_report.txt.

Both tools have changed their output headers between releases, and a silently
mis-read column would produce a plausible-looking but wrong baseline. So nothing
here reads by column position: headers are normalised (lowercased, punctuation
stripped) and matched against alias sets, every column a caller needs is asserted
to exist, and a missing one raises with the header that was actually seen.

`server/logs/tool_help_<date>.txt`, written by 00_setup_envs.sh, records the CLI
and version of the installed tools and is the reference when an assertion here
fires.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def _norm(name: str) -> str:
    """'# OriT Sequences' -> 'oritsequences'; 'Inc Type(s)' -> 'inctypes'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# Canonical name -> header spellings seen across releases, already normalised.
PLATON_ALIASES: dict[str, set[str]] = {
    "contig_id":      {"id", "contigid", "contig", "sequenceid", "sequence"},
    "length":         {"length", "contiglength"},
    "coverage":       {"coverage", "contigcoverage"},
    "n_orfs":         {"orfs", "norfs", "numorfs", "numberoforfs"},
    "rds":            {"rds", "replicondistributionscore"},
    "circularity":    {"circularity", "circular", "iscircular"},
    "inc_types":      {"inctypes", "inctype", "incompatibilitytypes",
                       "incompatibilitytype", "incompatibilitygroups"},
    "n_replication":  {"replication", "replicationgenes", "numreplicationgenes"},
    "n_mobilization": {"mobilization", "mobilizationgenes", "nummobilizationgenes"},
    "n_orit":         {"orit", "orits", "oritsequences", "numoritsequences"},
    "n_conjugation":  {"conjugation", "conjugationgenes", "numconjugationgenes"},
    "n_rrna":         {"rrna", "rrnas", "rrnagenes", "numrrnagenes"},
    "n_plasmid_hits": {"plasmiddatabasehits", "plasmidhits", "plasmiddbhits",
                       "plasmiddatabasehit", "refseqplasmidhits"},
}

MOB_ALIASES: dict[str, set[str]] = {
    "contig_id":        {"contigid", "contig", "queryid", "id", "seqid", "sequenceid"},
    "sample_id":        {"sampleid", "sample"},
    "molecule_type":    {"moleculetype", "molecule"},
    "primary_cluster":  {"primaryclusterid", "primarycluster", "clusterid"},
    "secondary_cluster": {"secondaryclusterid", "secondarycluster"},
    "size":             {"size", "length", "contigsize"},
    "gc":               {"gc", "gccontent"},
    "circularity":      {"circularitystatus", "circularity", "circular"},
    "rep_types":        {"reptypes", "reptype", "reptypes2", "replicontypes"},
    "relaxase_types":   {"relaxasetype", "relaxasetypes"},
    "mpf_type":         {"mpftype", "mpftypes"},
    "orit_types":       {"orittypes", "orittype"},
    "predicted_mobility": {"predictedmobility", "mobility"},
    "mash_neighbor":    {"mashnearestneighbor", "mashnearestneighbour",
                         "mashneighbor", "mashneighbour"},
    "mash_distance":    {"mashneighbordistance", "mashneighbourdistance",
                         "mashdistance", "mashdist"},
    "mash_identification": {"mashneighboridentification", "mashidentification"},
}

# Columns without which a result cannot be interpreted at all.
PLATON_REQUIRED = ("contig_id", "rds")
MOB_REQUIRED = ("contig_id", "molecule_type")


def _resolve(header, aliases: dict[str, set[str]]) -> dict[str, str]:
    """Map canonical name -> the actual column label present in this file."""
    seen = {_norm(c): c for c in header}
    found = {}
    for canonical, spellings in aliases.items():
        for norm, original in seen.items():
            if norm in spellings:
                found[canonical] = original
                break
    return found


class SchemaError(RuntimeError):
    """A tool's output does not carry a column this project depends on."""


def _rename(df: pd.DataFrame, aliases, required, what: str, path: Path) -> pd.DataFrame:
    found = _resolve(df.columns, aliases)
    missing = [c for c in required if c not in found]
    if missing:
        raise SchemaError(
            f"{what} ({path}) is missing required column(s) {missing}.\n"
            f"  header seen: {list(df.columns)}\n"
            f"  This usually means the installed tool version changed its output "
            f"schema. Check server/logs/tool_help_*.txt for the installed version "
            f"and extend the alias sets in tools-baseline/toolbase/parse.py."
        )
    out = df.rename(columns={v: k for k, v in found.items()})
    return out[[c for c in aliases if c in out.columns]]


def read_platon_tsv(path: Path) -> pd.DataFrame:
    """One row per contig Platon characterised, with canonical column names.

    Run with `--characterize`, Platon deactivates its filters and reports every
    contig, which is what makes a single run enough for all three filter modes.
    Without that flag the TSV holds only the plasmid-classified contigs.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=list(PLATON_ALIASES))
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = _rename(df, PLATON_ALIASES, PLATON_REQUIRED, "Platon TSV", path)
    df["rds"] = pd.to_numeric(df["rds"], errors="coerce")
    for col in ("length", "coverage", "n_orfs", "n_replication", "n_mobilization",
                "n_orit", "n_conjugation", "n_rrna", "n_plasmid_hits"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def read_mob_contig_report(path: Path) -> pd.DataFrame:
    """One row per contig mob_recon reported, with canonical column names.

    mob_recon drops contigs shorter than its --min_length (1000 bp by default), so
    the set of rows here is NOT guaranteed to cover every input fragment. The
    caller is responsible for re-joining against the full fragment index and
    treating the difference as unclassified rather than letting it vanish.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=list(MOB_ALIASES))
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = _rename(df, MOB_ALIASES, MOB_REQUIRED, "mob_recon contig_report", path)
    if "mash_distance" in df.columns:
        df["mash_distance"] = pd.to_numeric(df["mash_distance"], errors="coerce")
    if "size" in df.columns:
        df["size"] = pd.to_numeric(df["size"], errors="coerce")
    return df


# Values these tools use for "nothing found" in an otherwise textual column.
_EMPTY = {"", "-", "none", "nan", "na", "null", "0"}


def is_present(series: pd.Series) -> np.ndarray:
    """True where a textual hallmark column actually names something."""
    if series is None:
        return np.zeros(0, dtype=bool)
    text = series.fillna("").astype(str).str.strip().str.lower()
    return ~text.isin(_EMPTY)


def is_circular(series: pd.Series) -> np.ndarray:
    """True where a circularity column says the contig could be circularised."""
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.isin({"yes", "y", "true", "1", "circular", "circularized",
                      "circularised", "complete"})


# PLASMe writes two files per run. `<output>_report.csv` is TAB-separated despite
# the name and holds only the contigs it CALLED plasmid, which is useless for a
# denominator. `<temp>/PLASMe_candidate.csv` is the real result: one comma-
# separated row for every contig that got a BLASTn hit against the reference
# plasmid database, carrying the identity and coverage of that hit and, where the
# Transformer stage ran, its probability. Every published decision mode is a
# threshold pair over those three numbers, so one run is enough for all of them -
# the same reason Platon is run once with --characterize.
PLASME_ALIASES: dict[str, set[str]] = {
    "order":             {"order", "taxonomy", "taxonomicorder"},
    "contig_id":         {"query", "contig", "contigid", "id", "sequenceid", "seqid"},
    "identity":          {"identity", "ident", "pident", "blastnidentity"},
    "coverage":          {"coverage", "cov", "qcov", "blastncoverage"},
    "transformer_score": {"plasme", "score", "probability", "prob", "predscore"},
    "overlap":           {"overlap", "ambregion", "ambiguousregion", "ambregions"},
}

# PlasmidHunter's predictions.tsv is written with `DataFrame.to_csv(sep='\t')`, so
# the contig id is the unnamed index column and is handled separately below.
PLASMIDHUNTER_ALIASES: dict[str, set[str]] = {
    "pred":              {"prediction0chromosome1plasmid", "prediction", "pred",
                          "predictedclass", "label"},
    "proba_chromosome":  {"probabilityof0", "probability0", "probchromosome",
                          "probabilitychromosome"},
    "proba_plasmid":     {"probabilityof1", "probability1", "probplasmid",
                          "probabilityplasmid"},
}

PLASME_REQUIRED = ("contig_id", "identity", "coverage", "transformer_score")
PLASMIDHUNTER_REQUIRED = ("pred", "proba_plasmid")


def read_plasme_candidate(path: Path) -> pd.DataFrame:
    """One row per contig PLASMe aligned to its reference database.

    A contig that produced no BLASTn hit at all never reaches this file, and a
    contig whose proteins were not run through the Transformer carries
    `transformer_score == -1`. Both are distinctions the collector has to keep:
    the first is "PLASMe declined to classify it", the second is "classified, but
    by BLASTn alone, so there is no probability to put on a ROC curve".
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=list(PLASME_ALIASES))
    df = pd.read_csv(path, dtype=str)
    df = _rename(df, PLASME_ALIASES, PLASME_REQUIRED, "PLASMe candidate table", path)
    for col in ("identity", "coverage", "transformer_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_plasme_report(path: Path) -> pd.DataFrame:
    """PLASMe's own `<output>_report.csv`: the contigs it called plasmid.

    Only used to cross-check the offline re-derivation of the decision rule
    (server/13_smoke_tools.sh); the scored predictions come from the candidate
    table, which keeps the contigs PLASMe rejected as well.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["contig_id", "length", "reference", "order",
                                     "evidence", "score", "amb_region"])
    # Written with a tab separator under a .csv name - read it as what it is.
    return pd.read_csv(path, sep="\t", dtype=str).rename(columns={"contig": "contig_id"})


def read_plasmidhunter_tsv(path: Path) -> pd.DataFrame:
    """One row per contig PlasmidHunter scored, with canonical column names.

    PlasmidHunter builds its feature vector from DIAMOND hits of the contig's
    predicted proteins, so a contig whose proteins hit nothing in the database
    produces no row at all. As with mob_recon's length cutoff, those contigs must
    be re-joined as unclassified rather than dropped from the denominator.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["contig_id", *PLASMIDHUNTER_ALIASES])
    df = pd.read_csv(path, sep="\t", dtype=str, index_col=0)
    ids = df.index.astype(str)
    df = _rename(df.reset_index(drop=True), PLASMIDHUNTER_ALIASES,
                 PLASMIDHUNTER_REQUIRED, "PlasmidHunter predictions.tsv", path)
    df.insert(0, "contig_id", ids.to_numpy())
    for col in ("pred", "proba_chromosome", "proba_plasmid"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
