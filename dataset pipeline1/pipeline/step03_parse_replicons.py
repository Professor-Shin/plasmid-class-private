"""Step 3 - parse FASTA records into biological replicons and classify them.

A single assembly FASTA may hold several replicons in arbitrary order:

    assembly
    |- chromosome
    |- chromosome 2
    |- plasmid pA
    |- plasmid pB

Record order carries no meaning, so it is never used. Classification uses, in
order of precedence:

1. the assembly's ``*_assembly_report.txt`` - NCBI's own per-sequence
   ``Assigned-Molecule-Location/Type`` and ``Sequence-Role`` fields;
2. keyword evidence in the FASTA definition line;
3. otherwise the replicon is labelled ``unknown`` and reported - never
   silently forced into one of the two classes.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from .utils import (
    StageLog,
    get_logger,
    load_config,
    path_for,
    read_fasta,
    sha256_sequence,
    update_manifest,
)

CHROMOSOME, PLASMID, UNKNOWN = "chromosome", "plasmid", "unknown"
LABELS = {CHROMOSOME: 0, PLASMID: 1}

# Words that make a record something other than a bacterial chromosome or
# plasmid; they must not be forced into either class.
OTHER_MOLECULE = re.compile(
    r"\b(phage|prophage|virus|viral|mitochondri\w*|chloroplast|plastid|apicoplast|"
    r"provirus|integrative and conjugative element|linear phage)\b",
    re.IGNORECASE,
)
PLASMID_RE = re.compile(r"\b(plasmid|megaplasmid|mega-plasmid)\b", re.IGNORECASE)
CHROMOSOME_RE = re.compile(r"\b(chromosome|chromid)\b", re.IGNORECASE)
COMPLETE_GENOME_RE = re.compile(r"\bcomplete genome\b", re.IGNORECASE)


def parse_assembly_report(path: Path) -> dict[str, dict]:
    """Map RefSeq-Accn -> molecule info from a *_assembly_report.txt.

    Columns (tab separated, after the '# Sequence-Name ...' header):
      Sequence-Name, Sequence-Role, Assigned-Molecule,
      Assigned-Molecule-Location/Type, GenBank-Accn, Relationship,
      RefSeq-Accn, Assembly-Unit, Sequence-Length, UCSC-style-name
    """
    info: dict[str, dict] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            header: list[str] | None = None
            for line in fh:
                line = line.rstrip("\n\r")
                if line.startswith("#"):
                    stripped = line.lstrip("#").strip()
                    if stripped.startswith("Sequence-Name"):
                        header = [c.strip() for c in stripped.split("\t")]
                    continue
                if not line.strip() or header is None:
                    continue
                parts = line.split("\t")
                if len(parts) < len(header):
                    parts += [""] * (len(header) - len(parts))
                rowd = dict(zip(header, parts))
                for key in ("RefSeq-Accn", "GenBank-Accn"):
                    acc = rowd.get(key, "").strip()
                    if acc and acc != "na":
                        info[acc] = rowd
    except OSError:
        return {}
    return info


def classify_from_report(rowd: dict) -> tuple[str, str] | None:
    """Return (replicon_type, evidence) from an assembly-report row."""
    mol_type = (rowd.get("Assigned-Molecule-Location/Type") or "").strip().lower()
    role = (rowd.get("Sequence-Role") or "").strip().lower()
    assigned = (rowd.get("Assigned-Molecule") or "").strip()

    if role and role != "assembled-molecule":
        # unplaced/unlocalized scaffolds are not whole replicons
        return None
    if mol_type == "plasmid":
        return PLASMID, f"assembly_report:Assigned-Molecule-Location/Type=Plasmid({assigned})"
    if mol_type in ("chromosome", "linkage group"):
        return CHROMOSOME, f"assembly_report:Assigned-Molecule-Location/Type=Chromosome({assigned})"
    return None


def classify_from_defline(defline: str, n_records: int, cfg: dict) -> tuple[str, str]:
    """Keyword classification of an NCBI FASTA definition line."""
    desc = defline.split(" ", 1)[1] if " " in defline else ""
    if OTHER_MOLECULE.search(desc) and not PLASMID_RE.search(desc):
        return UNKNOWN, f"defline:non-chromosomal molecule keyword"
    if PLASMID_RE.search(desc):
        return PLASMID, "defline:plasmid keyword"
    if CHROMOSOME_RE.search(desc):
        return CHROMOSOME, "defline:chromosome keyword"
    if (
        n_records == 1
        and COMPLETE_GENOME_RE.search(desc)
        and cfg["replicons"].get("single_record_complete_genome_is_chromosome", True)
    ):
        # A Complete Genome assembly with exactly one record and no plasmid
        # keyword is the chromosome. Flagged explicitly so it stays auditable.
        return CHROMOSOME, "defline:sole record of a complete genome"
    return UNKNOWN, "defline:no decisive keyword"


def sequence_stats(seq: str) -> dict:
    counts = Counter(seq.upper())
    length = len(seq)
    acgt = counts["A"] + counts["C"] + counts["G"] + counts["T"]
    gc = counts["G"] + counts["C"]
    return {
        "length": length,
        "n_count": counts["N"],
        "non_acgt_count": length - acgt,
        "non_acgt_fraction": round((length - acgt) / length, 8) if length else 0.0,
        "gc_content": round(gc / acgt, 6) if acgt else 0.0,
    }


def type_suffix(rep_type: str, index: int) -> str:
    return {CHROMOSOME: f"chr{index}", PLASMID: f"p{index}", UNKNOWN: f"unk{index}"}[rep_type]


def run(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    logger = get_logger(cfg, "step03.replicons")
    slog = StageLog("step03_parse_replicons")

    meta_dir = path_for(cfg, "metadata_dir")
    assemblies = pd.read_csv(meta_dir / "assemblies.csv", dtype=str, keep_default_na=False)
    status = pd.read_csv(meta_dir / "download_status.csv", dtype=str, keep_default_na=False)
    ok = status[status["status"] != "failed"]
    logger.info("Parsing %d downloaded assemblies", len(ok))

    asm_by_acc = assemblies.set_index("assembly_accession").to_dict("index")
    records: list[dict] = []
    n_files = 0

    for _, srow in ok.iterrows():
        acc = srow["assembly_accession"]
        fasta_path = Path(srow["fasta_path"])
        if not fasta_path.exists():
            logger.warning("Missing FASTA for %s - skipped", acc)
            continue
        meta = asm_by_acc.get(acc, {})
        report = (
            parse_assembly_report(Path(srow["assembly_report_path"]))
            if srow.get("assembly_report_path")
            else {}
        )

        parsed = list(read_fasta(fasta_path))
        n_files += 1
        n_records = len(parsed)
        per_type_index: Counter = Counter()

        for order, (defline, seq) in enumerate(parsed):
            seq_id = defline.split()[0]
            decided = None
            rowd = report.get(seq_id) or report.get(seq_id.split(".")[0])
            if rowd is not None and "assembly_report" in cfg["replicons"]["classification_sources"]:
                decided = classify_from_report(rowd)
            if decided is None and "defline" in cfg["replicons"]["classification_sources"]:
                decided = classify_from_defline(defline, n_records, cfg)
            if decided is None:
                decided = (UNKNOWN, "no usable evidence")
            rep_type, evidence = decided

            per_type_index[rep_type] += 1
            stats = sequence_stats(seq)
            records.append(
                {
                    "replicon_id": f"{acc}_{type_suffix(rep_type, per_type_index[rep_type])}",
                    "assembly_accession": acc,
                    "sequence_id": seq_id,
                    "organism_name": meta.get("organism_name", ""),
                    "taxid": meta.get("taxid", ""),
                    "species_taxid": meta.get("species_taxid", ""),
                    "replicon_type": rep_type,
                    "label": LABELS.get(rep_type, ""),
                    "length": stats["length"],
                    "source_fasta": str(fasta_path),
                    "record_order": order,
                    "records_in_assembly": n_records,
                    "classification_evidence": evidence,
                    "classification_source": evidence.split(":")[0],
                    "definition_line": defline[:300],
                    "gc_content": stats["gc_content"],
                    "n_count": stats["n_count"],
                    "non_acgt_count": stats["non_acgt_count"],
                    "non_acgt_fraction": stats["non_acgt_fraction"],
                    "sequence_sha256": sha256_sequence(seq),
                    "assembly_seq_rel_date": meta.get("seq_rel_date", ""),
                    "assembly_refseq_category": meta.get("refseq_category", ""),
                }
            )
        if n_files % 50 == 0:
            logger.info("Parsed %d/%d assemblies, %d replicons so far", n_files, len(ok), len(records))

    rep = pd.DataFrame(records)
    rep = rep.sort_values(["assembly_accession", "record_order"]).reset_index(drop=True)
    out = meta_dir / "replicons.csv"
    rep.to_csv(out, index=False)

    counts = rep["replicon_type"].value_counts().to_dict()
    n_unknown = int(counts.get(UNKNOWN, 0))
    logger.info("Wrote %s: %d replicons (chromosome=%d plasmid=%d unknown=%d)",
                out, len(rep), counts.get(CHROMOSOME, 0), counts.get(PLASMID, 0), n_unknown)

    if n_unknown:
        unk_path = path_for(cfg, "reports_dir") / "unknown_replicons.csv"
        rep[rep["replicon_type"] == UNKNOWN].to_csv(unk_path, index=False)
        logger.warning("%d unknown replicons reported in %s (excluded from the binary "
                       "dataset but retained in replicons.csv)", n_unknown, unk_path)

    slog.record("assemblies parsed", len(ok), n_files)
    slog.record("FASTA records -> replicons", len(rep), len(rep))
    slog.record("classifiable replicons (chromosome|plasmid)", len(rep), len(rep) - n_unknown,
                f"{n_unknown} unknown kept in replicons.csv, excluded from the ML dataset")
    slog.flush(cfg)

    update_manifest(cfg, "step03_parse_replicons", {
        "assemblies_parsed": n_files,
        "replicons_total": len(rep),
        "chromosomes": int(counts.get(CHROMOSOME, 0)),
        "plasmids": int(counts.get(PLASMID, 0)),
        "unknown": n_unknown,
        "classification_sources": cfg["replicons"]["classification_sources"],
        "classification_source_counts": rep["classification_source"].value_counts().to_dict(),
        "replicons_csv": str(out),
    })
    return rep


if __name__ == "__main__":
    run()
