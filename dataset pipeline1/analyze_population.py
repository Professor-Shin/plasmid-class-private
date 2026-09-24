"""Characterise the full filtered assembly population (not just the sample).

Answers two questions about the assemblies that pass the step-1 filter:

1. Are they all really Bacteria? The ``group`` column of
   ``assembly_summary.txt`` says "bacteria" for every row, but that column
   only reflects the FTP directory, and NCBI's refseq/bacteria division also
   distributes Archaea. So the domain is resolved from NCBI Taxonomy itself
   (E-utilities, cached to metadata/taxonomy_species.csv).

2. How many chromosomes and plasmids do they contain? Counting these for
   66k assemblies by downloading their FASTA would be ~100 GB. Instead the
   replicon types come from NCBI's bulk report
   ``GENOME_REPORTS/prokaryotes.txt``, whose "Replicons" column lists each
   molecule as ``chromosome:<acc>`` or ``plasmid <name>:<acc>``.

    python analyze_population.py
"""
from __future__ import annotations

import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pandas as pd

from pipeline.step01_filter_assemblies import load_assembly_summary
from pipeline.utils import get_logger, load_config, path_for, resolve

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
USER_AGENT = "refseq-plasmid-chromosome-pipeline/1.0 (academic dataset build)"
PROKARYOTES_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/GENOME_REPORTS/prokaryotes.txt"


# --------------------------------------------------------------------- taxonomy
def fetch_lineages(taxids: list[str], logger, batch: int = 200) -> pd.DataFrame:
    """Resolve taxid -> (scientific name, lineage, domain) via NCBI Taxonomy."""
    rows: list[dict] = []
    for i in range(0, len(taxids), batch):
        chunk = taxids[i : i + batch]
        data = urllib.parse.urlencode({
            "db": "taxonomy", "id": ",".join(chunk), "retmode": "xml",
        }).encode()
        for attempt in range(1, 4):
            try:
                req = urllib.request.Request(EUTILS, data=data,
                                             headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    root = ET.fromstring(resp.read())
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 3:
                    raise
                logger.warning("efetch batch %d failed (%s), retrying", i // batch, exc)
                time.sleep(2.0 * attempt)

        for taxon in root.findall("Taxon"):
            lineage = (taxon.findtext("Lineage") or "")
            first = lineage.split(";")[0].strip() if lineage else ""
            # NCBI puts "cellular organisms" first; the domain is the next rank.
            parts = [p.strip() for p in lineage.split(";") if p.strip()]
            domain = ""
            for p in parts:
                if p in ("Bacteria", "Archaea", "Eukaryota", "Viruses"):
                    domain = p
                    break
            rows.append({
                "taxid": taxon.findtext("TaxId") or "",
                "scientific_name": taxon.findtext("ScientificName") or "",
                "rank": taxon.findtext("Rank") or "",
                "domain": domain or (first or "unresolved"),
                "lineage": lineage,
            })
        if (i // batch) % 10 == 0:
            logger.info("  taxonomy %d/%d", min(i + batch, len(taxids)), len(taxids))
        time.sleep(0.4)  # stay under the 3 req/s guideline for unauthenticated use
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- replicons
REPLICON_RE = re.compile(r"^\s*([^:]+?)\s*:", re.IGNORECASE)


def parse_replicons(field: str) -> tuple[int, int, int]:
    """Count (chromosomes, plasmids, other) in a prokaryotes.txt Replicons field."""
    if not field or field == "-":
        return 0, 0, 0
    chrom = plasmid = other = 0
    for part in field.split(";"):
        m = REPLICON_RE.match(part)
        name = (m.group(1) if m else part).strip().lower()
        if name.startswith("plasmid"):
            plasmid += 1
        elif name.startswith("chromosome") or name.startswith("chrom"):
            chrom += 1
        else:
            other += 1
    return chrom, plasmid, other


def main() -> int:
    cfg = load_config()
    logger = get_logger(cfg, "analyze_population")
    meta = path_for(cfg, "metadata_dir")
    reports = path_for(cfg, "reports_dir")

    # ------------------------------------------------- reproduce the filter
    summary_path = resolve(cfg, cfg["source"]["assembly_summary"])
    df = load_assembly_summary(summary_path, logger)
    f = cfg["filter"]
    cur = df[(df["assembly_level"] == f["assembly_level"])
             & (df["version_status"] == f["version_status"])]
    if f.get("group"):
        cur = cur[cur["group"] == f["group"]]
    if f.get("require_refseq_accession", True):
        cur = cur[cur["assembly_accession"].str.startswith("GCF_")]
    if f.get("require_ftp_path", True):
        cur = cur[(cur["ftp_path"] != "na") & (cur["ftp_path"].str.len() > 0)]
    if f.get("drop_excluded_from_refseq", True):
        cur = cur[cur["excluded_from_refseq"].isin(["na", ""])]
    filtered = cur.reset_index(drop=True)
    logger.info("Filtered population: %d assemblies", len(filtered))

    # ------------------------------------------------------------- taxonomy
    cache = meta / "taxonomy_species.csv"
    species = sorted(set(filtered["species_taxid"]))
    if cache.exists():
        tax = pd.read_csv(cache, dtype=str, keep_default_na=False)
        missing = sorted(set(species) - set(tax["taxid"]))
        if missing:
            logger.info("Fetching %d new taxids", len(missing))
            tax = pd.concat([tax, fetch_lineages(missing, logger)], ignore_index=True)
            tax.to_csv(cache, index=False)
    else:
        logger.info("Resolving %d distinct species_taxid via NCBI Taxonomy", len(species))
        tax = fetch_lineages(species, logger)
        tax.to_csv(cache, index=False)
    logger.info("Taxonomy cache: %s (%d taxids)", cache, len(tax))

    dom = dict(zip(tax["taxid"], tax["domain"]))
    filtered["domain"] = filtered["species_taxid"].map(dom).fillna("unresolved")

    # ------------------------------------------------------------- replicons
    prok_path = resolve(cfg, "data/raw/prokaryotes.txt")
    if not prok_path.exists():
        logger.error("Missing %s - download it from %s", prok_path, PROKARYOTES_URL)
        return 1
    prok = pd.read_csv(prok_path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=3, on_bad_lines="warn",
                       usecols=["Assembly Accession", "Replicons", "Status"])
    prok = prok[prok["Assembly Accession"] != "-"]
    logger.info("prokaryotes.txt rows usable: %d", len(prok))

    # assembly_summary gives the paired GenBank accession for each GCF_ record.
    filtered["gca"] = filtered["gbrs_paired_asm"]
    prok_by_gca = dict(zip(prok["Assembly Accession"], prok["Replicons"]))
    # Version-insensitive fallback: GCA_000006945.2 -> GCA_000006945
    prok_by_base = {}
    for acc, rep in zip(prok["Assembly Accession"], prok["Replicons"]):
        prok_by_base.setdefault(acc.split(".")[0], rep)

    counts = []
    for gca in filtered["gca"]:
        rep = prok_by_gca.get(gca)
        matched = "exact"
        if rep is None:
            rep = prok_by_base.get(str(gca).split(".")[0])
            matched = "version_insensitive" if rep is not None else "unmatched"
        c, p, o = parse_replicons(rep or "")
        counts.append((c, p, o, matched))
    filtered["n_chromosomes"] = [c[0] for c in counts]
    filtered["n_plasmids"] = [c[1] for c in counts]
    filtered["n_other_replicons"] = [c[2] for c in counts]
    filtered["replicon_match"] = [c[3] for c in counts]

    matched = filtered[filtered["replicon_match"] != "unmatched"]

    # ---------------------------------------------------------------- report
    out = []
    w = out.append
    w("=" * 78)
    w("FILTERED POPULATION CHARACTERISATION")
    w("(all assemblies passing the step-1 filter, not just the sampled subset)")
    w("=" * 78)
    w("")
    w(f"Assemblies passing the filter ....... {len(filtered):,}")
    w(f"Distinct species_taxid .............. {filtered['species_taxid'].nunique():,}")
    w("")
    w("-- 1. DOMAIN (resolved from NCBI Taxonomy, not the 'group' column) --------")
    w("")
    w("   NOTE: the assembly_summary 'group' column reads 'bacteria' for every")
    w("   row, but it reflects the FTP directory. NCBI's refseq/bacteria")
    w("   division also distributes Archaea, so 'group' cannot establish the")
    w("   domain on its own.")
    w("")
    dom_counts = filtered["domain"].value_counts()
    for d, n in dom_counts.items():
        w(f"   {d:<24} {n:>8,} assemblies  ({100*n/len(filtered):5.2f}%)"
          f"   {filtered[filtered['domain']==d]['species_taxid'].nunique():>6,} species")
    w("")
    w("-- 2. REPLICON CONTENT (from GENOME_REPORTS/prokaryotes.txt) -------------")
    w("")
    w(f"   Assemblies matched to the bulk report  {len(matched):,} "
      f"({100*len(matched)/len(filtered):.2f}%)")
    w(f"   Unmatched (no replicon breakdown) ..... {len(filtered)-len(matched):,}")
    w("")
    w(f"   Total chromosomes ..................... {int(matched['n_chromosomes'].sum()):,}")
    w(f"   Total plasmids ........................ {int(matched['n_plasmids'].sum()):,}")
    w(f"   Total other/unnamed replicons ......... {int(matched['n_other_replicons'].sum()):,}")
    tot = int(matched["n_chromosomes"].sum() + matched["n_plasmids"].sum()
              + matched["n_other_replicons"].sum())
    w(f"   Total replicons ....................... {tot:,}")
    if tot:
        w(f"   Plasmid share of replicons ............ "
          f"{100*matched['n_plasmids'].sum()/tot:.2f}%")
    w("")
    w(f"   Assemblies with >=1 plasmid ........... {int((matched['n_plasmids']>0).sum()):,}"
      f"  ({100*(matched['n_plasmids']>0).mean():.2f}%)")
    w(f"   Assemblies with no plasmid ............ {int((matched['n_plasmids']==0).sum()):,}")
    w(f"   Assemblies with >1 plasmid ............ {int((matched['n_plasmids']>1).sum()):,}")
    w(f"   Assemblies with >1 chromosome ......... {int((matched['n_chromosomes']>1).sum()):,}")
    w(f"   Max plasmids in one assembly .......... {int(matched['n_plasmids'].max()):,}")
    w("")
    w("   Plasmids per assembly:")
    for k, n in matched["n_plasmids"].value_counts().sort_index().head(12).items():
        w(f"     {k:>3} plasmid(s)  {n:>8,} assemblies")
    w("")
    # -------------------------------------------------- full-population estimate
    # prokaryotes.txt is a periodic snapshot and lags assembly_summary.txt, so
    # the newest assemblies have no row in it. The assembly_summary's own
    # `replicon_count` column covers all 66k and agrees with the bulk report's
    # replicon total on the matched subset, so it carries the extrapolation.
    filtered["replicon_count_i"] = pd.to_numeric(filtered["replicon_count"],
                                                 errors="coerce").fillna(0).astype(int)
    m2 = filtered[filtered["replicon_match"] != "unmatched"]
    u2 = filtered[filtered["replicon_match"] == "unmatched"]
    bulk_total = (m2["n_chromosomes"] + m2["n_plasmids"] + m2["n_other_replicons"])
    agree = float((m2["replicon_count_i"] == bulk_total).mean()) if len(m2) else 0.0
    chr_rate = m2["n_chromosomes"].sum() / len(m2) if len(m2) else 0.0
    other_rate = m2["n_other_replicons"].sum() / len(m2) if len(m2) else 0.0

    est_chr = chr_rate * len(u2)
    est_other = other_rate * len(u2)
    est_plasmid = u2["replicon_count_i"].sum() - est_chr - est_other

    w("-- 3. FULL-POPULATION ESTIMATE -------------------------------------------")
    w("")
    w("   The bulk report is a periodic snapshot that lags assembly_summary.txt,")
    w("   so the newest assemblies are simply absent from it. The extrapolation")
    w("   below rests on assembly_summary's own `replicon_count` column, which")
    w(f"   reproduces the bulk report's replicon total for {100*agree:.2f}% of the")
    w("   matched assemblies - so TOTAL replicons are near-exact for all 66,230,")
    w("   and only the chromosome/plasmid SPLIT of the unmatched part is inferred.")
    w("")
    w(f"   Measured on the matched {len(m2):,}: {chr_rate:.4f} chromosomes per assembly")
    w("")
    w(f"   {'':<26}{'exact (matched)':>18}{'estimated (rest)':>18}{'population':>14}")
    w(f"   {'assemblies':<26}{len(m2):>18,}{len(u2):>18,}{len(filtered):>14,}")
    w(f"   {'total replicons':<26}{int(bulk_total.sum()):>18,}"
      f"{int(u2['replicon_count_i'].sum()):>18,}"
      f"{int(bulk_total.sum() + u2['replicon_count_i'].sum()):>14,}")
    w(f"   {'chromosomes':<26}{int(m2['n_chromosomes'].sum()):>18,}{est_chr:>18,.0f}"
      f"{int(m2['n_chromosomes'].sum()) + est_chr:>14,.0f}")
    w(f"   {'plasmids':<26}{int(m2['n_plasmids'].sum()):>18,}{est_plasmid:>18,.0f}"
      f"{int(m2['n_plasmids'].sum()) + est_plasmid:>14,.0f}")
    w("")
    w("   Caveat: the unmatched assemblies are systematically NEWER, and plasmid")
    w("   carriage varies by taxon and by sequencing era, so the estimated half")
    w("   carries real uncertainty. Only the 'exact' column is measured.")
    w("")
    w("-- 4. BY DOMAIN ----------------------------------------------------------")
    w("")
    w(f"   {'domain':<22}{'assemblies':>12}{'chromosomes':>13}{'plasmids':>11}"
      f"{'% w/ plasmid':>14}")
    for d in dom_counts.index:
        sub = matched[matched["domain"] == d]
        if not len(sub):
            continue
        w(f"   {d:<22}{len(sub):>12,}{int(sub['n_chromosomes'].sum()):>13,}"
          f"{int(sub['n_plasmids'].sum()):>11,}{100*(sub['n_plasmids']>0).mean():>13.2f}%")
    w("")
    w("=" * 78)

    text = "\n".join(out)
    print(text)
    (reports / "population_characterisation.txt").write_text(text + "\n", encoding="utf-8")
    keep = ["assembly_accession", "organism_name", "taxid", "species_taxid", "domain",
            "refseq_category", "replicon_count", "n_chromosomes", "n_plasmids",
            "n_other_replicons", "replicon_match", "gca"]
    filtered[keep].to_csv(reports / "population_replicons.csv", index=False)
    logger.info("Wrote reports/population_characterisation.txt and population_replicons.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
