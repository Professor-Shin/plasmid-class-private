#!/usr/bin/env python3
"""Rebuild dataset_mixed/{train,validation,test}.fasta without the laptop.

01_sync_from_laptop.sh is the normal path: it copies the fragment FASTAs from the
machine that generated them. When that machine is unavailable, this script
reconstructs the same files from what IS in git - the split CSVs and
metadata/replicons.csv - by re-fetching each replicon from NCBI and cutting the
fragments again.

Why this reproduces the dataset instead of drifting away from it:

  * A fragment is fully determined by (sequence_id, start, end). step07 cut it as
    `seq[start:end].upper()` and wrote it as `>fragment_id` + 70-column lines, so
    the same slice of the same record gives the same bytes.
  * sequence_id is a VERSIONED accession (NC_007949.1). A versioned record is
    immutable at NCBI: a correction gets a new version. This is the opposite of
    re-running step01/step02, which re-samples assembly_summary.txt and would
    pick a different set of genomes every week.
  * replicons.csv carries sequence_sha256 for every replicon, so each download is
    checked against the hash recorded when the dataset was built, before any
    fragment is cut from it. Length, GC and the non-ACGT fraction of every single
    fragment are then checked against the split CSV.

Anything that fails those checks stops the run: a half-rebuilt dataset would
silently invalidate every number in the report.

Where the sequences come from: the assembly files on ftp.ncbi.nlm.nih.gov, whose
exact filename is recorded in replicons.csv (`source_fasta`). That is 500
gzipped files rather than 908 single-record requests - roughly 650 MB instead of
2.4 GB, and no rate limit to respect. Any record the assembly file does not
provide falls back to Entrez efetch by versioned accession.

Usage (on CU-AIM, from $PROJECT, after `source server/env.sh`):

    python3 server/01b_rebuild_fasta_from_ncbi.py --smoke 3    # prove the route works
    python3 server/01b_rebuild_fasta_from_ncbi.py              # the real rebuild

Options worth knowing:

    --only test            rebuild one split (the tool baselines only need test)
    --smoke N              fetch and verify N assemblies, write no FASTA
    --cache-dir DIR        where downloads are kept (default $PROJECT/tmp/ncbi_replicons)
    --keep-cache           do not print the cache-removal hint at the end
    --source efetch        skip the FTP route and use Entrez for everything
    --dry-run              report what would be fetched, download nothing

Set NCBI_EMAIL so NCBI can contact you about a misbehaving script (their policy
asks for it), and NCBI_API_KEY to raise the efetch rate limit from 3 to 10
requests/second. Stdlib only, so any python3 >= 3.8 can run it; both caches make
a rerun after an interruption cheap.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
FTP_ALL = "https://ftp.ncbi.nlm.nih.gov/genomes/all"
SPLITS = ("train", "validation", "test")
WIDTH = 70          # step07 wrote 70-column FASTA (pipeline/utils.write_fasta_record)


# --------------------------------------------------------------------------- #
# the two primitives copied from the pipeline, so the checks mean something
# --------------------------------------------------------------------------- #

def sha256_sequence(sequence: str) -> str:
    """pipeline/utils.py:sha256_sequence"""
    return hashlib.sha256(sequence.upper().encode("ascii", "replace")).hexdigest()


def base_stats(seq: str) -> tuple[float, float]:
    """pipeline/step07_fragments.py:base_stats, without the numpy dependency."""
    n = len(seq)
    if n == 0:
        return 0.0, 1.0
    up = seq.upper()
    a = up.count("A"); c = up.count("C"); g = up.count("G"); t = up.count("T")
    acgt = a + c + g + t
    gc = (g + c) / acgt if acgt else 0.0
    return round(gc, 6), round((n - acgt) / n, 6)


def write_fasta_record(fh, seq_id: str, sequence: str, width: int = WIDTH) -> None:
    """pipeline/utils.py:write_fasta_record"""
    fh.write(f">{seq_id}\n")
    for i in range(0, len(sequence), width):
        fh.write(sequence[i:i + width] + "\n")


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #

def load_replicons(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = {}
        for r in csv.DictReader(fh):
            rows[r["sequence_id"]] = r
        return rows


def load_fragments(dataset_dir: Path, splits) -> dict[str, list[dict]]:
    out = {}
    for split in splits:
        path = dataset_dir / f"{split}.csv"
        if not path.is_file():
            sys.exit(f"FATAL: {path} is missing. It is tracked in git - run `git pull`.")
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        # fragment_id was assigned by a counter as step07 walked the replicons, so
        # sorting by it restores the original record order AND groups the
        # fragments of one replicon together - one load per replicon below.
        rows.sort(key=lambda r: r["fragment_id"])
        out[split] = rows
        print(f"  {split:<11} {len(rows):>7,} fragments from {path.name}")
    return out


# --------------------------------------------------------------------------- #
# fetching
# --------------------------------------------------------------------------- #

def fetch_one(accession: str, api_key: str | None, email: str | None,
              attempts: int = 5) -> str:
    """Return the sequence of one versioned accession, without its defline."""
    params = {"db": "nuccore", "id": accession, "rettype": "fasta", "retmode": "text",
              "tool": "plasmid-class-rebuild"}
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email
    url = f"{EFETCH}?{urllib.parse.urlencode(params)}"

    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=180) as resp:
                text = resp.read().decode("ascii", "replace")
            break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == attempts:
                raise RuntimeError(f"{accession}: giving up after {attempts} tries ({exc})") from exc
            print(f"    retry {attempt}/{attempts - 1} for {accession} in {delay:.0f}s ({exc})")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)

    lines = [ln.strip() for ln in text.splitlines()]
    if not lines or not lines[0].startswith(">"):
        head = text[:200].replace("\n", " ")
        raise RuntimeError(f"{accession}: NCBI did not return FASTA ({head!r})")
    seq = "".join(ln for ln in lines[1:] if ln and not ln.startswith(">"))
    if not seq:
        raise RuntimeError(f"{accession}: NCBI returned an empty sequence")
    return seq


def assembly_url(assembly_accession: str, source_fasta: str) -> tuple[str, str]:
    """Derive the ftp.ncbi URL of an assembly from what replicons.csv recorded.

    replicons.csv stores the path of the file the pipeline downloaded, e.g.
    ...\\GCF_000012425.1\\GCF_000012425.1_ASM1242v1_genomic.fna.gz. The basename
    carries the assembly name, which is exactly what the FTP layout needs:
    genomes/all/GCF/000/012/425/GCF_000012425.1_ASM1242v1/<basename>
    """
    base = source_fasta.replace("\\", "/").rstrip("/").split("/")[-1]
    suffix = "_genomic.fna.gz"
    if not base.endswith(suffix):
        raise RuntimeError(f"{assembly_accession}: unexpected source_fasta {source_fasta!r}")
    stem = base[: -len(suffix)]
    prefix, _, number = assembly_accession.partition("_")
    digits = number.split(".")[0]
    if len(digits) != 9 or prefix not in ("GCF", "GCA"):
        raise RuntimeError(f"{assembly_accession}: cannot build an FTP path from it")
    return (f"{FTP_ALL}/{prefix}/{digits[0:3]}/{digits[3:6]}/{digits[6:9]}/{stem}/{base}", base)


def http_download(url: str, dest: Path, attempts: int = 4) -> None:
    """Download to dest atomically, retrying with backoff."""
    tmp = dest.with_name(dest.name + ".part")
    delay = 3.0
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "plasmid-class-rebuild"})
            with urllib.request.urlopen(req, timeout=300) as resp, tmp.open("wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            tmp.replace(dest)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                raise FileNotFoundError(url) from exc
            if attempt == attempts:
                raise RuntimeError(f"{url}: giving up after {attempts} tries ({exc})") from exc
            print(f"    retry {attempt}/{attempts - 1} in {delay:.0f}s ({exc})")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)


def read_gz_fasta(path: Path):
    """Yield (record_id, sequence) from a gzipped multi-record FASTA."""
    rid, buf = None, []
    with gzip.open(path, "rt", encoding="ascii", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if rid is not None:
                    yield rid, "".join(buf)
                rid, buf = line[1:].split()[0], []
            elif line:
                buf.append(line)
    if rid is not None:
        yield rid, "".join(buf)


def load_recorded_urls(path: Path) -> dict[str, str]:
    """The URL step02 actually downloaded each assembly from, if it is on hand.

    metadata/download_status.csv is tracked, so the exact URL and the fact that
    its NCBI md5 checksum was verified at build time travel with the repository.
    Using it beats deriving the path, even though the two agree for all 500.
    """
    if not path.is_file():
        return {}
    out = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            url = (r.get("url") or "").strip()
            if url:
                out[r["assembly_accession"]] = url
    return out


def fetch_assembly(assembly_accession: str, source_fasta: str, wanted: dict[str, dict],
                   cache_dir: Path, asm_dir: Path, keep_assembly: bool,
                   recorded_url: str | None = None) -> tuple[set[str], int]:
    """Download one assembly and cache every wanted record it verifies.

    Returns (accessions written, bytes downloaded). A record whose sha256 does
    not match replicons.csv is left out, so the caller falls back to efetch and
    the mismatch is reported there rather than silently accepted here.
    """
    derived, base = assembly_url(assembly_accession, source_fasta)
    url = recorded_url or derived
    blob = asm_dir / base
    downloaded = 0
    if not blob.is_file():
        http_download(url, blob)
        downloaded = blob.stat().st_size

    done: set[str] = set()
    for rid, seq in read_gz_fasta(blob):
        meta = wanted.get(rid)
        if meta is None:
            continue
        if sha256_sequence(seq) != meta["sequence_sha256"]:
            continue
        out = cache_dir / f"{rid}.fasta.gz"
        tmp = out.with_name(out.name + ".part")
        with gzip.open(tmp, "wt", encoding="ascii") as fh:
            fh.write(seq)
        tmp.replace(out)
        done.add(rid)

    if not keep_assembly:
        blob.unlink(missing_ok=True)
    return done, downloaded


def cached_sequence(accession: str, cache_dir: Path, expect_sha: str,
                    expect_len: int, api_key, email, pause: float) -> tuple[str, bool]:
    """Return (sequence, downloaded_now). Verifies the hash before caching."""
    blob = cache_dir / f"{accession}.fasta.gz"
    if blob.is_file():
        try:
            with gzip.open(blob, "rt", encoding="ascii") as fh:
                seq = fh.read()
            if sha256_sequence(seq) == expect_sha:
                return seq, False
            print(f"    cached {accession} does not match its recorded hash - refetching")
        except OSError:
            print(f"    cached {accession} is unreadable - refetching")

    seq = fetch_one(accession, api_key, email)
    time.sleep(pause)

    got_sha = sha256_sequence(seq)
    if got_sha != expect_sha:
        raise RuntimeError(
            f"{accession}: sha256 mismatch against replicons.csv\n"
            f"       expected {expect_sha}\n"
            f"       got      {got_sha}\n"
            f"       length expected {expect_len:,}, got {len(seq):,}\n"
            "       The record changed at NCBI (or is a different version). Do NOT\n"
            "       use it: fragments cut from it would not be the published dataset."
        )

    tmp = blob.with_suffix(".gz.part")
    with gzip.open(tmp, "wt", encoding="ascii") as fh:
        fh.write(seq)
    tmp.replace(blob)
    return seq, True


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=os.environ.get("PROJECT"),
                    help="repository root (default: $PROJECT)")
    ap.add_argument("--only", action="append", choices=SPLITS,
                    help="rebuild only this split (repeatable)")
    ap.add_argument("--cache-dir", default=None,
                    help="where downloaded replicons live (default $PROJECT/tmp/ncbi_replicons)")
    ap.add_argument("--keep-cache", action="store_true",
                    help="do not print the cache-removal hint at the end")
    ap.add_argument("--keep-assembly-cache", action="store_true",
                    help="keep the downloaded assembly archives as well as the records")
    ap.add_argument("--source", choices=("ftp", "efetch"), default="ftp",
                    help="ftp (default, 500 gzipped assemblies) or efetch (908 requests)")
    ap.add_argument("--smoke", type=int, metavar="N",
                    help="fetch and verify N assemblies, write no FASTA")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the plan, download nothing")
    args = ap.parse_args()

    if not args.project:
        return print("FATAL: pass --project or `source server/env.sh` first") or 2
    root = Path(args.project).resolve()
    dataset_dir = root / "dataset pipeline1" / "data" / "dataset_mixed"
    replicons_csv = root / "dataset pipeline1" / "data" / "metadata" / "replicons.csv"
    if not replicons_csv.is_file():
        return print(f"FATAL: {replicons_csv} is missing (it is tracked in git)") or 2

    splits = tuple(args.only) if args.only else SPLITS
    cache_dir = Path(args.cache_dir) if args.cache_dir else root / "tmp" / "ncbi_replicons"
    api_key = os.environ.get("NCBI_API_KEY") or None
    email = os.environ.get("NCBI_EMAIL") or None
    pause = 0.11 if api_key else 0.35      # NCBI allows 10/s with a key, 3/s without

    print("=== rebuilding fragment FASTAs from the committed CSVs + NCBI")
    print(f"    repository   {root}")
    print(f"    dataset dir  {dataset_dir}")
    print(f"    cache        {cache_dir}")
    print(f"    rate limit   {'10/s (API key found)' if api_key else '3/s (no NCBI_API_KEY)'}")
    if not email:
        print("    NOTE: set NCBI_EMAIL=<you@example.com>; NCBI asks for a contact address")
    print()

    replicons = load_replicons(replicons_csv)
    print(f"  replicons.csv  {len(replicons):,} records with recorded sha256")
    frags = load_fragments(dataset_dir, splits)

    needed: dict[str, int] = {}
    for split in splits:
        for r in frags[split]:
            needed[r["sequence_id"]] = needed.get(r["sequence_id"], 0) + 1
    missing_meta = sorted(a for a in needed if a not in replicons
                          or not replicons[a].get("sequence_sha256"))
    if missing_meta:
        print(f"FATAL: {len(missing_meta)} accessions have no sha256 in replicons.csv, "
              "so a download could not be verified:", file=sys.stderr)
        for a in missing_meta[:10]:
            print(f"       {a}", file=sys.stderr)
        return 1

    total_bases = sum(int(replicons[a]["length"]) for a in needed)
    have = sum(1 for a in needed if (cache_dir / f"{a}.fasta.gz").is_file())
    print(f"\n  accessions to fetch  {len(needed):,}  ({have:,} already cached)")
    print(f"  replicon bases       {total_bases / 1e9:.2f} Gb  (~{total_bases / 1e9:.1f} GB of text)")
    print(f"  fragments to write   {sum(len(frags[s]) for s in splits):,}")
    if args.dry_run:
        print("\n  --dry-run: stopping before any download")
        return 0

    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    asm_dir = cache_dir / "assemblies"
    asm_dir.mkdir(parents=True, exist_ok=True)

    # ---- pass 1: whole assemblies over HTTP, which is where the bandwidth is won
    if args.source == "ftp":
        by_asm: dict[tuple[str, str], dict[str, dict]] = {}
        for acc in sorted(needed):
            if (cache_dir / f"{acc}.fasta.gz").is_file():
                continue                      # verified on an earlier run
            meta = replicons[acc]
            key = (meta["assembly_accession"], meta["source_fasta"])
            by_asm.setdefault(key, {})[acc] = meta

        recorded = load_recorded_urls(root / "dataset pipeline1" / "data" / "metadata"
                                      / "download_status.csv")
        if recorded:
            print(f"  download_status.csv  {len(recorded):,} recorded assembly URLs "
                  "(used instead of deriving the FTP path)")

        todo = sorted(by_asm)
        if args.smoke:
            todo = todo[: args.smoke]
        print(f"\n--- fetching {len(todo):,} assemblies from ftp.ncbi.nlm.nih.gov"
              f"{' (smoke test)' if args.smoke else ''}")

        got: set[str] = set()
        unresolved: list[str] = []
        bytes_in = 0
        t_ftp = time.time()
        for n, key in enumerate(todo, 1):
            asm_acc, src = key
            try:
                done, nbytes = fetch_assembly(asm_acc, src, by_asm[key], cache_dir,
                                              asm_dir, args.keep_assembly_cache,
                                              recorded.get(asm_acc))
            except FileNotFoundError:
                print(f"    {asm_acc}: not on the FTP site any more -> efetch fallback")
                unresolved.extend(by_asm[key])
                continue
            except RuntimeError as exc:
                print(f"    {asm_acc}: {exc} -> efetch fallback")
                unresolved.extend(by_asm[key])
                continue
            got |= done
            bytes_in += nbytes
            short = set(by_asm[key]) - done
            if short:
                print(f"    {asm_acc}: {len(short)} record(s) not verified from the "
                      "assembly -> efetch fallback")
                unresolved.extend(sorted(short))
            if n % 25 == 0 or n == len(todo):
                el = max(time.time() - t_ftp, 1e-9)
                print(f"    {n:,}/{len(todo):,} assemblies, {len(got):,} records verified, "
                      f"{bytes_in / 1e6:.0f} MB, {bytes_in / el / 1e6:.1f} MB/s")

        print(f"    verified {len(got):,} records from the assembly files"
              f"{f', {len(unresolved)} to fall back on efetch' if unresolved else ''}")
        if args.smoke:
            print("\n  --smoke: every record above matched the sha256 recorded in "
                  "replicons.csv.\n  Nothing was written. Drop --smoke for the real rebuild.")
            return 1 if (unresolved and len(unresolved) == len(todo)) else 0

    downloaded = 0
    checked = 0
    t0 = time.time()
    for split in splits:
        out = dataset_dir / f"{split}.fasta"
        tmp = dataset_dir / f"{split}.fasta.part"
        print(f"\n--- {split}: {len(frags[split]):,} fragments -> {out.name}")
        current_acc, current_seq = None, ""
        bad: list[str] = []
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            for i, row in enumerate(frags[split], 1):
                acc = row["sequence_id"]
                if acc != current_acc:
                    meta = replicons[acc]
                    current_seq, fresh = cached_sequence(
                        acc, cache_dir, meta["sequence_sha256"], int(meta["length"]),
                        api_key, email, pause)
                    current_acc = acc
                    downloaded += int(fresh)
                    if downloaded and downloaded % 25 == 0 and fresh:
                        rate = downloaded / max(time.time() - t0, 1e-9) * 60
                        print(f"    {downloaded:,} downloaded, {i:,}/{len(frags[split]):,} "
                              f"fragments written ({rate:.0f} replicons/min)")

                start, end = int(row["start"]), int(row["end"])
                piece = current_seq[start:end].upper()

                # every fragment is checked against the three independent numbers
                # the CSV recorded for it when it was first cut
                gc, bad_frac = base_stats(piece)
                if (len(piece) != int(row["length"])
                        or f"{gc:.6f}" != f"{float(row['gc_content']):.6f}"
                        or f"{bad_frac:.6f}" != f"{float(row['non_acgt_fraction']):.6f}"):
                    bad.append(
                        f"{row['fragment_id']} ({acc}:{start}-{end}): "
                        f"len {len(piece)} vs {row['length']}, "
                        f"gc {gc} vs {row['gc_content']}, "
                        f"non_acgt {bad_frac} vs {row['non_acgt_fraction']}")
                    if len(bad) > 20:
                        break
                    continue

                write_fasta_record(fh, row["fragment_id"], piece)
                checked += 1

        if bad:
            tmp.unlink(missing_ok=True)
            print(f"\nFATAL: {len(bad)} fragment(s) in {split} do not match the CSV:",
                  file=sys.stderr)
            for line in bad[:20]:
                print(f"       {line}", file=sys.stderr)
            print("       Nothing was written for this split. This means the slice no\n"
                  "       longer reproduces the published fragment, so the rebuild cannot\n"
                  "       stand in for the laptop's files.", file=sys.stderr)
            return 1

        tmp.replace(out)
        size = out.stat().st_size
        print(f"    wrote {out.name}  {size / 1e6:.0f} MB  ({len(frags[split]):,} records)")

    print(f"\n=== rebuilt {checked:,} fragments; {downloaded:,} replicons downloaded "
          f"in {(time.time() - t0) / 60:.1f} min")
    print("    every replicon matched its sha256 in replicons.csv, and every fragment")
    print("    matched the length, GC and non-ACGT fraction recorded in the split CSV.")

    # 02_verify_data.sh expects a manifest. Build it here, and say plainly what it
    # does and does not prove on this path.
    manifest = root / "server" / "dataset_md5.txt"
    items = [f"dataset pipeline1/data/dataset_mixed/{s}.{ext}"
             for s in SPLITS for ext in ("fasta", "csv")]
    items += ["dataset pipeline1/data/metadata/replicons.csv",
              "dataset pipeline1/data/metadata/fragments_mixed.csv"]
    lines = []
    for rel in items:
        p = root / rel
        if not p.is_file():
            continue
        h = hashlib.md5()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {rel}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    origin = root / "server" / "dataset_md5.origin.txt"
    origin.write_text(
        "server/dataset_md5.txt was built by server/01b_rebuild_fasta_from_ncbi.py,\n"
        "not by 01_sync_from_laptop.sh. The CSVs come from git, so their checksums\n"
        "are the laptop's. The FASTAs were rebuilt here, so their checksums only\n"
        "prove the files have not changed since the rebuild - they are not evidence\n"
        "of a faithful transfer.\n"
        "\n"
        "What IS evidence, and is stronger: every replicon was verified against the\n"
        "sequence_sha256 recorded in metadata/replicons.csv when the dataset was\n"
        "built, and every fragment against the length, gc_content and\n"
        "non_acgt_fraction in its split CSV. Step 2 of 02_verify_data.sh (the\n"
        "pipeline's own re-derivation) and step 3 (the counts) are unaffected.\n",
        encoding="utf-8")
    print(f"    wrote server/dataset_md5.txt ({len(lines)} files) and "
          "server/dataset_md5.origin.txt")

    if not args.keep_cache:
        on_disk = sum(p.stat().st_size for p in cache_dir.rglob("*") if p.is_file())
        print(f"\n    the {on_disk / 1e9:.1f} GB download cache can be removed once "
              "02_verify_data.sh passes:")
        print(f"      rm -rf {cache_dir}")
    print("\nNext: sbatch server/02_verify_data.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
