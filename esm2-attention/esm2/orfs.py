"""Six-frame ORF extraction, transcribed from the prototype in `old resource/`."""

from __future__ import annotations

from Bio.Seq import Seq


def extract_orfs(dna_seq: str, min_length: int = 100, max_orfs: int = 10) -> list[str]:
    """Translate all six frames, cut at stop codons, keep the longest peptides.

    These are inter-stop peptides, not ORFs in the start-codon sense - that is what
    the prototype does and it is kept unchanged so the architecture is comparable.
    Ties keep frame order because Python's sort is stable.
    """
    seq_obj = Seq(dna_seq)
    peptides = []
    for seq in (seq_obj, seq_obj.reverse_complement()):
        for frame in range(3):
            usable = len(seq) - frame
            in_frame = seq[frame:frame + (usable - usable % 3)]
            peptides.extend(p for p in str(in_frame.translate()).split("*")
                            if len(p) >= min_length)
    peptides.sort(key=len, reverse=True)
    return peptides[:max_orfs]
