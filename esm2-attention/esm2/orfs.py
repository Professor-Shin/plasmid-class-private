"""Six-frame ORF extraction, transcribed from the prototype in `old resource/`."""

from __future__ import annotations

from Bio.Seq import Seq


def extract_orfs(dna_seq: str, min_length: int = 100, max_orfs: int | None = 10,
                 hard_cap: int | None = None) -> list[str]:
    """Translate all six frames, cut at stop codons, keep the longest peptides.

    These are inter-stop peptides, not ORFs in the start-codon sense - that is what
    the prototype does and it is kept unchanged so the architecture is comparable.
    Ties keep frame order because Python's sort is stable.

    ``max_orfs=None`` keeps every peptide above ``min_length``. That is the point
    of the server run: with the laptop's cap of 10, 80.5% of test fragments lost
    ORFs before the model ever saw them, and a long fragment lost most of its
    coding content. ``hard_cap`` is a padding guard, not a modelling choice - a
    100 kb fragment can yield a hundred peptides and the attention pooling pads to
    the longest member of its batch.
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
    limit = max_orfs if max_orfs else hard_cap
    return peptides[:limit] if limit else peptides
