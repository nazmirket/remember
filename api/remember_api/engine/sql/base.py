"""Shared SQL scoring helper."""


def bm25_score_gate(bm25_min_score: float) -> str:
    """Return the comparison a BM25 arm applies to its relevance score.

    Two different jobs share one number. With no caller floor (the default,
    ``0.0``) the gate is structural — ``> 0`` keeps only genuine term matches on
    backends whose operator ranks every document instead of pre-filtering. With a
    caller floor (recall's ``min_scores.keyword``) the gate becomes that floor,
    **inclusive**, matching the documented contract and the semantic arm's
    ``>= min_similarity``. A positive floor subsumes the structural gate, so the
    two never need to be applied together.
    """
    if bm25_min_score <= 0:
        return "> 0"
    # `!r` (shortest round-tripping repr), not `:g` — `:g` truncates to 6
    # significant digits, so a caller echoing a `scores.keyword` value back as a
    # floor could get a literal that rounds up past its own row and drops it.
    return f">= {bm25_min_score!r}"
