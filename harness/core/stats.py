"""Small statistics helpers. Eval sets are small (tens of cases), so a raw
pass rate like 20/25 = 80% carries real uncertainty -- these exist so the
reports can say how much, instead of implying false precision."""
from __future__ import annotations

import math


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion. Unlike the
    naive normal approximation it behaves sensibly at 0% / 100% and for
    small n, which is exactly the regime a golden set lives in."""
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar p-value for two models run on the SAME cases.

    Only the cases where they disagree carry information: if the models were
    equally good, each disagreement would be a coin flip. The result is the
    chance of a split at least this lopsided from coin flips alone. This is the
    right test here; comparing two confidence intervals ignores that the cases
    are shared and is far too cautious.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile, pct in [0, 100]. 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100) * (len(ordered) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation, or None when it is undefined (fewer than two
    points, or either series has zero variance)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return sxy / math.sqrt(sxx * syy)
