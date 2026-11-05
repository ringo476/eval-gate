import math

from harness.core.stats import intervals_overlap, pearson, percentile, wilson_interval


def test_wilson_interval_known_value():
    lo, hi = wilson_interval(8, 10)
    # standard textbook value for 8/10 at 95%
    assert math.isclose(lo, 0.490, abs_tol=0.005)
    assert math.isclose(hi, 0.943, abs_tol=0.005)


def test_wilson_interval_is_sane_at_the_extremes():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.4
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and 0.6 < lo < 1.0


def test_wilson_interval_empty_sample():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_narrows_with_more_data():
    small = wilson_interval(8, 10)
    large = wilson_interval(80, 100)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_intervals_overlap():
    assert intervals_overlap((0.1, 0.5), (0.4, 0.9))
    assert not intervals_overlap((0.1, 0.3), (0.4, 0.9))


def test_percentile():
    assert percentile([], 50) == 0.0
    assert percentile([7.0], 95) == 7.0
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert math.isclose(percentile([0, 10], 25), 2.5)


def test_pearson():
    assert math.isclose(pearson([1, 2, 3], [2, 4, 6]), 1.0)
    assert math.isclose(pearson([1, 2, 3], [6, 4, 2]), -1.0)
    assert pearson([1, 1, 1], [1, 2, 3]) is None  # zero variance: undefined, not a crash
    assert pearson([1], [1]) is None
