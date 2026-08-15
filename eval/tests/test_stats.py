"""eval/tests/test_stats.py — 纯标准库统计模块 stats.py 的 TDD 测试。

运行: cd eval && python -m pytest tests/test_stats.py -v
约束: 只用标准库 + pytest,不依赖 scipy/numpy。
"""
import random
import sys
from pathlib import Path

import pytest

# 让测试在任意 CWD 下都能找到 eval/stats.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import stats  # noqa: E402


# ══════════════════════════════════════════════════════════════
# 任务1:Wilson score 置信区间
# ══════════════════════════════════════════════════════════════
class TestWilsonCI:
    """Wilson 区间:小样本成功率的合理置信区间。"""

    def test_point_estimate_11_of_13(self):
        lo, hi = stats.wilson_ci(11, 13)
        assert lo == pytest.approx(0.578, abs=1e-2)
        assert hi == pytest.approx(0.956, abs=1e-2)

    def test_zero_success(self):
        lo, hi = stats.wilson_ci(0, 13)
        assert lo == 0.0
        assert 0.0 < hi < 0.30

    def test_all_success(self):
        lo, hi = stats.wilson_ci(13, 13)
        assert lo > 0.70
        assert hi == 1.0

    def test_width_shrinks_with_n(self):
        # 同比例(k/n 不变)下样本量增大 → 区间收窄
        w_small = stats.wilson_ci(11, 13)
        w_large = stats.wilson_ci(110, 130)
        assert (w_small[1] - w_small[0]) > (w_large[1] - w_large[0])

    def test_coverage_simulation(self):
        # 蒙特卡洛:p=0.8, n=13, 2000 次抽样,经验覆盖率应接近名义 95%
        rng = random.Random(42)
        n, p, trials = 13, 0.8, 2000
        covered = 0
        for _ in range(trials):
            k = sum(1 for _ in range(n) if rng.random() < p)
            lo, hi = stats.wilson_ci(k, n)
            if lo <= p <= hi:
                covered += 1
        coverage = covered / trials
        assert 0.92 < coverage < 0.98, f"经验覆盖率 {coverage:.4f} 超出 (0.92, 0.98)"

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            stats.wilson_ci(0, 0)      # n<=0
        with pytest.raises(ValueError):
            stats.wilson_ci(-1, 13)    # k<0
        with pytest.raises(ValueError):
            stats.wilson_ci(14, 13)    # k>n
