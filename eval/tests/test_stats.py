"""eval/tests/test_stats.py — 纯标准库统计模块 stats.py 的 TDD 测试。

运行: cd eval && python -m pytest tests/test_stats.py -v
约束: 只用标准库 + pytest,不依赖 scipy/numpy。
"""
import random
import sys
from math import comb
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


# ══════════════════════════════════════════════════════════════
# 任务2:pass@k 无偏估计器 + pass^k
# ══════════════════════════════════════════════════════════════
class TestPassAtK:
    """pass@k:HumanEval 式无偏估计;pass^k:n 次采样全过的概率。"""

    def test_boundary_all_or_none(self):
        assert stats.pass_at_k(20, 20, 5) == 1.0   # 全过 → 必过
        assert stats.pass_at_k(20, 0, 5) == 0.0    # 全挂 → 必不过

    def test_point_estimate(self):
        # c=15, n=20, k=1 → 15/20
        assert stats.pass_at_k(20, 15, 1) == pytest.approx(0.75)

    def test_n_minus_c_less_than_k(self):
        # 失败数 n-c < k → 抽 k 个必然含一个通过的 → 1.0
        assert stats.pass_at_k(20, 15, 25) == 1.0
        assert stats.pass_at_k(20, 15, 21) == 1.0

    def test_matches_bruteforce_combination(self):
        # 与暴力组合数公式 1 - C(n-c,k)/C(n,k) 一致(n=10,c=7,k=3)
        bruteforce = 1 - comb(3, 3) / comb(10, 3)
        assert stats.pass_at_k(10, 7, 3) == pytest.approx(bruteforce, rel=1e-12)

    def test_unbiasedness_simulation(self):
        # 无偏性:真 p=0.7, n=20, k=5;估计均值应逼近解析真值
        n, k, p = 20, 5, 0.7
        true_val = sum(comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1))
        rng = random.Random(7)
        ests = []
        for _ in range(2500):
            c = sum(1 for _ in range(n) if rng.random() < p)
            ests.append(stats.pass_at_k(n, c, k))
        assert abs(sum(ests) / len(ests) - true_val) < 0.01

    def test_pass_pow_k(self):
        # pass^k = (c/n)^k:每次采样结果独立的近似
        assert stats.pass_pow_k(20, 15, 4) == pytest.approx(0.75**4)

    def test_invalid_args(self):
        # k>n 不算非法:k>n 时 n-c<k 必然成立,返回 1.0(见上)
        for bad in [(0, 0, 1), (10, -1, 1), (10, 11, 1), (10, 5, 0)]:
            with pytest.raises(ValueError):
                stats.pass_at_k(*bad)
        for bad in [(0, 0, 1), (10, -1, 1), (10, 11, 1), (10, 5, 0)]:
            with pytest.raises(ValueError):
                stats.pass_pow_k(*bad)


# ══════════════════════════════════════════════════════════════
# 任务3:McNemar 精确检验
# ══════════════════════════════════════════════════════════════
class TestMcNemarExact:
    """McNemar 精确二项检验:配对模型 A/B 的显著性(只看不一致格 b/c)。"""

    def test_no_discordant_pairs(self):
        # b=c=0 → 无任何不一致,无证据拒绝
        assert stats.mcnemar_exact_p(0, 0) == 1.0

    def test_symmetric_discordant(self):
        # b=c=5 → 完全对称,p 应接近 1
        assert stats.mcnemar_exact_p(5, 5) > 0.9

    def test_asymmetric_reference_value(self):
        # b=1,c=10 → 与 scipy binomtest(1,11,0.5) 双侧一致
        assert stats.mcnemar_exact_p(1, 10) == pytest.approx(0.01171875, abs=1e-6)

    def test_extreme_asymmetry(self):
        # b=0,c=12 → 极端不一致,强显著
        assert stats.mcnemar_exact_p(0, 12) < 0.01

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            stats.mcnemar_exact_p(-1, 5)
        with pytest.raises(ValueError):
            stats.mcnemar_exact_p(5, -1)


# ══════════════════════════════════════════════════════════════
# 任务4:Cohen's κ + judge 诊断指标
# ══════════════════════════════════════════════════════════════
class TestCohensKappa:
    """κ:超出随机一致性的部分,校正 accuracy 在不平衡数据上的虚高。"""

    def test_perfect_agreement(self):
        assert stats.cohens_kappa(90, 0, 0, 10) == pytest.approx(1.0)

    def test_chance_agreement(self):
        # 对角线均分 → 与随机一致无异,κ=0
        assert stats.cohens_kappa(25, 25, 25, 25) == pytest.approx(0.0, abs=1e-9)

    def test_worse_than_chance(self):
        assert stats.cohens_kappa(0, 50, 50, 0) < 0

    def test_reference_value(self):
        # 手算:po=0.85;行/列边际 (40,60)x(35,65) → pe=5300/10000=0.53
        # κ=(0.85-0.53)/(1-0.53)=0.32/0.47≈0.68085
        # (注:任务书参考值 0.6907 源于 pe 误算为 0.515,1400+3900=5300 非 5150)
        assert stats.cohens_kappa(30, 10, 5, 55) == pytest.approx(0.68085, abs=1e-3)


class TestJudgeMetrics:
    """judge 诊断四格:正类=有缺陷(judge 判 not-ok)。"""

    def test_balanced_case(self):
        m = stats.judge_metrics(45, 5, 15, 35)
        assert m["tpr"] == pytest.approx(45 / 60)
        assert m["tnr"] == pytest.approx(35 / 40)
        assert m["ppv"] == pytest.approx(45 / 50)
        assert m["npv"] == pytest.approx(35 / 50)
        assert m["accuracy"] == pytest.approx(0.8)
        assert m["prevalence"] == pytest.approx(0.6)
        assert m["kappa"] > 0
        assert m["n"] == 100
        assert m["ppv_defined"] is True

    def test_base_rate_fallacy(self):
        # 永远 PASS 的 judge + 缺陷率 10%:accuracy 0.9 看着不错,
        # 但 tpr=0、κ=0 暴露它毫无判别力;ppv 无定义(0/0)
        m = stats.judge_metrics(0, 0, 10, 90)
        assert m["accuracy"] == pytest.approx(0.9)
        assert m["tpr"] == 0.0
        assert m["kappa"] == pytest.approx(0.0, abs=1e-9)
        assert m["ppv"] is None
        assert m["ppv_defined"] is False

    def test_all_zero_returns_empty_dict(self):
        # 无样本 → 无任何可算指标,约定返回空 dict
        assert stats.judge_metrics(0, 0, 0, 0) == {}

    def test_cohens_kappa_rejects_empty(self):
        with pytest.raises(ValueError):
            stats.cohens_kappa(0, 0, 0, 0)


# ══════════════════════════════════════════════════════════════
# 任务5:bootstrap percentile 置信区间
# ══════════════════════════════════════════════════════════════
class TestBootstrapCI:
    """bootstrap:对任意统计量做重采样 percentile 区间,不依赖分布假设。"""

    def _gauss_samples(self, n=200):
        rng = random.Random(3)
        return [rng.gauss(10, 2) for _ in range(n)]

    def test_mean_ci_brackets_truth(self):
        samples = self._gauss_samples()
        lo, hi = stats.bootstrap_ci(samples, stat="mean", B=1000, seed=42)
        point = sum(samples) / len(samples)
        assert lo < point < hi  # 区间包住点估计
        assert lo < 10.5 and hi > 9.5  # 也应接近真值 10
        assert (hi - lo) < (max(samples) - min(samples))  # 宽度 < 样本极差

    def test_outlier_sample_brackets_statistic(self):
        # 均值 19.166…(被 100 拉高),区间应包含该点估计
        lo, hi = stats.bootstrap_ci([1, 2, 3, 4, 5, 100], stat="mean", B=500, seed=1)
        assert lo <= 115 / 6 <= hi

    def test_callable_stat(self):
        samples = self._gauss_samples(50)
        point = sum(samples) / len(samples)
        lo, hi = stats.bootstrap_ci(samples, stat=lambda x: sum(x) / len(x),
                                    B=200, seed=9)
        assert lo < point < hi

    def test_reproducible_same_seed(self):
        samples = self._gauss_samples(50)
        r1 = stats.bootstrap_ci(samples, stat="median", B=200, seed=7)
        r2 = stats.bootstrap_ci(samples, stat="median", B=200, seed=7)
        assert r1 == r2

    def test_median_stat(self):
        samples = self._gauss_samples(100)
        point = sorted(samples)[len(samples) // 2]
        lo, hi = stats.bootstrap_ci(samples, stat="median", B=500, seed=2)
        assert lo < point < hi

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            stats.bootstrap_ci([], stat="mean", B=100)          # 空 samples
        with pytest.raises(ValueError):
            stats.bootstrap_ci([1, 2, 3], stat="mean", B=9)     # B < 10
        with pytest.raises(ValueError):
            stats.bootstrap_ci([1, 2, 3], stat="variance", B=100)  # 未知 stat 名
