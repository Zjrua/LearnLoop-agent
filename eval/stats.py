"""eval/stats.py — eval 统计基础设施(纯标准库)。

对应《AI Agent 评测指南》第九章方法论:
Wilson / Clopper-Pearson 区间、pass@k / pass^k、McNemar 精确检验、
Cohen's κ 与 judge 诊断指标、bootstrap percentile 置信区间。

约束:只用 math/random/statistics 标准库,不依赖 scipy/numpy,
便于在任何 Python 环境下直接复用。
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable
from math import comb


# ── Wilson score 置信区间 ──
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson score 置信区间。

    相比 Wald 区间(正态近似),Wilson 在小样本/比例接近 0 或 1 时
    仍能保持名义覆盖率,不会越界到 [0,1] 之外。

    Args:
        k: 成功次数
        n: 总试验次数
        z: 标准正态分位数,默认 1.96(95% 置信水平)

    Returns:
        (lo, hi) 置信区间端点,已 clamp 到 [0, 1]
    """
    if n <= 0:
        raise ValueError(f"n 必须 > 0,收到 n={n}")
    if k < 0:
        raise ValueError(f"k 必须 >= 0,收到 k={k}")
    if k > n:
        raise ValueError(f"k 不能大于 n,收到 k={k}, n={n}")

    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


# ── pass@k 无偏估计器 / pass^k ──
def _validate_counts(c: int, n: int, k: int) -> None:
    """pass@k / pass^k 共用的参数校验。"""
    if n <= 0:
        raise ValueError(f"n 必须 > 0,收到 n={n}")
    if c < 0:
        raise ValueError(f"c 必须 >= 0,收到 c={c}")
    if c > n:
        raise ValueError(f"c 不能大于 n,收到 c={c}, n={n}")
    if k <= 0:
        raise ValueError(f"k 必须 > 0,收到 k={k}")


def pass_at_k(n: int, c: int, k: int) -> float:
    """无偏 pass@k 估计器(HumanEval 公式)。

    c/n 次采样中通过 c 次,估计"抽 k 个至少一个通过"的概率:
        pass@k = 1 - C(n-c, k) / C(n, k)
    用数值稳定连乘实现,避免大组合数溢出:
        prod_{i=0}^{k-1} (n-c-i) / (n-i)
    """
    _validate_counts(c, n, k)
    if n - c < k:
        # 失败样本不足 k 个 → 抽 k 个必然含通过样本
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod


def pass_pow_k(n: int, c: int, k: int) -> float:
    """pass^k:k 次独立采样全部通过的概率估计 = (c/n)^k。

    与 pass@k(至少一个通过)互补,衡量"稳定全过"的难度。
    """
    _validate_counts(c, n, k)
    return (c / n) ** k


# ── McNemar 精确检验 ──
def mcnemar_exact_p(b: int, c: int) -> float:
    """McNemar 精确二项检验的双侧 p 值。

    配对场景(同一批用例跑模型 A/B),2x2 列联表中只有不一致格
    b(A 过 B 挂)与 c(A 挂 B 过)携带 A/B 差异信息。
    零假设下 b ~ Binomial(b+c, 0.5),双侧精确 p:
        p = min(1, 2 * P(X <= min(b, c))),  X ~ Binomial(b+c, 0.5)
    """
    if b < 0:
        raise ValueError(f"b 必须 >= 0,收到 b={b}")
    if c < 0:
        raise ValueError(f"c 必须 >= 0,收到 c={c}")

    n = b + c
    if n == 0:
        # 没有不一致样本 → 无证据拒绝
        return 1.0
    tail = sum(comb(n, i) * 0.5**n for i in range(min(b, c) + 1))
    return min(1.0, 2.0 * tail)


# ── Cohen's κ + judge 诊断指标 ──
def cohens_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    """Cohen's κ:两评分者(judge vs 真值)一致性超出随机的部分。

    κ = (po - pe) / (1 - pe)
    po = 观察一致率 = (tp+tn)/n
    pe = 随机期望一致率 = (行1边际×列1边际 + 行2边际×列2边际)/n²
    κ=1 完美一致;κ=0 不优于随机;κ<0 比随机还差。
    """
    n = tp + fp + fn + tn
    if n == 0:
        raise ValueError("四格全为 0,无法计算 κ")
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fp + tn) * (fn + tn)) / (n * n)
    if pe >= 1.0:
        # 边际完全一致且 po=1 → 完美一致
        return 1.0
    return (po - pe) / (1.0 - pe)


def judge_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    """LLM judge 的诊断四格指标(正类 = 有缺陷,judge 判 not-ok)。

    tp: 缺陷被抓住  fp: 误报   fn: 漏检   tn: 正常放行

    返回 dict:tpr/tnr/ppv/npv/accuracy/prevalence/kappa/ppv_defined/n。
    除零时对应值为 None(ppv_defined=False);无样本时返回空 dict。
    """
    n = tp + fp + fn + tn
    if n == 0:
        return {}
    return {
        "tpr": tp / (tp + fn) if (tp + fn) > 0 else None,
        "tnr": tn / (tn + fp) if (tn + fp) > 0 else None,
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else None,
        "ppv_defined": (tp + fp) > 0,
        "npv": tn / (tn + fn) if (tn + fn) > 0 else None,
        "accuracy": (tp + tn) / n,
        "prevalence": (tp + fn) / n,
        "kappa": cohens_kappa(tp, fp, fn, tn),
        "n": n,
    }


# ── bootstrap percentile 置信区间 ──
def bootstrap_ci(
    samples: list[float],
    stat: str | Callable[[list[float]], float] = "mean",
    B: int = 1000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> tuple[float, float]:
    """bootstrap percentile 法置信区间。

    对 samples 有放回重采样 B 次,每次计算统计量 stat,取经验分布的
    alpha/2 与 1-alpha/2 分位作为区间端点。不依赖任何分布假设,
    适用于延迟、token 数等形状任意的指标。

    Args:
        samples: 原始样本(非空)
        stat: "mean" / "median" / 任意 list->float 可调用对象
        B: 重采样次数(>= 10)
        alpha: 显著性水平,默认 0.05 → 95% 区间
        seed: 随机种子,None 表示不固定

    Returns:
        (lo, hi) percentile 置信区间
    """
    if not samples:
        raise ValueError("samples 不能为空")
    if B < 10:
        raise ValueError(f"B 必须 >= 10,收到 B={B}")

    if callable(stat):
        stat_fn = stat
    elif stat == "mean":
        stat_fn = lambda x: sum(x) / len(x)  # noqa: E731
    elif stat == "median":
        stat_fn = lambda x: sorted(x)[len(x) // 2]  # noqa: E731
    else:
        raise ValueError(f"未知 stat:{stat!r}(支持 'mean'/'median'/callable)")

    rng = random.Random(seed)
    n = len(samples)
    vals = sorted(stat_fn(rng.choices(samples, k=n)) for _ in range(B))
    lo_idx = int((alpha / 2) * B)
    hi_idx = min(B - 1, int((1 - alpha / 2) * B))
    return vals[lo_idx], vals[hi_idx]


# ── Clopper-Pearson 精确二项置信区间 ──
def _binom_cdf(k: int, n: int, p: float) -> float:
    """二项分布 CDF:P(X <= k),X ~ Binomial(n, p)。k<0 时约定为 0。"""
    if k < 0:
        return 0.0
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def _solve(
    f: Callable[[float], float],
    target: float,
    lo: float = 0.0,
    hi: float = 1.0,
    iters: int = 100,
) -> float:
    """二分法解 f(x) = target(f 单调,不要求方向),返回最终区间中点。"""
    f_lo = f(lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        f_mid = f(mid)
        if f_mid == target:
            return mid
        if (f_mid - target) * (f_lo - target) > 0:
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson 精确二项置信区间。

    通过求解二项分布尾概率方程构造:
        lo: P(X >= k | p=lo) = alpha/2   (即 1 - CDF(k-1) = alpha/2)
        hi: P(X <= k | p=hi) = alpha/2
    等价于 Beta 分布分位数(BetaInverse),这里用 bisection 数值求解。

    与 Wilson 的区别:CP 是精确法,覆盖概率保证 >= 1-alpha(保守,
    实际覆盖常略超名义水平,区间偏宽);Wilson 基于正态近似 score,
    区间更窄但覆盖概率只在平均意义下接近名义水平。报告需要"保险"
    的下界(如评测通过率声明)时用 CP,日常比较用 Wilson。

    Args:
        k: 成功次数
        n: 总试验次数
        alpha: 显著性水平,须在 (0, 1),默认 0.05 → 95% 区间

    Returns:
        (lo, hi) 置信区间端点,始终落在 [0, 1] 内
    """
    if n <= 0:
        raise ValueError(f"n 必须 > 0,收到 n={n}")
    if k < 0:
        raise ValueError(f"k 必须 >= 0,收到 k={k}")
    if k > n:
        raise ValueError(f"k 不能大于 n,收到 k={k}, n={n}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha 必须在 (0, 1),收到 alpha={alpha}")

    tail = alpha / 2
    if k == 0:
        return 0.0, _solve(lambda p: _binom_cdf(0, n, p), tail)
    if k == n:
        return _solve(lambda p: 1.0 - _binom_cdf(n - 1, n, p), tail), 1.0
    lo = _solve(lambda p: 1.0 - _binom_cdf(k - 1, n, p), tail)
    hi = _solve(lambda p: _binom_cdf(k, n, p), tail)
    return lo, hi
