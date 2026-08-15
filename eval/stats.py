"""eval/stats.py — eval 统计基础设施(纯标准库)。

对应《AI Agent 评测指南》第九章方法论:
Wilson / Clopper-Pearson 区间、pass@k / pass^k、McNemar 精确检验、
Cohen's κ 与 judge 诊断指标、bootstrap percentile 置信区间。

约束:只用 math/random/statistics 标准库,不依赖 scipy/numpy,
便于在任何 Python 环境下直接复用。
"""
from __future__ import annotations

import math
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
