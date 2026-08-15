"""eval/stats.py — eval 统计基础设施(纯标准库)。

对应《AI Agent 评测指南》第九章方法论:
Wilson / Clopper-Pearson 区间、pass@k / pass^k、McNemar 精确检验、
Cohen's κ 与 judge 诊断指标、bootstrap percentile 置信区间。

约束:只用 math/random/statistics 标准库,不依赖 scipy/numpy,
便于在任何 Python 环境下直接复用。
"""
from __future__ import annotations

import math


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
