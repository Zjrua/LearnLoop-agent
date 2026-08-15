"""eval/tests/test_run_eval_repeats.py — --repeats 多次重复聚合的 TDD 测试(不调 LLM)。

运行: cd eval && python -m pytest tests/test_run_eval_repeats.py -v
"""
import sys
from pathlib import Path

import pytest

# 让测试找到 eval/run_eval.py 与 eval/stats.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_eval  # noqa: E402


class TestAggregateRepeats:
    def test_mixed_runs(self):
        agg = run_eval.aggregate_repeats(
            [{"complete_ok": True}, {"complete_ok": False}, {"complete_ok": True}]
        )
        assert agg["k"] == 3
        assert agg["successes"] == 2
        # k 次中至少 1 次成功 → pass@k = 1.0;全对频率 → 2/3 中 0 次全对 = 0.0
        assert agg["pass_at_k"] == 1.0
        assert agg["pass_pow_k"] == 0.0
        assert agg["pass_at_1"] == pytest.approx(2 / 3)
        lo, hi = agg["ci95"]
        assert isinstance(lo, float) and isinstance(hi, float)
        assert lo <= 2 / 3 <= hi

    def test_single_run_ok(self):
        agg = run_eval.aggregate_repeats([{"complete_ok": True}])
        assert agg["k"] == 1
        assert agg["pass_at_k"] == 1.0
        assert agg["pass_pow_k"] == 1.0

    def test_single_run_fail(self):
        agg = run_eval.aggregate_repeats([{"complete_ok": False}])
        assert agg["pass_at_k"] == 0.0
        assert agg["pass_pow_k"] == 0.0

    def test_all_fail(self):
        agg = run_eval.aggregate_repeats([{"complete_ok": False}] * 5)
        assert agg["pass_at_k"] == 0.0
        assert agg["pass_pow_k"] == 0.0
        assert agg["successes"] == 0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            run_eval.aggregate_repeats([])


class TestParseArgs:
    def test_defaults_backward_compatible(self):
        args = run_eval.parse_args([])
        assert args.repeats == 1
        assert args.only is None
        assert args.out_prefix == "eval"

    def test_custom_values(self):
        args = run_eval.parse_args(
            ["--repeats", "5", "--only", "你好", "--out-prefix", "smoke"]
        )
        assert args.repeats == 5
        assert args.only == "你好"
        assert args.out_prefix == "smoke"
