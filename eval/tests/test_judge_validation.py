"""eval/tests/test_judge_validation.py — run_judge_validation.py 的 TDD 测试(不调 LLM)。

运行: cd eval && python -m pytest tests/test_judge_validation.py -q

覆盖:
  1. build_validation_set(): 20 条、字段齐全、ok/not_ok 各 10、无重复 (q, draft[:40])
  2. majority_ok(): 3 票多数
  3. tally(): 四格各占一类 → 四计数各 1
  4. parse_verdict(): judge 回复解析(ok/not-ok 判定)
  5. --dry-run subprocess: exit 0 且输出含三组混淆矩阵
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# 让测试在任意 CWD 下都能找到 eval/run_judge_validation.py 与 eval/stats.py
EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))

REPO = EVAL_DIR.parent
BACKEND_DIR = REPO / "skill-tree" / "backend"
DATA_DIR = REPO / "skill-tree" / "data"

import run_judge_validation as rjv  # noqa: E402


# ══════════════════════════════════════════════════════════════
# 1. build_validation_set
# ══════════════════════════════════════════════════════════════
class TestBuildValidationSet:
    def test_len_and_fields(self):
        s = rjv.build_validation_set()
        assert len(s) == 20
        for item in s:
            # 每样本必含 q/observations/draft/label
            assert {"q", "observations", "draft", "label"} <= set(item.keys())
            assert item["label"] in ("ok", "not_ok")
            # S- 样本必带缺陷类型
            if item["label"] == "not_ok":
                assert "defect_type" in item and item["defect_type"]

    def test_balance(self):
        s = rjv.build_validation_set()
        ok = [x for x in s if x["label"] == "ok"]
        nok = [x for x in s if x["label"] == "not_ok"]
        assert len(ok) == 10
        assert len(nok) == 10

    def test_unique(self):
        s = rjv.build_validation_set()
        keys = [(x["q"], x["draft"][:40]) for x in s]
        assert len(keys) == len(set(keys)), "存在重复 (q, draft[:40]) 样本"


# ══════════════════════════════════════════════════════════════
# 2. majority_ok
# ══════════════════════════════════════════════════════════════
class TestMajorityOk:
    def test_majority_true(self):
        assert rjv.majority_ok([True, True, False]) is True

    def test_majority_false(self):
        assert rjv.majority_ok([False, False, True]) is False


# ══════════════════════════════════════════════════════════════
# 3. tally
# ══════════════════════════════════════════════════════════════
class TestTally:
    def test_one_each(self):
        # 4 样本:tp / fp / fn / tn 各占一类
        labels = ["not_ok", "ok", "not_ok", "ok"]
        preds = [True, True, False, False]  # True = 判 not_ok(有缺陷)
        t = rjv.tally(labels, preds)
        assert t == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


# ══════════════════════════════════════════════════════════════
# 4. parse_verdict
# ══════════════════════════════════════════════════════════════
class TestParseVerdict:
    def test_plain_json_ok(self):
        assert rjv.parse_verdict('{"ok": true, "gap": ""}') is True

    def test_plain_json_not_ok(self):
        assert rjv.parse_verdict('{"ok": false, "gap": "漏了 DeepFM"}') is False

    def test_code_fence_not_ok(self):
        assert rjv.parse_verdict('```json\n{"ok": false, "gap": "编造数字"}\n```') is False

    def test_prose_with_json(self):
        assert rjv.parse_verdict('校验结果：{"ok": true}') is True

    def test_unparseable_falls_back_ok(self):
        # 对齐生产 loop._reflect 语义: 解析失败回退 ok=True(不阻塞)
        assert rjv.parse_verdict("完全无法解析的乱码") is True


# ══════════════════════════════════════════════════════════════
# 5. --dry-run subprocess(全管道,不调 LLM)
# ══════════════════════════════════════════════════════════════
class TestDryRun:
    def test_dry_run_exit0_and_three_matrices(self):
        env = dict(os.environ)
        env["DATA_ROOT"] = str(DATA_DIR)
        proc = subprocess.run(
            [sys.executable, str(EVAL_DIR / "run_judge_validation.py"), "--dry-run"],
            cwd=str(BACKEND_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
        out = proc.stdout + proc.stderr
        assert "混淆矩阵" in out
        # 三组 FakeJudge 桩的组名都要出现
        assert "恒ok" in out
        assert "恒not_ok" in out
        assert "随机" in out
        # 永远-PASS 退化基线对照行
        assert "退化基线" in out
