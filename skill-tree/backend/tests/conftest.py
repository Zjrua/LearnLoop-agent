import os, tempfile
# 测试用临时 DATA_ROOT,避免污染本机 ~/.skill-tree(必须在 import main 前 set)
os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="st_test_"))
os.environ.setdefault("SEED_DIR", tempfile.mkdtemp(prefix="st_seed_"))


import pytest


@pytest.fixture(autouse=True)
def _force_react_path(monkeypatch):
    """默认强制走 ReAct 文本路径（跳过供应商能力探测）。
    需要测原生 function calling 路径的用例可在此 fixture 之后单独 monkeypatch。"""
    from agent import protocol
    monkeypatch.setattr(protocol, "detect_native_support", lambda cfg: False)
    monkeypatch.setattr(protocol, "detect_json_mode", lambda cfg: False)
    # 同步清 embedding 探测缓存，避免测试间残留
    from rag import indexer
    indexer._reset_embed_support_cache()
