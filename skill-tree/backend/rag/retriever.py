"""rag/retriever.py — 混合检索：精确(图谱/简历 BM25) + 向量(源码) + 外部(论文)，带排序融合。"""
from __future__ import annotations
import math
import re
from pathlib import Path

from rag import store
from rag import paper_fetch
from rag.indexer import cosine, embed

# 融合权重。实验 4（eval/RESULTS.md）显示 W_STRUCT=0.4 时结构化通道命中率 0%、0.5 时 100%——
# 0.4 恰低于拐点，导致查图谱内容时被向量通道的系统性压制。取等权 0.5 消除该压制。
W_STRUCT = 0.5
W_VEC = 0.5
W_EXT = 0.1

# BM25 参数
_BM25_K1 = 1.5
_BM25_B = 0.75

try:
    import numpy as np  # 可选加速矩阵化检索
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _tokens(q: str) -> list[str]:
    """把查询切成关键词（中英混合）。
    英文/数字按空格与标点切词；中文按字符 bigram 切（零依赖）。
    旧实现把整段中文当单个 token，导致中文 BM25 全失效（见 eval/RESULTS.md「中文分词」）。"""
    out: list[str] = []
    for seg in re.split(r"[\s,，、。./]+", q.lower()):
        for w in re.findall(r"[a-z0-9_]+", seg):
            if len(w) >= 2:
                out.append(w)
        cjk = _CJK_RE.findall(seg)
        if len(cjk) == 1:
            out.append(cjk[0])
        else:
            out.extend("".join(cjk[i:i + 2]) for i in range(len(cjk) - 1))
    return out


def _bm25(query_tokens: list[str], doc_tokens: list[str],
          df_map: dict[str, int], N: int, avg_dl: float) -> float:
    """纯标准库 BM25 打分。query_tokens/doc_tokens 已分词；df_map=词→文档频率；N=文档总数；avg_dl=平均文档长度。
    返回分数（未归一化）。空 doc 或空 query 返回 0。"""
    if not query_tokens or not doc_tokens or avg_dl == 0:
        return 0.0
    dl = len(doc_tokens)
    # 词频统计
    tf: dict[str, int] = {}
    for t in doc_tokens:
        tf[t] = tf.get(t, 0) + 1
    score = 0.0
    for qt in query_tokens:
        if qt not in tf:
            continue
        f = tf[qt]
        df = df_map.get(qt, 0)
        if df == 0:
            continue
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
        denom = f + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avg_dl)
        score += idf * (f * (_BM25_K1 + 1)) / denom
    return score


def _build_df(docs_tokens: list[list[str]]) -> tuple[dict[str, int], float]:
    """从一批文档词列表构建 df_map + avg_dl。"""
    df_map: dict[str, int] = {}
    total_dl = 0
    for dt in docs_tokens:
        for t in set(dt):
            df_map[t] = df_map.get(t, 0) + 1
        total_dl += len(dt)
    avg_dl = total_dl / len(docs_tokens) if docs_tokens else 0.0
    return df_map, avg_dl


class Retriever:
    def __init__(self, index_dir, cfg: dict):
        self.index_dir = Path(index_dir)
        self.cfg = cfg
        self._matrix = None        # numpy 矩阵缓存 (N, D)
        self._matrix_ids: list[str] = []   # 与矩阵行对应的 chunk id
        self._matrix_dim: int = 0

    def invalidate(self) -> None:
        """索引重建后清矩阵缓存（让下次 search 重新加载）。"""
        self._matrix = None
        self._matrix_ids = []
        self._matrix_dim = 0

    def _load_matrix(self) -> None:
        """懒加载所有 chunk 向量堆成 (N, D) 矩阵（numpy 可用时）。"""
        if not _HAS_NUMPY or self._matrix is not None:
            return
        chunks = store.read_chunks(self.index_dir / "code_chunks.jsonl")
        vecs, ids, dim = [], [], 0
        for c in chunks:
            v = c.get("vector")
            if v:
                vecs.append(v)
                ids.append(c.get("id", ""))
                dim = len(v)
        if vecs and dim:
            self._matrix = np.asarray(vecs, dtype=float)   # (N, D)
            self._matrix_ids = ids
            self._matrix_dim = dim

    def search(self, query: str, top_k: int = 5,
               graph: dict | None = None, resume: dict | None = None,
               paper_cache=None) -> list[dict]:
        """混合检索。返回 [{id, source, text, score, ref}]，按分数降序。"""
        results: list[dict] = []
        qt = _tokens(query)

        # 1) 向量通道（源码）
        chunks = store.read_chunks(self.index_dir / "code_chunks.jsonl")
        if chunks:
            qvec = embed(self.cfg, [query])[0] if query.strip() else None
            if qvec:
                scored = self._vector_scores(chunks, qvec)
                if scored:
                    mx = scored[0][0] if scored[0][0] > 0 else 1.0
                    for sc, c in scored[:top_k]:
                        if sc <= 0:
                            continue
                        results.append({"id": c["id"], "source": "code", "file": c.get("file", ""),
                                        "symbol": c.get("symbol", ""), "text": c.get("text", ""),
                                        "score": W_VEC * (sc / mx if mx else 0.0)})

        # 2) 结构化通道（图谱）— BM25 打分
        if graph and qt:
            graph_docs = []
            graph_nodes = []
            for n in graph.get("nodes", []):
                blob = (n.get("name", "") + " " + n.get("category", "") + " " +
                        " ".join(t.get("title", "") for t in n.get("tasks", [])))
                graph_docs.append(_tokens(blob))
                graph_nodes.append(n)
            df_map, avg_dl = _build_df(graph_docs)
            scored = [(score, n) for score, n in
                      ((_bm25(qt, dt, df_map, len(graph_docs), avg_dl), n)
                       for dt, n in zip(graph_docs, graph_nodes))
                      if score > 0]
            scored.sort(key=lambda t: t[0], reverse=True)
            mx = scored[0][0] if scored and scored[0][0] > 0 else 1.0
            for sc, n in scored[:top_k]:
                results.append({"id": "graph:" + n.get("id", ""), "source": "graph",
                                "text": f"[节点] {n.get('name')} ({n.get('category','')})",
                                "score": W_STRUCT * (sc / mx if mx else 0.0)})

        # 3) 简历通道（BM25）
        if resume and qt:
            resume_docs = []
            resume_exps = []
            for exp in resume.get("experience", []):
                blob = str(exp.get("desc", "")) + " " + str(exp.get("title", ""))
                resume_docs.append(_tokens(blob))
                resume_exps.append(exp)
            df_map, avg_dl = _build_df(resume_docs)
            scored = [(score, exp) for score, exp in
                      ((_bm25(qt, dt, df_map, len(resume_docs), avg_dl), exp)
                       for dt, exp in zip(resume_docs, resume_exps))
                      if score > 0]
            scored.sort(key=lambda t: t[0], reverse=True)
            mx = scored[0][0] if scored and scored[0][0] > 0 else 1.0
            for sc, exp in scored[:top_k]:
                results.append({"id": "resume:" + str(exp.get("title", "")), "source": "resume",
                                "text": f"[简历] {exp.get('title')}: {exp.get('desc','')}",
                                "score": W_STRUCT * (sc / mx if mx else 0.0)})

        # 3.5) 外部通道（论文）：query 含 arxiv id 时抓 abstract
        if paper_cache is not None:
            aid = paper_fetch.extract_arxiv_id(query)
            if not aid:
                # 退路：匹配裸 arxiv id（如 "1703.04247"）
                m = re.search(r"\b(\d{4}\.\d{4,5})\b", query)
                aid = m.group(1) if m else None
            if aid:
                try:
                    rec = paper_fetch.fetch_abstract(f"https://arxiv.org/abs/{aid}", paper_cache)
                    if rec.get("abstract"):
                        results.append({"id": "paper:" + aid, "source": "paper",
                                        "text": f"[论文] {rec.get('title', aid)}: {rec['abstract'][:200]}",
                                        "score": W_EXT})
                except Exception:
                    pass

        # 去重（同 id 取高分）+ 归一化排序 + 加引用编号
        best: dict[str, dict] = {}
        for r in results:
            bid = r["id"]
            if bid not in best or r["score"] > best[bid]["score"]:
                best[bid] = r
        out = sorted(best.values(), key=lambda r: r["score"], reverse=True)[:top_k]
        for i, r in enumerate(out, 1):
            r["ref"] = f"[{i}]"
        return out

    def _vector_scores(self, chunks: list[dict], qvec: list[float]) -> list[tuple[float, dict]]:
        """向量通道打分。numpy 可用时矩阵化（一次 np.dot 出全部分数），否则逐个 cosine。"""
        if _HAS_NUMPY:
            # 矩阵化路径
            if self._matrix is None:
                self._load_matrix()
            if self._matrix is not None and len(qvec) == self._matrix_dim:
                q = np.asarray(qvec, dtype=float)
                qn = np.linalg.norm(q)
                if qn == 0.0:
                    return []
                # 矩阵归一化后一次点积
                norms = np.linalg.norm(self._matrix, axis=1)
                safe = norms > 0
                sims = np.zeros(len(self._matrix))
                if safe.any():
                    sims[safe] = (self._matrix[safe] @ q) / (norms[safe] * qn)
                # 重建 (score, chunk) —— 用矩阵对应的 id 找回 chunk
                id_to_chunk = {c.get("id"): c for c in chunks if c.get("vector")}
                out = []
                for i, cid in enumerate(self._matrix_ids):
                    c = id_to_chunk.get(cid)
                    if c:
                        out.append((float(sims[i]), c))
                out.sort(key=lambda t: t[0], reverse=True)
                return out
        # 回退路径：逐个 cosine
        scored = [(cosine(qvec, c.get("vector") or []), c) for c in chunks if c.get("vector")]
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored
