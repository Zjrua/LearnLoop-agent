"""chat_store.py — 多会话对话存储（JSON 文件，单用户本地存储）。

数据模型: {sessions: [Session], current_session_id: str|null}
Session: {id, title, created_at, updated_at, messages: [Msg]}
Msg: {role, content, ts, events?: [...]}
"""
from __future__ import annotations
import json
import time
import re
from pathlib import Path


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class ChatStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> dict:
        if not self.path.exists():
            return {"sessions": [], "current_session_id": None, "updated_at": _now()}
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, state: dict) -> None:
        state["updated_at"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def new_session(self) -> str:
        state = self.load()
        sid = f"s_{int(time.time() * 1000)}"
        state["sessions"].append({
            "id": sid, "title": "新会话", "created_at": _now(),
            "updated_at": _now(), "messages": [],
        })
        state["current_session_id"] = sid
        self._save(state)
        return sid

    def append_message(self, session_id: str, msg: dict) -> None:
        state = self.load()
        for s in state["sessions"]:
            if s["id"] == session_id:
                s["messages"].append(msg)
                s["updated_at"] = _now()
                break
        self._save(state)

    def set_current(self, session_id: str) -> None:
        state = self.load()
        state["current_session_id"] = session_id
        self._save(state)

    def set_title(self, session_id: str, title: str) -> None:
        state = self.load()
        for s in state["sessions"]:
            if s["id"] == session_id:
                s["title"] = title
                break
        self._save(state)

    def delete_session(self, session_id: str) -> None:
        state = self.load()
        state["sessions"] = [s for s in state["sessions"] if s["id"] != session_id]
        if state["current_session_id"] == session_id:
            state["current_session_id"] = state["sessions"][0]["id"] if state["sessions"] else None
        self._save(state)

    def replace_all(self, sessions: list, current_session_id: str | None) -> None:
        """双写同步：前端发回完整数据，后端覆盖。"""
        self._save({"sessions": sessions, "current_session_id": current_session_id})

    def search(self, query: str) -> list[dict]:
        """跨所有会话子串搜索消息。返回 [{session_id, session_title, message_index, role, snippet}]。"""
        if not query.strip():
            return []
        q = query.lower()
        hits = []
        for s in self.load()["sessions"]:
            for i, m in enumerate(s["messages"]):
                content = m.get("content", "")
                if q in content.lower():
                    idx = content.lower().find(q)
                    start = max(0, idx - 15)
                    snippet = content[start:start + len(q) + 30]
                    hits.append({
                        "session_id": s["id"], "session_title": s["title"],
                        "message_index": i, "role": m.get("role", ""),
                        "snippet": ("…" if start > 0 else "") + snippet,
                    })
        return hits

    def export(self, session_id: str | None) -> dict:
        """导出为 JSON 可序列化结构。session_id=None 导出全部。"""
        state = self.load()
        if session_id is None:
            return {"sessions": state["sessions"], "exported_at": _now()}
        for s in state["sessions"]:
            if s["id"] == session_id:
                return {**s, "exported_at": _now()}
        return {}

    # ── 符号引用 ──
    # 形如 #node_id  @resource_id  $dir_keyword
    _REF_RE = re.compile(r"([#@$])([^\s#@$，。、]+)")

    def resolve_refs(self, text: str, graph: dict, dirs: list, resources: list) -> list[dict]:
        """解析文本里的 #/@/$ 引用，返回各对象展开内容 [{type, id, content}]。"""
        out = []
        seen = set()
        for m in self._REF_RE.finditer(text):
            sym, key = m.group(1), m.group(2).lower()
            tag = (sym, key)
            if tag in seen:
                continue
            seen.add(tag)
            if sym == "#":
                for n in graph.get("nodes", []):
                    if key in (n.get("id", "").lower(), n.get("name", "").lower()):
                        tasks = ", ".join(t.get("title", "") for t in n.get("tasks", []))
                        out.append({"type": "node", "id": n.get("id"), "name": n.get("name"),
                                    "content": f"[节点] {n.get('name')}({n.get('category','')}), "
                                               f"依赖: {n.get('depends_on',[])}, 任务: {tasks}"})
                        break
            elif sym == "$":
                for d in dirs:
                    if key in d.get("id", "").lower() or key in d.get("title", "").lower():
                        # 展开该方向所有节点 + 进度 + 下一步建议
                        nodes = d.get("nodes", [])
                        lines = [f"[方向] {d.get('title')} {d.get('icon','')}"]
                        if nodes:
                            lines.append("节点进度：")
                            ready = []  # 前置已满足可推进的 locked 节点
                            for n in nodes:
                                state = n.get("state", "locked")
                                pct = n.get("pct", 0)
                                lines.append(f"- {n.get('name','?')} ({state}, {pct}%)")
                                if state == "locked":
                                    deps = n.get("depends_on", [])
                                    # 前置全部 done/learning 视为可推进
                                    dep_nodes = {x.get("id"): x for x in nodes}
                                    if all(dep_nodes.get(dep, {}).get("state") in ("done", "learning")
                                           for dep in deps if dep in dep_nodes):
                                        ready.append(n.get("name", n.get("id")))
                            if ready:
                                lines.append(f"可推进的下一步：{', '.join(ready[:5])}")
                        out.append({"type": "dir", "id": d.get("id"), "name": d.get("title"),
                                    "content": "\n".join(lines)})
                        break
            elif sym == "@":
                for r in resources:
                    if key in r.get("id", "").lower() or key in r.get("label", "").lower():
                        out.append({"type": "resource", "id": r.get("id"), "name": r.get("label"),
                                    "content": f"[资源] {r.get('label')}: {r.get('url','')}"})
                        break
        return out

    def suggest(self, ref_type: str, prefix: str, graph: dict, dirs: list, resources: list) -> list[dict]:
        """mention 补全：按类型 + 前缀模糊匹配。返回 [{id, name, label?}]。"""
        p = prefix.lower()
        out = []
        if ref_type == "node":
            for n in graph.get("nodes", []):
                if p in n.get("id", "").lower() or p in n.get("name", "").lower():
                    out.append({"id": n.get("id"), "name": n.get("name")})
        elif ref_type == "dir":
            for d in dirs:
                if p in d.get("id", "").lower() or p in d.get("title", "").lower():
                    out.append({"id": d.get("id"), "name": d.get("title")})
        elif ref_type == "resource":
            for r in resources:
                if p in r.get("id", "").lower() or p in r.get("label", "").lower():
                    out.append({"id": r.get("id"), "name": r.get("label")})
        return out[:8]
