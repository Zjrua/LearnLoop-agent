"""
eval/run_eval.py — Agent 实测脚本:跑黄金用例,统计量化指标。

用法:
  cd skill-tree/backend        # 必须在 backend 目录(模块路径)
  set DATA_ROOT=..\\..\\skill-tree\\data
  python ..\\..\\eval\\run_eval.py

统计指标:
  1. Planner 意图分类准确率(chat/query/mutate/produce 四类)
  2. 工具调用成功率(是否正确触发预期工具 + 是否无报错完成)
  3. Reflexion 触发率 / 续跑率
  4. 平均响应延迟(端到端,秒)
  5. 平均 Token 消耗(prompt+completion)

读取配置:eval/config.local.json(已 gitignore)。
结果输出:eval/results/eval_<timestamp>.json + 控制台汇总。

数据口径:所有数字来自真实 LLM 调用。config.local.json 含 api_key 不入库。
"""
from __future__ import annotations
import json
import sys
import time
import urllib.request
from pathlib import Path

# ── 配置加载 ──
EVAL_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EVAL_DIR / "config.local.json"

if not CONFIG_PATH.exists():
    print("[ERROR] 找不到 eval/config.local.json。请复制 config.local.json.example 并填入真实配置。")
    sys.exit(1)

CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
CHAT_CFG = CONFIG["chat"]


# ── 黄金用例:每条 = (问题, 期望意图, 期望触发的工具集合或None) ──
# 期望意图:与 prompts.py SYS_PLANNER 的四类对齐(chat/query/mutate/produce)
# 期望工具:None=不期望工具(chat);list=期望命中的工具名(任一即可)
GOLDEN = [
    # chat:闲聊/问候,不应触发工具
    ("你好", "chat", None),
    ("学算法有什么用", "chat", None),
    ("谢谢", "chat", None),
    # query:查状态/查知识,应触发 get_progress / search_knowledge / get_node
    ("我整体进度怎么样", "query", {"get_progress"}),
    ("我学到哪了", "query", {"get_progress"}),
    ("下一步学什么", "query", {"get_next", "get_progress"}),
    ("DeepFM 是什么", "query", {"search_knowledge"}),
    ("推荐方向有哪些内容", "query", {"get_direction"}),
    # mutate:改图谱,应触发 add_node/add_tasks/toggle_task
    ("加一个 LightGCN 节点", "mutate", {"add_node"}),
    ("帮我加个 xDeepFM", "mutate", {"add_node"}),
    # produce:产出文档,needs_doc=true
    ("帮我整理个 DeepFM 的学习笔记", "produce", None),
    ("生成一份复习卡", "produce", None),
    ("整理本周学习周报", "produce", None),
]


# ── 真实 chat_fn 包装:记录每次调用的 latency + token usage ──
class CallRecorder:
    """包一层 chat_with_tools,统计 latency 与 token usage。"""
    def __init__(self, base_cfg: dict):
        self.base_cfg = base_cfg
        self.latencies: list[float] = []
        self.usages: list[dict] = []

    def __call__(self, cfg, messages, tools=None, stream=False, response_format=None):
        # 强制用 base_cfg 的 base_url/api_key/model(忽略传入的空 cfg)
        merged = {**self.base_cfg, **{k: v for k, v in (cfg or {}).items() if v}}
        t0 = time.time()
        res = self._call(merged, messages, tools, stream, response_format)
        self.latencies.append(time.time() - t0)
        self._last_usage = res.get("__usage__")
        if self._last_usage:
            self.usages.append(self._last_usage)
        return res

    def _call(self, cfg, messages, tools, stream, response_format):
        from agent.protocol import chat_with_tools
        # chat_with_tools 不返回 usage;用底层 HTTP 抓 usage
        base = cfg["base_url"].rstrip("/")
        url = f"{base}/chat/completions"
        body = {
            "model": cfg.get("model") or "gpt-3.5-turbo",
            "messages": messages,
            "temperature": 0.5,
        }
        if tools:
            from agent.protocol import to_tool_schemas
            body["tools"] = to_tool_schemas(tools)
        if response_format:
            body["response_format"] = response_format
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {cfg['api_key']}")
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        msg = data["choices"][0]["message"]
        return {
            "content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls") or [],
            "__usage__": data.get("usage"),
        }


def run_one(question: str, graph_ctx, recorder: CallRecorder):
    """跑一条用例,返回事件序列 + 统计。"""
    from agent.loop import run_agent
    events = []
    t0 = time.time()
    try:
        for ev in run_agent(graph_ctx, question, chat_fn=recorder, cfg=CHAT_CFG, max_steps=6):
            events.append(ev)
    except Exception as e:
        events.append({"type": "error", "content": f"{type(e).__name__}: {e}"})
    wall = time.time() - t0
    return events, wall


def summarize(events: list[dict]) -> dict:
    """从事件序列提炼指标。"""
    types = [e["type"] for e in events]
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    tools_used = {e["action"] for e in tool_calls}
    tool_errors = [e for e in events if e["type"] == "error"]
    thinking = [e.get("content", "") for e in events if e["type"] == "thinking"]
    # Reflexion 续跑：reflect 判 not ok → loop 注入 gap 续跑，thinking 事件含「自查发现遗漏」。
    # 注意：这测的是「自校验后发生续跑」的比例，不是「自校验被触发」的比例——
    # _reflect 对 query/produce 的 final 答案无条件执行，触发率恒为 100%（设计如此）。
    reflect_rerun = any("自查发现遗漏" in t for t in thinking)
    has_final = any(e["type"] == "final_done" for e in events)
    has_error = bool(tool_errors)
    return {
        "tools_used": sorted(tools_used),
        "n_tool_calls": len(tool_calls),
        "reflect_rerun": reflect_rerun,
        "has_final": has_final,
        "has_error": has_error,
        "error": "; ".join(e.get("content", "") for e in tool_errors),
        "thinking": thinking,
    }


def main():
    # 复用 backend 的 _build_ctx:构建带真实图谱(layout+掌握度)的 Context。
    # DATA_ROOT 由 env 指向 skill-tree/data(含 recommendation.json/agent.json 等真实树)。
    import os
    # 必须在 import main 前设好 DATA_ROOT(它在 import 时就读)
    if "DATA_ROOT" not in os.environ:
        data_dir = Path(__file__).resolve().parent.parent / "skill-tree" / "data"
        os.environ["DATA_ROOT"] = str(data_dir)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skill-tree" / "backend"))
    import main as be_main  # backend main.py

    ctx, _ = be_main._build_ctx()
    # 覆盖 ctx.cfg:让 add_node 等(ai.generate_node)用实测 chat 配置
    ctx.cfg = CHAT_CFG
    print(f"[INFO] DATA_ROOT = {be_main.DATA_ROOT}")
    print(f"[INFO] 图谱节点数 = {len(ctx.graph.get('nodes', []))}, "
          f"整体掌握度 = {ctx.graph.get('overview', {}).get('overall_pct', 0)}%")

    recorder = CallRecorder(CHAT_CFG)

    results = []
    print(f"\n{'='*60}\n开始实测:共 {len(GOLDEN)} 条黄金用例\n{'='*60}\n")
    for i, (q, exp_intent, exp_tools) in enumerate(GOLDEN, 1):
        print(f"[{i}/{len(GOLDEN)}] {q}  (期望意图={exp_intent}, 期望工具={exp_tools})")
        events, wall = run_one(q, ctx, recorder)
        s = summarize(events)
        # 判定
        # 意图:从第一条 thinking 事件 "意图：X" 提取
        intent_actual = None
        for t in s["thinking"]:
            if t.startswith("意图"):
                intent_actual = t.split("：", 1)[-1].strip()
                break
        intent_ok = (intent_actual == exp_intent)
        # 工具:期望工具集合与实际有交集(或期望为 None 时实际无工具)
        if exp_tools is None:
            tool_ok = (s["n_tool_calls"] == 0)
        else:
            tool_ok = bool(s["tools_used"] and set(s["tools_used"]) & exp_tools)
        # 完成:无 error 且有 final
        complete_ok = s["has_final"] and not s["has_error"]

        results.append({
            "q": q, "exp_intent": exp_intent, "intent_actual": intent_actual,
            "intent_ok": intent_ok,
            "exp_tools": sorted(exp_tools) if exp_tools else None,
            "tools_used": s["tools_used"], "tool_ok": tool_ok,
            "reflect_rerun": s["reflect_rerun"],
            "complete_ok": complete_ok,
            "wall_s": round(wall, 2),
            "error": s["error"],
        })
        flag = "OK" if (intent_ok and tool_ok and complete_ok) else "XX"
        print(f"    -> 意图={intent_actual}({'' if intent_ok else 'X'}) "
              f"工具={s['tools_used'] or '-'}({'' if tool_ok else 'X'}) "
              f"Reflex续跑={s['reflect_rerun']} 完成={complete_ok} {wall:.1f}s [{flag}]")
        if s["error"]:
            print(f"       ERROR: {s['error'][:120]}")

    # ── 汇总统计 ──
    n = len(results)
    intent_acc = sum(r["intent_ok"] for r in results) / n
    tool_acc = sum(r["tool_ok"] for r in results) / n
    complete_rate = sum(r["complete_ok"] for r in results) / n
    # Reflexion 只对 query/produce 生效（chat 短路、mutate 走提案不 reflect），
    # 续跑率的分母应为 query+produce 用例数，而非全部 13 条。
    reflect_eligible = [r for r in results if r["exp_intent"] in ("query", "produce")]
    reflect_rerun_rate = (sum(r["reflect_rerun"] for r in reflect_eligible) / len(reflect_eligible)
                          if reflect_eligible else 0.0)
    avg_wall = sum(r["wall_s"] for r in results) / n
    # token:总 usage(全用例所有调用累计)
    total_prompt = sum(u.get("prompt_tokens", 0) for u in recorder.usages)
    total_completion = sum(u.get("completion_tokens", 0) for u in recorder.usages)
    total_tokens = sum(u.get("total_tokens", 0) for u in recorder.usages)
    n_calls = len(recorder.usages)
    avg_wall_chat = sum(recorder.latencies) / n_calls if n_calls else 0

    summary = {
        "n_cases": n,
        "intent_accuracy": round(intent_acc, 4),
        "tool_call_success": round(tool_acc, 4),
        "task_complete_rate": round(complete_rate, 4),
        "reflect_rerun_rate": round(reflect_rerun_rate, 4),
        "avg_end_to_end_latency_s": round(avg_wall, 2),
        "avg_single_call_latency_s": round(avg_wall_chat, 2),
        "total_llm_calls": n_calls,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "avg_tokens_per_case": round(total_tokens / n) if n else 0,
        "model": CHAT_CFG.get("model"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print(f"\n{'='*60}\n实测汇总 (model={summary['model']})\n{'='*60}")
    for k, v in summary.items():
        if k in ("model", "timestamp"):
            continue
        print(f"  {k:.<36} {v}")

    # 保存结果
    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eval_{ts}.json"
    out_path.write_text(
        json.dumps({"summary": summary, "cases": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[OK] 详细结果已保存: {out_path}")


if __name__ == "__main__":
    main()
