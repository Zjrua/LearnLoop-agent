"""agent/loop.py — Agent 主控：意图分类 → Executor(ReAct) → Writer。

run_agent 是生成器，yield SSE 事件 dict。上层（FastAPI）序列化成 text/event-stream。
注入点：chat_fn（可桩）、cfg、ctx；便于测试。
"""
from __future__ import annotations
import json
import re
from typing import Iterator

from agent.prompts import render_planner, render_executor, render_chat_direct, render_reflect
from agent.tools import TOOLS_EXECUTOR, tool_schema_text
from agent.tool_runtime import execute_tool, Context
from agent.protocol import resolve_tool_calls


_REF_RE = re.compile(r"([#@$])([^\s#@$，。、]+)")


def extract_refs(text: str) -> list[tuple[str, str]]:
    """提取文本里的 #/@/$ 引用，返回 [(symbol, key), ...]，去重保序。"""
    seen = set()
    out = []
    for m in _REF_RE.finditer(text):
        tag = (m.group(1), m.group(2))
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def inject_refs(system_prompt: str, refs_context: str) -> str:
    """把引用解析出的上下文注入 system prompt。无引用则原样返回。"""
    if not refs_context.strip():
        return system_prompt
    return system_prompt + "\n\n用户引用了以下内容（作为额外上下文）：\n" + refs_context


def parse_react(text: str) -> dict:
    """解析 ReAct 一步。返回 {type: tool|final, action?, arguments?, answer?}。
    正则化容错:Action 后中文注释/括号、Action 换行、多行 JSON Arguments。
    都不匹配→当 final(原样 answer,优雅降级)。"""
    # 工具步:Action 行
    m_action = re.search(r"Action\s*:\s*([A-Za-z_]\w*)", text)
    if m_action:
        action = m_action.group(1)
        m_args = re.search(r"Arguments\s*:\s*([\s\S]*?)(?=\n(?:Thought|Action|Final Answer)\s*:|$)", text)
        args: dict = {}
        if m_args:
            raw = m_args.group(1).strip()
            if raw:
                try:
                    parsed = json.loads(raw)
                    args = parsed if isinstance(parsed, dict) else {}
                except Exception:
                    args = {}
        return {"type": "tool", "action": action, "arguments": args}
    # Final Answer
    m_final = re.search(r"Final Answer\s*:\s*([\s\S]*)", text)
    if m_final:
        return {"type": "final", "answer": m_final.group(1).strip()}
    # 都没有→当 final(原样)
    return {"type": "final", "answer": text.strip()}


# 默认 chat_fn：真实工具调用协议
def _default_chat(cfg, messages, tools, stream=False, response_format=None):
    from agent.protocol import chat_with_tools, chat_stream
    if stream:
        # 流式：返回迭代器，逐 token yield {type:delta}
        return chat_stream(cfg, messages, tools)
    res = chat_with_tools(cfg, messages, tools, response_format=response_format)
    return res


def run_agent(ctx: Context, user_input: str, chat_fn=_default_chat,
              cfg: dict | None = None, max_steps: int = 6,
              history: list[dict] | None = None) -> Iterator[dict]:
    cfg = cfg or {}
    tools = TOOLS_EXECUTOR
    graph_summary = _graph_summary(ctx.graph)

    # ── 1. 意图分类（chat/query/mutate/produce 四类）──
    sys_p = render_planner(progress_summary=graph_summary, user_input=user_input)
    json_fmt = _json_fmt(cfg)   # 探测 JSON mode（支持才传，否则 None 走 _safe_intent 容错）
    try:
        pres = chat_fn(cfg, [{"role": "system", "content": sys_p},
                             {"role": "user", "content": user_input}], tools=None,
                       response_format=json_fmt)
        intent = _safe_intent(pres.get("content", ""))
    except Exception:
        intent = {"intent": "query", "needs_doc": False}

    yield {"type": "thinking", "content": f"意图：{intent.get('intent', 'query')}"}

    # ── 2a. chat 短路:单步直答,不进 ReAct ──
    if intent.get("intent") == "chat":
        yield from _chat_direct(user_input, history or [], chat_fn, cfg)
        yield {"type": "done"}
        return

    # ── 2. Executor（chat 短路；其余走 ReAct 或原生 function calling）──
    sys_e = render_executor(tools_text=tool_schema_text(tools), graph_summary=graph_summary)
    # mutate 意图：实测 2/2 用例模型只给文字建议、不调 add_node/add_tasks（核心写链路失效）。
    # 显式注入强指令，把「修改」从「给建议」掰回「必须调工具产可确认的 node_proposal 卡片」。
    if intent.get("intent") == "mutate":
        sys_e = sys_e + (
            "\n\n【本轮强制要求】用户意图是修改技能树。你必须调用工具（add_node / add_tasks / toggle_task）"
            "生成可确认的建议卡片（node_proposal），不要只用文字描述该怎么改。"
        )
    # 引用预处理：解析用户消息里的 #/@/$ 并注入上下文
    refs_text = ""
    has_dir_ref = False   # 是否有 $方向 引用（需聚焦）
    if hasattr(ctx, "resolve_refs_fn") and ctx.resolve_refs_fn:
        refs = extract_refs(user_input)
        if refs:
            refs_str = " ".join(f"{s}{k}" for s, k in refs)
            has_dir_ref = any(s == "$" for s, _ in refs)
            try:
                resolved = ctx.resolve_refs_fn(refs_str)
                refs_text = "\n".join(r.get("content", "") for r in resolved)
            except Exception:
                refs_text = ""
    # 有 $方向 引用时，加聚焦指令，避免模型跑偏到其他方向
    if has_dir_ref:
        dir_names = ", ".join(k for s, k in extract_refs(user_input) if s == "$")
        refs_text = (refs_text + f"\n\n【重要】用户聚焦于「{dir_names}」方向，"
                     f"请只围绕该方向的节点和进度回答，不要扯到其他学习方向。"
                     f"若需查该方向详情，调用 get_direction 工具。")
    sys_e = inject_refs(sys_e, refs_text)
    messages = [{"role": "system", "content": sys_e}]
    # 只注入最近 8 条历史（前端已 slice(-12)，此处再收一层防长上下文累积触发供应商 400）
    for m in (history or [])[-8:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_input})

    # 会话级选定一条路：探测供应商是否支持原生 function calling。
    # 支持则全程走原生（assistant tool_calls + tool role 回填），不支持则全程走 ReAct 文本。
    # 两条路不再交叉。
    use_native = False
    try:
        from agent.protocol import detect_native_support
        use_native = detect_native_support(cfg)
    except Exception:
        use_native = False
    yield {"type": "thinking", "content": f"工具协议：{'原生 function calling' if use_native else 'ReAct 文本'}"}

    reflect_used = False   # Reflexion 仅允许 1 轮(query/produce)
    observations_log: list[str] = []   # P0-#2: 独立累积，不再从 messages 反推
    format_retried = False   # P0-#3: ReAct 格式不符重试一次标志

    for step_i in range(max_steps):
        # 本轮是否走原生路径：use_native 且上一轮没因原生报错降级
        native_this_turn = use_native
        try:
            res = chat_fn(cfg, messages, tools if native_this_turn else None)
        except Exception:
            if native_this_turn:
                # 原生路径报错 → 本次降级 ReAct 文本（不改 use_native 标志，下次仍尝试原生）
                native_this_turn = False
                try:
                    res = chat_fn(cfg, messages, tools=None)
                except Exception as e2:
                    yield {"type": "error", "content": f"模型调用失败: {e2}"}
                    yield {"type": "done"}
                    return
            else:
                yield {"type": "error", "content": "模型调用失败"}
                yield {"type": "done"}
                return

        content = res.get("content", "")

        # ── 原生路径：解析 tool_calls，按 OpenAI 标准回填 messages ──
        if native_this_turn:
            calls = resolve_tool_calls(res, chat_fn)
            if calls:
                # 执行所有工具调用（不再只取第一个）
                # 先回填 assistant 消息（带原始 tool_calls 结构）
                messages.append({"role": "assistant", "content": content or "",
                                 "tool_calls": res.get("tool_calls", [])})
                for tc in calls:
                    action = tc.name
                    args = tc.arguments
                    yield {"type": "tool_call", "action": action, "arguments": args}
                    try:
                        tresult = execute_tool(action, args, ctx)
                        observation = tresult.get("text", "")
                        for ev in tresult.get("events", []):
                            yield ev
                    except Exception as e:
                        observation = f"工具执行出错: {e}"
                    observations_log.append(observation)
                    yield {"type": "tool_result", "action": action, "content": observation}
                    # OpenAI 标准：每个工具结果用 role=tool + tool_call_id 回填
                    # tool_call_id 从原始 tool_calls 取（FakeChat/真实供应商都有）
                    tc_id = _find_tool_call_id(res.get("tool_calls", []), action)
                    messages.append({"role": "tool", "tool_call_id": tc_id,
                                     "content": observation})
                continue   # 进入下一轮，让模型基于工具结果继续

            # 原生路径无 tool_calls → 当作 final answer
            step = {"type": "final", "answer": content.strip()}
        else:
            # ── ReAct 文本路径 ──
            calls = resolve_tool_calls(res, chat_fn)   # 兜底：模型在 ReAct 模式仍输出 <tool_call>
            if calls:
                # 模型在 ReAct 模式输出了指令式工具标记 → 当工具步处理（走文本回填）
                step = {"type": "tool", "action": calls[0].name,
                        "arguments": calls[0].arguments}
            elif "Action:" in content or "Final Answer:" in content:
                step = parse_react(content)
            else:
                # P0-#3: 既无 Action 也无 Final Answer → 格式不符，重试一次
                if not format_retried and (step_i + 1) < max_steps:
                    format_retried = True
                    yield {"type": "thinking", "content": "输出格式不符，提示重新用 ReAct 格式"}
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user",
                                     "content": "请严格用 ReAct 格式输出：Thought/Action/Arguments 或 Thought/Final Answer。"})
                    continue
                # 已重试过或即将超步数 → 当 final（优雅降级）
                step = {"type": "final", "answer": content.strip()}

        if step["type"] == "final":
            answer = step.get("answer", "")
            if not answer or answer.startswith("（已达到最大"):
                yield {"type": "final_answer", "content": answer}
                break
            # Reflexion(query/produce 才做,仅 1 轮)
            if intent.get("intent") in ("query", "produce") and not reflect_used:
                reflect_used = True   # 先置位:即使本轮续跑,下一轮 final 也不再 reflect(封顶 1 轮)
                ok, gap = _reflect(user_input, observations_log, answer, chat_fn, cfg)
                if not ok and (step_i + 1) < max_steps:
                    yield {"type": "thinking", "content": f"自查发现遗漏: {gap}，补充中…"}
                    messages.append({"role": "assistant", "content": answer})
                    messages.append({"role": "user",
                                     "content": f"上面的草稿遗漏: {gap}。请用工具补充后给出更完整的最终回答。"})
                    continue   # 续跑一轮(下一轮 final 时 reflect_used 已 True,不再 reflect)
            # 接受答案
            yield from _emit_text_as_delta(answer)
            yield {"type": "final_done"}
            break

        # 工具步（ReAct 文本路径）
        action = step["action"]
        args = step.get("arguments", {})
        yield {"type": "tool_call", "action": action, "arguments": args}
        try:
            tresult = execute_tool(action, args, ctx)
            observation = tresult.get("text", "")
            for ev in tresult.get("events", []):
                yield ev
        except Exception as e:
            observation = f"工具执行出错: {e}"
        observations_log.append(observation)
        yield {"type": "tool_result", "action": action, "content": observation}

        # ReAct 文本路径：把这一轮塞回 messages 维持多轮
        messages.append({"role": "assistant", "content": content or f"Action: {action}"})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    else:
        # 达到最大步数仍未收敛 → 降级 final
        yield {"type": "final_answer", "content": "（已达到最大推理步数，基于已有信息停止。）"}

    # ── 3. Writer（条件触发）──
    if intent.get("needs_doc"):
        doc_type = intent.get("doc_type")
        yield from _run_writer(ctx, user_input, messages, chat_fn, cfg,
                               doc_type=doc_type, observations_log=observations_log)

    yield {"type": "done"}


def _find_tool_call_id(raw_tool_calls: list, action: str) -> str:
    """从原始 tool_calls 结构里找匹配 action 的 call id（OpenAI 标准回填用）。
    FakeChat 可能直接传 ToolCall 对象或 dict；找不到时回退 'call_0'。"""
    for i, tc in enumerate(raw_tool_calls or []):
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            if fn.get("name") == action:
                return tc.get("id", f"call_{i}")
        else:
            # ToolCall dataclass
            if getattr(tc, "name", None) == action:
                return getattr(tc, "id", f"call_{i}")
    return "call_0"


def _chat_direct(user_input, history, chat_fn, cfg) -> Iterator[dict]:
    """chat 意图:单步直答。无 Reflexion 依赖，直接流式逐 token 返回；
    流式失败/无产出时回退为「拿全文本再 chunk」。"""
    sys_c = render_chat_direct(_history_summary(history))
    messages = [{"role": "system", "content": sys_c}]
    # 摘要已覆盖最近 6 条，messages 只补最近 8 条原文，避免全量历史冗余膨胀
    for m in history[-8:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_input})

    # 优先真流式：stream=True 时 chat_fn 返回 delta 迭代器（首 token 延迟低）
    try:
        res = chat_fn(cfg, messages, tools=None, stream=True)
    except Exception:
        res = None

    if res is not None and not isinstance(res, dict):
        streamed = False
        try:
            for ev in res:
                if isinstance(ev, dict) and ev.get("type") == "delta":
                    streamed = True
                    yield ev
        except Exception as e:
            # 已产出部分内容无法撤回，标记中断
            yield {"type": "delta", "content": f"（流式中断: {e}）"}
            streamed = True
        if streamed:
            yield {"type": "final_done"}
            return

    # 回退：非流式整段返回，再 chunk
    try:
        res2 = chat_fn(cfg, messages, tools=None)
        answer = res2.get("content", "") if isinstance(res2, dict) else ""
    except Exception as e:
        answer = f"（生成失败: {e}）"
    yield from _emit_text_as_delta(answer or "（无回复）")
    yield {"type": "final_done"}


def _emit_text_as_delta(text: str, chunk_size: int = 8) -> Iterator[dict]:
    """把完整文本 chunk 成 delta 事件(统一流式口径:先拿全文本再分段吐)。"""
    for i in range(0, len(text), chunk_size):
        yield {"type": "delta", "content": text[i:i + chunk_size]}


def _reflect(user_input, observations_log: list[str], draft, chat_fn, cfg) -> tuple[bool, str]:
    """校验草稿是否回答了问题。返回 (ok, gap)。失败回退 ok=True(不阻塞主流程)。
    observations_log: 工具执行结果独立累积，不再从 messages 反推（P0-#2）。"""
    observations = "\n".join(observations_log)[:2000]
    sys_r = render_reflect(question=user_input, observations=observations, draft=draft[:800])
    json_fmt = _json_fmt(cfg)   # 探测 JSON mode
    try:
        res = chat_fn(cfg, [{"role": "system", "content": sys_r},
                            {"role": "user", "content": "请输出校验 JSON。"}], tools=None,
                      response_format=json_fmt)
        obj = _safe_intent(res.get("content", ""))
        return bool(obj.get("ok", True)), str(obj.get("gap", ""))
    except Exception:
        return True, ""   # 回退:不阻塞


def _json_fmt(cfg: dict) -> dict | None:
    """探测供应商是否支持 JSON mode，支持则返回 response_format，否则 None（走容错解析）。"""
    try:
        from agent.protocol import detect_json_mode
        return {"type": "json_object"} if detect_json_mode(cfg) else None
    except Exception:
        return None


def _history_summary(history: list[dict]) -> str:
    if not history:
        return ""
    lines = []
    for m in history[-6:]:
        role = "用户" if m.get("role") == "user" else "助手"
        lines.append(f"{role}: {m.get('content','')[:80]}")
    return "\n".join(lines)


def _pick_doc_type(request: str) -> str:
    """按用户措辞判定文档类型（意图分类未给 doc_type 时的兜底）。"""
    r = request
    if any(k in r for k in ("复习", "自测", "review")):
        return "review"
    if any(k in r for k in ("周报", "本周", "总结", "weekly")):
        return "weekly"
    return "note"


def _run_writer(ctx, user_input, executor_messages, chat_fn, cfg,
                doc_type: str | None = None,
                observations_log: list[str] | None = None) -> Iterator[dict]:
    from agent.prompts import render_writer
    if not doc_type:
        doc_type = _pick_doc_type(user_input)
    # 素材来自工具执行结果：observations_log 在原生 function calling 与 ReAct 两条路径都会累积。
    # 旧实现从 messages 里反推 role=user 的 "Observation" 文本，但原生路径下工具结果用 role=tool
    # 回填、不含 "Observation" 字样，导致 produce 场景（整理笔记/复习卡/周报）素材恒为空、Writer 空手编造。
    if observations_log is not None:
        materials = "\n".join(observations_log)
    else:
        materials = "\n".join(m.get("content", "") for m in executor_messages
                              if m.get("role") == "user" and "Observation" in m.get("content", ""))
    sys_w = render_writer(materials=materials[:2000], request=user_input, doc_type=doc_type)
    try:
        res = chat_fn(cfg, [{"role": "system", "content": sys_w},
                            {"role": "user", "content": f"请生成{doc_type}类型文档。"}], tools=None)
        content = res.get("content", "")
        # 标题:取首个 <title>...</title>,否则用类型默认名
        if "<title>" in content and "</title>" in content:
            title = content.split("</title>")[0].split("<title>")[-1]
        else:
            title = {"note": "学习笔记", "review": "复习卡", "weekly": "周报"}.get(doc_type, "文档")
        yield {"type": "doc_card", "doc_type": doc_type, "content": content, "title": title}
    except Exception as e:
        yield {"type": "error", "content": f"文档生成失败: {e}"}


def _safe_intent(text: str) -> dict:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {"intent": "query", "needs_doc": False}


def _graph_summary(graph: dict) -> str:
    ov = graph.get("overview", {})
    nodes = graph.get("nodes", [])
    names = "、".join(n.get("name", n.get("id")) for n in nodes[:12])
    return (f"整体 {ov.get('overall_pct',0)}%"
            f"，已掌握 {ov.get('mastered_points',0)}/{ov.get('total_points',0)}。"
            f"节点：{names}")
