# tests/test_loop.py
from __future__ import annotations

from agent.loop import run_agent, parse_react
from agent.tool_runtime import Context
from tests.fakes import FakeChat


def _ctx(graph=None):
    return Context(uid="u", graph=graph or {"nodes": [], "overview": {}},
                   resume=None, retriever=None, rag_index_dir=None)


def test_parse_react_tool_step():
    text = "Thought: 要查进度\nAction: get_progress\nArguments: {}"
    step = parse_react(text)
    assert step["type"] == "tool"
    assert step["action"] == "get_progress"


def test_parse_react_final_answer():
    text = "Thought: 够了\nFinal Answer: 建议学 DCN。"
    step = parse_react(text)
    assert step["type"] == "final"
    assert "DCN" in step["answer"]


def test_parse_react_garbage_returns_final():
    step = parse_react("乱七八糟没有格式")
    assert step["type"] == "final"


def test_loop_chat_intent_short_circuits():
    """Planner 判 chat → 单步直答，不进 ReAct。"""
    fake = FakeChat([
        {"content": '{"intent":"chat","sub_tasks":[],"needs_doc":false}', "tool_calls": []},
        {"content": "你好！加油学算法！", "tool_calls": []},
    ])
    ctx = _ctx()
    events = list(run_agent(ctx, "你好", chat_fn=fake, cfg={"base_url": "x", "api_key": "y"}))
    types = [e["type"] for e in events]
    full = "".join(e.get("content", "") for e in events if e["type"] == "delta")
    assert "thinking" in types
    assert "delta" in types and "final_done" in types
    assert "加油" in full
    assert "done" in types


def test_loop_tool_step_then_final():
    fake = FakeChat([
        {"content": '{"intent":"query","sub_tasks":[],"needs_doc":false}', "tool_calls": []},
        {"content": "Thought: 查进度\nAction: get_progress\nArguments: {}", "tool_calls": []},
        {"content": "Thought: 好了\nFinal Answer: 你整体 45%。", "tool_calls": []},
    ])
    ctx = _ctx(graph={"nodes": [], "overview": {"overall_pct": 45, "mastered_points": 0, "total_points": 0}})
    events = list(run_agent(ctx, "我学到哪了", chat_fn=fake, cfg={"base_url": "x", "api_key": "y"}))
    assert any(e["type"] == "tool_call" and e["action"] == "get_progress" for e in events)
    assert any(e["type"] == "tool_result" for e in events)
    assert any(e["type"] == "delta" for e in events)
    assert any(e["type"] == "final_done" for e in events)


def test_loop_max_steps_guard():
    """连续返回工具调用不收敛时，应被最大步数截断，产出降级 final。"""
    tool_only = {"content": "Thought: 继续\nAction: get_progress\nArguments: {}", "tool_calls": []}
    fake = FakeChat([
        {"content": '{"intent":"query","sub_tasks":[],"needs_doc":false}', "tool_calls": []},
        tool_only, tool_only, tool_only, tool_only, tool_only, tool_only,
    ])
    ctx = _ctx()
    events = list(run_agent(ctx, "x", chat_fn=fake, cfg={"base_url": "x", "api_key": "y"}, max_steps=4))
    assert any(e["type"] == "final_answer" for e in events)  # 被截断后仍有 final


# 追加到 tests/test_loop.py
from agent.loop import extract_refs, inject_refs


def test_extract_refs_finds_all_symbols():
    text = "帮我讲讲 #deepfm 和 @dssm论文 以及 $推荐"
    refs = extract_refs(text)
    assert ("#", "deepfm") in refs
    assert ("@", "dssm论文") in refs
    assert ("$", "推荐") in refs


def test_inject_refs_appends_context():
    sys_msg = "你是助手。"
    refs_ctx = "[节点] DeepFM(特征交叉), 依赖: ['fm']"
    out = inject_refs(sys_msg, refs_ctx)
    assert "用户引用了以下内容" in out
    assert refs_ctx in out


def test_inject_refs_empty_returns_unchanged():
    assert inject_refs("原文", "") == "原文"


def test_parse_react_action_with_chinese_comment():
    """Action 后跟中文括号注释,应只取工具名。"""
    step = parse_react("Thought: 查\nAction: get_progress（查进度）\nArguments: {}")
    assert step["type"] == "tool"
    assert step["action"] == "get_progress"


def test_parse_react_action_on_next_line():
    """Action 与工具名之间换行。"""
    step = parse_react("Thought: 查\nAction:\nget_progress\nArguments: {}")
    assert step["action"] == "get_progress"


def test_parse_react_multiline_json_arguments():
    """Arguments 是多行 JSON。"""
    text = ('Thought: x\nAction: add_node\nArguments: {\n  "description": "LightGCN"\n}')
    step = parse_react(text)
    assert step["type"] == "tool"
    assert step["arguments"] == {"description": "LightGCN"}


def test_parse_react_final_answer_multiline():
    """Final Answer 后是多行内容。"""
    step = parse_react("Thought: ok\nFinal Answer: 第一行\n第二行")
    assert step["type"] == "final"
    assert "第一行" in step["answer"] and "第二行" in step["answer"]


def test_loop_history_is_injected_into_messages():
    """前端发来的 history 应前置注入 Executor 的 messages。"""
    fake = FakeChat([
        {"content": '{"intent":"query","sub_tasks":[],"needs_doc":false}', "tool_calls": []},
        {"content": "Thought: 够\nFinal Answer: 你上次问的是 DCN。", "tool_calls": []},
    ])
    ctx = _ctx()
    history = [{"role": "user", "content": "DeepFM 学完了"},
               {"role": "assistant", "content": "建议学 DCN"}]
    list(run_agent(ctx, "那 DCN 之后呢", chat_fn=fake, cfg={"base_url": "x", "api_key": "y"},
                   history=history))
    # 第二次调用(Executor)的 messages 应含 history
    executor_call = fake.calls[1]
    roles_content = [(m["role"], m["content"]) for m in executor_call["messages"]]
    assert ("user", "DeepFM 学完了") in roles_content
    assert ("assistant", "建议学 DCN") in roles_content


def test_loop_history_capped_to_last_8():
    """history 超过 8 条时，只注入最近 8 条，防长上下文累积。"""
    fake = FakeChat([
        {"content": '{"intent":"query","sub_tasks":[],"needs_doc":false}', "tool_calls": []},
        {"content": "Thought: x\nFinal Answer: ok。", "tool_calls": []},
    ])
    ctx = _ctx()
    history = [{"role": "user", "content": f"msg{i}"} for i in range(12)]
    list(run_agent(ctx, "继续", chat_fn=fake, cfg={"base_url": "x", "api_key": "y"},
                   history=history))
    exec_messages = fake.calls[1]["messages"]
    injected = [m["content"] for m in exec_messages
                if m["role"] == "user" and m["content"].startswith("msg")]
    assert len(injected) == 8
    assert injected[0] == "msg4"   # 12 条历史，注入最近 8 条 = msg4..msg11


def test_loop_mutate_intent_injects_tool_directive():
    """mutate 意图时，Executor system prompt 应注入「必须调工具」的强指令。"""
    fake = FakeChat([
        {"content": '{"intent":"mutate","sub_tasks":[],"needs_doc":false}', "tool_calls": []},
        {"content": 'Thought: 加节点\nAction: add_node\nArguments: {"description":"LightGCN"}',
         "tool_calls": []},
        {"content": "Thought: ok\nFinal Answer: 已生成 LightGCN 节点建议。", "tool_calls": []},
    ])
    ctx = _ctx()
    list(run_agent(ctx, "加一个 LightGCN 节点", chat_fn=fake,
                   cfg={"base_url": "x", "api_key": "y"}))
    exec_sys = fake.calls[1]["messages"][0]["content"]
    assert "强制要求" in exec_sys
    assert "add_node" in exec_sys


def test_loop_chat_short_circuit_no_react():
    """chat intent → 单步直答,不进 ReAct(无 tool_call 事件)。"""
    fake = FakeChat([
        {"content": '{"intent":"chat","sub_tasks":[],"needs_doc":false}', "tool_calls": []},
        {"content": "你好！学算法能锻炼思维。", "tool_calls": []},
    ])
    ctx = _ctx()
    events = list(run_agent(ctx, "你好", chat_fn=fake, cfg={"base_url": "x", "api_key": "y"}))
    types = [e["type"] for e in events]
    assert "thinking" in types
    assert "delta" in types and "final_done" in types
    assert not any(e["type"] == "tool_call" for e in events)   # 无工具
    full = "".join(e["content"] for e in events if e["type"] == "delta")
    assert "学算法" in full
    # Executor (chat direct) called once; total calls = Planner(1) + chat direct(1) = 2
    assert len(fake.calls) == 2


def test_loop_chat_direct_streams_when_available():
    """chat 短路在 chat_fn 支持 stream 时应走真流式(逐 token delta)，不回退整段 chunk。"""
    fake = FakeChat([
        {"content": '{"intent":"chat","sub_tasks":[],"needs_doc":false}', "tool_calls": []},
        {"content": "你好呀", "tool_calls": []},
    ])
    ctx = _ctx()
    events = list(run_agent(ctx, "你好", chat_fn=fake, cfg={"base_url": "x", "api_key": "y"}))
    deltas = [e.get("content", "") for e in events if e["type"] == "delta"]
    assert "".join(deltas) == "你好呀"
    # 流式逐 token：3 个中文字符 → 3 个 delta
    assert len([d for d in deltas if d]) == 3
    # 只调 2 次：Planner + chat direct(stream=True)，无第二次非流式回退
    assert len(fake.calls) == 2
    assert fake.calls[1]["stream"] is True


def test_loop_reflect_ok_accepts_draft():
    """Reflect 判 ok=true → 直接接受草稿,不续跑。"""
    fake = FakeChat([
        {"content": '{"intent":"query","sub_tasks":[],"needs_doc":false}', "tool_calls": []},
        {"content": "Thought: x\nFinal Answer: 你整体 45%。", "tool_calls": []},
        {"content": '{"ok": true, "gap": ""}', "tool_calls": []},
    ])
    ctx = _ctx()
    events = list(run_agent(ctx, "进度", chat_fn=fake, cfg={"base_url": "x", "api_key": "y"}))
    full = "".join(e.get("content", "") for e in events if e["type"] == "delta")
    assert "45%" in full


def test_loop_reflect_gap_triggers_rerun():
    """Reflect 判 ok=false → 注入 gap 续跑一轮,再产出更完整答案。"""
    fake = FakeChat([
        {"content": '{"intent":"query","sub_tasks":[],"needs_doc":false}', "tool_calls": []},
        {"content": "Thought: x\nFinal Answer: 建议学 DCN。", "tool_calls": []},          # 草稿
        {"content": '{"ok": false, "gap": "没说为什么推荐 DCN"}', "tool_calls": []},        # Reflect → not ok
        {"content": "Thought: 补充\nFinal Answer: 建议 DCN，因为它承接 DeepFM 的特征交叉。", "tool_calls": []},  # 续跑草稿
        {"content": '{"ok": true, "gap": ""}', "tool_calls": []},                          # 二次 Reflect → ok
    ])
    ctx = _ctx()
    events = list(run_agent(ctx, "下一步学啥,为什么", chat_fn=fake,
                            cfg={"base_url": "x", "api_key": "y"}))
    full = "".join(e.get("content", "") for e in events if e["type"] == "delta")
    assert "特征交叉" in full    # 续跑后的内容被采纳


def test_loop_reflect_capped_accepts_second_draft():
    """Reflect 已用过一次 + 仍不 ok → 接受第二次草稿,不无限续跑。"""
    fake = FakeChat([
        {"content": '{"intent":"query","sub_tasks":[],"needs_doc":false}', "tool_calls": []},
        {"content": "Thought: x\nFinal Answer: 草稿A。", "tool_calls": []},
        {"content": '{"ok": false, "gap": "x"}', "tool_calls": []},
        {"content": "Thought: y\nFinal Answer: 草稿B。", "tool_calls": []},
        {"content": '{"ok": false, "gap": "y"}', "tool_calls": []},   # 二次仍不 ok,但 reflect 已用过
    ])
    ctx = _ctx()
    events = list(run_agent(ctx, "x", chat_fn=fake, cfg={"base_url": "x", "api_key": "y"}))
    assert any(e["type"] == "final_done" for e in events)   # 不卡死
    full = "".join(e.get("content", "") for e in events if e["type"] == "delta")
    assert "草稿B" in full    # 接受第二次草稿


def test_pick_doc_type_by_keyword():
    """含'复习/自测/review'→review,含'周报/本周/总结'→weekly,否则 note。"""
    from agent.loop import _pick_doc_type
    assert _pick_doc_type("帮我生成复习卡") == "review"
    assert _pick_doc_type("帮我自测 DeepFM 结构") == "review"
    assert _pick_doc_type("整理本周学习周报") == "weekly"
    assert _pick_doc_type("做个月度总结") == "weekly"
    assert _pick_doc_type("整理个 DeepFM 笔记") == "note"
    assert _pick_doc_type("讲讲 DCN") == "note"


# ─────────────── P0-#1: 原生 function calling 路径 ───────────────

def test_loop_native_path_uses_openai_tool_messages(monkeypatch):
    """供应商支持原生 function calling 时，全程走原生路径：
    assistant 消息带 tool_calls，工具结果用 role=tool + tool_call_id 回填。
    不再伪造 ReAct 文本 Observation。"""
    from agent import protocol
    monkeypatch.setattr(protocol, "detect_native_support", lambda cfg: True)

    native_tc = [{"id": "call_1", "type": "function",
                  "function": {"name": "get_progress", "arguments": "{}"}}]
    responses = [
        {"content": '{"intent":"query","sub_tasks":[],"needs_doc":false}', "tool_calls": []},
        {"content": "", "tool_calls": native_tc},          # Executor 第1轮：原生工具调用
        {"content": "整体进度 45%。", "tool_calls": []},   # Executor 第2轮：final answer
    ]
    # 自定义 chat_fn：记录每次调用时 messages 的深拷贝（避免 list 引用污染）
    import copy
    snapshots: list[list[dict]] = []
    def capturing_chat(cfg, messages, tools=None, stream=False, response_format=None):
        snapshots.append(copy.deepcopy(messages))
        if responses:
            return responses.pop(0) if len(responses) > 1 else responses[0]
        return {"content": "", "tool_calls": []}

    ctx = _ctx(graph={"nodes": [], "overview": {"overall_pct": 45, "mastered_points": 0, "total_points": 0}})
    events = list(run_agent(ctx, "我学到哪了", chat_fn=capturing_chat,
                            cfg={"base_url": "x", "api_key": "y"}))
    # snapshots[0]=Planner, [1]=Executor第1轮, [2]=Executor第2轮(基于工具结果)
    # 第2轮 messages 应含 role=tool 回填 + assistant 带 tool_calls（OpenAI 标准）
    exec_round2 = snapshots[2]
    roles = [m["role"] for m in exec_round2]
    assert "tool" in roles   # P0-#1 关键：原生路径用 tool role 而非文本 Observation
    asst_with_tc = [m for m in exec_round2
                    if m["role"] == "assistant" and m.get("tool_calls")]
    assert asst_with_tc     # assistant 消息带原始 tool_calls 结构
    tool_msg = [m for m in exec_round2 if m["role"] == "tool"][0]
    assert tool_msg.get("tool_call_id") == "call_1"   # OpenAI 标准：tool_call_id 关联
    # 第1轮 messages 不该有 tool role（还没回填）
    exec_round1 = snapshots[1]
    assert not any(m["role"] == "tool" for m in exec_round1)
    # 事件序列正常
    assert any(e["type"] == "tool_call" and e["action"] == "get_progress" for e in events)
    assert any(e["type"] == "tool_result" for e in events)
    full = "".join(e.get("content", "") for e in events if e["type"] == "delta")
    assert "45%" in full


def test_loop_native_path_executes_all_tool_calls(monkeypatch):
    """原生路径返回多个 tool_calls 时，全部执行（不再只取第一个）。"""
    from agent import protocol
    monkeypatch.setattr(protocol, "detect_native_support", lambda cfg: True)

    multi_tc = [
        {"id": "call_1", "type": "function",
         "function": {"name": "get_progress", "arguments": "{}"}},
        {"id": "call_2", "type": "function",
         "function": {"name": "get_next", "arguments": '{"node_id":"deepfm"}'}},
    ]
    fake = FakeChat([
        {"content": '{"intent":"query","sub_tasks":[],"needs_doc":false}', "tool_calls": []},
        {"content": "", "tool_calls": multi_tc},
        {"content": "查完了。", "tool_calls": []},
    ])
    ctx = _ctx(graph={"nodes": [], "overview": {}})
    events = list(run_agent(ctx, "进度和下一步", chat_fn=fake,
                            cfg={"base_url": "x", "api_key": "y"}))
    actions = [e["action"] for e in events if e["type"] == "tool_call"]
    assert "get_progress" in actions and "get_next" in actions   # 两个都执行了


# ─────────────── P0-#3: ReAct 格式不符重试一次 ───────────────

def test_loop_malformed_react_retries_once(monkeypatch):
    """模型输出既无 Action 也无 Final Answer（格式崩坏）→ 提示重试一次，仍失败才当 final。"""
    fake = FakeChat([
        {"content": '{"intent":"query","sub_tasks":[],"needs_doc":false}', "tool_calls": []},
        {"content": "嗯，让我想想这个问题。"},   # 格式不符，触发重试
        {"content": "Thought: ok\nFinal Answer: 正确答案。"},   # 重试后正常
    ])
    ctx = _ctx()
    events = list(run_agent(ctx, "DeepFM 是什么", chat_fn=fake,
                            cfg={"base_url": "x", "api_key": "y"}))
    # 应有 thinking 事件提示格式重试
    thinking_texts = [e.get("content", "") for e in events if e["type"] == "thinking"]
    assert any("格式" in t for t in thinking_texts)
    full = "".join(e.get("content", "") for e in events if e["type"] == "delta")
    assert "正确答案" in full


# ─────────────── P1-#4: Planner 顺带输出 doc_type ───────────────

def test_loop_planner_doc_type_passes_to_writer():
    """Planner 返回 doc_type 时，Writer 直接用，不调 _pick_doc_type。"""
    fake = FakeChat([
        {"content": '{"intent":"produce","sub_tasks":[],"needs_doc":true,"doc_type":"weekly"}',
         "tool_calls": []},
        {"content": "Thought: ok\nFinal Answer: 已收集素材。", "tool_calls": []},
        {"content": "<title>周报</title><p>本周内容</p>", "tool_calls": []},   # Writer
    ])
    ctx = _ctx()
    events = list(run_agent(ctx, "整理个周报", chat_fn=fake,
                            cfg={"base_url": "x", "api_key": "y"}))
    doc_cards = [e for e in events if e["type"] == "doc_card"]
    assert doc_cards
    assert doc_cards[0]["doc_type"] == "weekly"   # 来自 Planner，非关键词判断


# ─────────────── P0: Writer 素材来自 observations_log（原生路径修复）───────────────

def test_loop_writer_materials_from_observations_in_native_path(monkeypatch):
    """原生 function calling 路径下，Writer 素材必须来自工具结果（observations_log），
    而非从 messages 里反推 role=user 的 "Observation" 文本——旧实现导致原生路径 materials 恒为空。"""
    from agent import protocol
    monkeypatch.setattr(protocol, "detect_native_support", lambda cfg: True)

    native_tc = [{"id": "call_1", "type": "function",
                  "function": {"name": "get_progress", "arguments": "{}"}}]
    responses = [
        {"content": '{"intent":"produce","sub_tasks":[],"needs_doc":true,"doc_type":"note"}',
         "tool_calls": []},
        {"content": "", "tool_calls": native_tc},          # Executor：原生工具调用
        {"content": "Thought: ok\nFinal Answer: 整理好了。", "tool_calls": []},  # final
        {"content": '{"ok": true, "gap": ""}', "tool_calls": []},               # Reflect
        {"content": "<title>笔记</title><p>内容</p>", "tool_calls": []},        # Writer
    ]
    import copy
    snapshots: list[list[dict]] = []
    def capturing_chat(cfg, messages, tools=None, stream=False, response_format=None):
        snapshots.append(copy.deepcopy(messages))
        if responses:
            return responses.pop(0) if len(responses) > 1 else responses[0]
        return {"content": "", "tool_calls": []}

    ctx = _ctx(graph={"nodes": [], "overview": {"overall_pct": 45, "mastered_points": 1, "total_points": 3}})
    events = list(run_agent(ctx, "整理个笔记", chat_fn=capturing_chat,
                            cfg={"base_url": "x", "api_key": "y"}))
    assert any(e["type"] == "doc_card" for e in events)
    # 最后一个调用是 Writer：system prompt 应含工具观察结果（整体掌握度 45%）
    writer_sys = snapshots[-1][0]["content"]
    assert "45%" in writer_sys
