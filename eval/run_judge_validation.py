"""eval/run_judge_validation.py — Reflexion 校验器(judge)诊断质量实测。

【目的】
回应 RESULTS.md 中的口径限制自批评:「校验器与主循环是同一个 LLM,无独立 ground truth;
高续跑率既可能是草稿常遗漏,也可能是校验器倾向判 not ok」。本脚本给 judge 配一个
独立、带人工标注的验证集,量化它「判 not-ok(抓缺陷)」的能力 —— 而不是只在主循环里
数「判 not ok 后续跑」的次数。

【验证集 build_validation_set()】
20 条人工构造:5 个问题模板(见 _build_templates),每模板 4 样本 =
  2 × S+  (label="ok"):observations 从 skill-tree/data/recommendation.json 取真实
                       节点名/任务完成数组装,draft 完整正确且引用真实数字;
  2 × S−  (label="not_ok"):draft 三类缺陷轮换 —— 编造进度数字 / 遗漏关键节点 / 答非所问,
                       每条带 defect_type 字段。
总计 10 ok + 10 not_ok。每样本字段:q / observations / draft / label。

【口径】
  正类 = not_ok(有缺陷)。tally():
    tp = 判 not_ok 且真缺陷;  fp = 判 not_ok 且真 ok
    fn = 判 ok 且真缺陷;       tn = 判 ok 且真 ok
  每样本判 --votes 次(默认 3)多数投票 majority_ok() 定最终判定。

【judge 调用】
  sys.path 指向 skill-tree/backend;from agent.prompts import render_reflect 渲染 prompt;
  urllib 直调 ark /chat/completions(读 eval/config.local.json,最小复制 CallRecorder 模式);
  解析回复中 ok/not-ok(见 parse_verdict,对齐 loop._reflect 的容错语义:解析失败回退 ok)。

【输出】
  judge_metrics(tp,fp,fn,tn) + wilson_ci 准确率 CI + 永远-PASS 退化基线对照行
  (judge_metrics(0,0,10,10)) + JSON 落盘 eval/results/judge_val_{ts}.json(含 limitations)。

【--dry-run】
  FakeJudge 桩(恒 ok / 恒 not_ok / seed 随机 三组)全管道跑通,不调 LLM,
  打印三组混淆矩阵对照表。

【限制(与 JSON limitations 一致)】
  1. 验证集人工构造、非盲标(构造者即标注者),存在利于检出的构造偏差;
  2. S− 为刻意注入的明显缺陷,非自然错误分布 —— 结论仅限定「对明显缺陷的检出能力」;
  3. 正类 prevalence 固定 50%,与线上真实缺陷率不同,PPV 不可外推;
  4. judge 与被测 agent 同源 LLM 时存在自评偏置风险。

用法:
  cd skill-tree/backend
  DATA_ROOT=$(pwd)/../../skill-tree/data python ../../eval/run_judge_validation.py --dry-run
  DATA_ROOT=$(pwd)/../../skill-tree/data python ../../eval/run_judge_validation.py          # 真实 LLM
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

# ── 路径:脚本在 eval/,仓库根在上一级,backend 在 skill-tree/backend ──
EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
BACKEND_DIR = REPO / "skill-tree" / "backend"

# 让 `import agent.prompts` / `import stats` 在任意 CWD 下可用
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(EVAL_DIR))

from agent.prompts import render_reflect  # noqa: E402
from stats import judge_metrics, wilson_ci  # noqa: E402


# ══════════════════════════════════════════════════════════════
# 验证集构建
# ══════════════════════════════════════════════════════════════
def _data_dir() -> Path:
    """skill-tree/data 目录:DATA_ROOT env 优先,否则仓库默认路径。"""
    if os.environ.get("DATA_ROOT"):
        return Path(os.environ["DATA_ROOT"])
    return REPO / "skill-tree" / "data"


def _load_tree(tree_id: str) -> dict:
    """读取 skill-tree/data/<tree_id>.json(真实节点/任务数据)。"""
    p = _data_dir() / f"{tree_id}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _rec_summary() -> dict:
    """从 recommendation.json 提炼构建 S+ observations 所需的真实数字。"""
    rec = _load_tree("recommendation")
    nodes = [n for b in rec["branches"] for n in b["nodes"]]
    done_tasks = sum(t["done"] for n in nodes for t in n["tasks"])
    total_tasks = sum(len(n["tasks"]) for n in nodes)
    done_nodes = "、".join(n["name"] for n in nodes if n["status"] == "done")
    learning = [
        f"{n['name']}（{sum(t['done'] for t in n['tasks'])}/{len(n['tasks'])}）"
        for n in nodes if n["status"] == "learning"
    ]
    branch_lines = [
        f"{b['name']}：{'、'.join(n['name'] for n in b['nodes'])}"
        for b in rec["branches"]
    ]
    # 依赖 deepfm 的节点(DeepFM 解锁的下游)
    unlocked_by_deepfm = [
        n["name"] for n in nodes if "deepfm" in n.get("depends_on", [])
    ]
    return {
        "pct": round(done_tasks / total_tasks * 100),
        "done_tasks": done_tasks,
        "total_tasks": total_tasks,
        "done_nodes": done_nodes,
        "learning": "、".join(learning),
        "branch_lines": branch_lines,
        "branch_names": "、".join(b["name"] for b in rec["branches"]),
        "unlocked_by_deepfm": "、".join(unlocked_by_deepfm),
    }


def _irrelevant_obs() -> str:
    """与 5 个模板问题均无关的 observations(来自 AI Agent 树),供 S− 使用。"""
    return (
        "「AI Agent」树基础层：Python（done，1/2）、PyTorch（learning，2/2）、"
        "机器学习基础（learning，1/2）。LLM 基础分支：Transformer（learning）、"
        "LLM 核心概念（locked）、主流大模型（locked）。"
    )


def _build_templates() -> list[dict]:
    """构造 5 个问题模板,每模板:q / obs_ok / ok[draft×2] / bad[{obs,defect,draft}×2]。"""
    s = _rec_summary()
    pct, d, t = s["pct"], s["done_tasks"], s["total_tasks"]
    learning = s["learning"]
    done_nodes = s["done_nodes"]
    progress_obs = (
        f"「推荐算法」树任务完成率 {pct}%（{d}/{t}）。\n"
        f"已完成节点：{done_nodes}。\n进行中：{learning}。\n"
        f"锁定：DSSM、YouTube DNN、MIND、FM / FFM、DNN 精排基线、DCN / DCN-Mix、xDeepFM 等。"
    )
    deepfm_obs = (
        "「DeepFM」节点（精排 / CTR 分支，learning 1/4）：\n"
        "- 读 DeepFM 论文 (2018)：未完成（arXiv:1703.04247）\n"
        "- 读 DeepCTR-Torch 的 deepfm.py：已完成\n"
        "- 跑通 Criteo demo：未完成\n"
        "- 能讲清 FM + DNN 并联共享 embedding：未完成\n"
        "前置依赖：FM / FFM、DNN 精排基线。"
    )
    direction_obs = (
        f"「推荐算法」方向共 5 个分支：\n"
        + "\n".join(f"- {line}" for line in s["branch_lines"])
        + f"\n整体任务完成率 {pct}%（{d}/{t}）。"
    )

    templates = [
        # 1. 我整体进度怎么样
        {
            "q": "我整体进度怎么样",
            "obs_ok": progress_obs,
            "ok": [
                f"你的整体进度：推荐算法方向任务完成率约 {pct}%（{d}/{t}）[1]。已完成地基节点 {done_nodes}；"
                f"进行中的有 {learning}；其余精排/序列模型节点（DSSM、YouTube DNN、MIND、FM / FFM、"
                f"DCN / DCN-Mix、xDeepFM 等）仍锁定。建议先收尾进行中的四个节点。",
                f"整体来看，推荐算法完成率 {pct}%（{d}/{t}）[1]。{done_nodes} 已 done；"
                f"{learning} 在推进；DSSM、FM / FFM、DCN / DCN-Mix 等模型节点仍锁定。",
            ],
            "bad": [
                {  # 编造进度数字
                    "obs": "",
                    "defect": "fabricated_number",
                    "draft": "你的推荐方向整体掌握度已达 **92%**，Python、PyTorch、机器学习基础全部 100% 完成，"
                             "DeepFM 也只剩 demo 一个任务了，很快就能全通关。",
                },
                {  # 遗漏关键节点(observations 给了真实进度,草稿只提地基)
                    "obs": progress_obs,
                    "defect": "missing_key_node",
                    "draft": "你的整体进度：已完成 Python 和概率统计与数学两个节点，其他节点还在推进中，"
                             "继续加油即可。",
                },
            ],
        },
        # 2. 我学到哪了
        {
            "q": "我学到哪了",
            "obs_ok": progress_obs,
            "ok": [
                f"你目前学到：推荐算法方向完成率 {pct}%[1]。已完成 {done_nodes}；"
                f"正在学的有 {learning}；其余节点未解锁。当前主战场是 DeepFM（1/4），"
                f"其次 PyTorch 和机器学习基础各差 2 个任务。",
                f"最近进度：{done_nodes} 已 done，{learning} 在推进，整体 {pct}%（{d}/{t}）[1]。",
            ],
            "bad": [
                {  # 答非所问
                    "obs": "",
                    "defect": "off_topic",
                    "draft": "学到哪了这个问题建议你用 get_progress 工具查一下，我是你的学习助手，"
                             "可以帮你规划复习节奏和整理笔记。",
                },
                {  # 编造进度数字
                    "obs": _irrelevant_obs(),
                    "defect": "fabricated_number",
                    "draft": "你整体进度已达 88%，推荐方向所有节点都学完了，只剩大模型推荐还没开始，"
                             "随时可以进入复习阶段。",
                },
            ],
        },
        # 3. 下一步学什么
        {
            "q": "下一步学什么",
            "obs_ok": f"进行中：{learning}。\n锁定但依赖 DeepFM：{s['unlocked_by_deepfm']}。",
            "ok": [
                f"下一步建议先收尾 DeepFM（还差读论文、Criteo demo、讲清 FM+DNN 共享 embedding 三项）[1]；"
                f"它解锁 {s['unlocked_by_deepfm']} 等下游节点。可并行补 PyTorch 的 Dataset/DataLoader 与 GPU 训练。",
                f"优先把进行中的 DeepFM 学完，然后是它解锁的精排模型（{s['unlocked_by_deepfm']}）与 DIN；"
                f"同时别落下 PyTorch 剩余两个任务。[1]",
            ],
            "bad": [
                {  # 遗漏关键节点(漏掉 DeepFM 主战场及其解锁链)
                    "obs": f"进行中：{learning}。\n锁定但依赖 DeepFM：{s['unlocked_by_deepfm']}。",
                    "defect": "missing_key_node",
                    "draft": "下一步建议继续学 PyTorch，把 Dataset/DataLoader 和 GPU 训练两个任务补完就行，"
                             "其他不着急。",
                },
                {  # 答非所问
                    "obs": "",
                    "defect": "off_topic",
                    "draft": "推荐系统的精排模型里 FM 用隐向量做二阶交叉，FFM 引入 field 概念，"
                             "你可以去读一下这两篇论文。",
                },
            ],
        },
        # 4. DeepFM 是什么
        {
            "q": "DeepFM 是什么",
            "obs_ok": deepfm_obs,
            "ok": [
                "DeepFM 是华为 2017 年提出的 CTR 预估模型（论文 arXiv:1703.04247）[1]。核心结构是 "
                "**FM 部分**与 **DNN 部分**并联：FM 显式建模二阶特征交叉，DNN 隐式建模高阶交叉，"
                "两部分共享同一套 embedding 层[1]。它结合了 FM 与 Wide & Deep 的思路，"
                "避免了人工构造交叉特征。你已完成 deepfm.py 源码阅读，还差论文、Criteo demo 与"
                "「讲清共享 embedding」三个任务。",
                "DeepFM 把 FM 与 DNN 并联在一起并共享 embedding[1]。FM 负责低阶显式交叉，"
                "DNN 负责高阶隐式交叉，输出相加后做 CTR 预测。出自华为 2017 年论文 (arXiv:1703.04247)，"
                "即 DeepCTR-Torch 里 deepfm.py 实现的模型[1]。",
            ],
            "bad": [
                {  # 编造数字/事实
                    "obs": "",
                    "defect": "fabricated_number",
                    "draft": "DeepFM 是 2019 年 Google 提出的序列推荐模型，主要用于行为序列建模；"
                             "你的 DeepFM 论文、demo、讲解任务已全部完成（4/4）。",
                },
                {  # 遗漏关键点(FM+DNN 并联共享 embedding 是核心,被漏掉)
                    "obs": deepfm_obs,
                    "defect": "missing_key_node",
                    "draft": "DeepFM 是一个 CTR 模型，主要由一个 DNN 部分组成，用于做特征交叉[1]。"
                             "你已完成 deepfm.py 阅读，接下来读论文即可。",
                },
            ],
        },
        # 5. 推荐方向有哪些内容
        {
            "q": "推荐方向有哪些内容",
            "obs_ok": direction_obs,
            "ok": [
                f"推荐方向包含 5 个分支[1]：{s['branch_names']}。"
                f"基础层有 Python、PyTorch、概率统计与数学、机器学习基础；"
                f"召回有 DSSM、YouTube DNN、MIND；精排 / CTR 有 FM / FFM、DeepFM、DCN / DCN-Mix、xDeepFM；"
                f"序列推荐有 DIN、DIEN；大模型推荐有生成式推荐 (HSTU) 等。整体完成率约 {pct}%。",
                f"一共 5 个分支：{s['branch_names']}[1]。基础层已学完 Python 与概率统计；"
                f"召回和精排还有 DeepFM 等在推进；序列推荐与大模型推荐尚锁定。",
            ],
            "bad": [
                {  # 答非所问
                    "obs": "",
                    "defect": "off_topic",
                    "draft": "你的学习周报可以这样写：本周完成 Python 语法复习、概率统计刷题，"
                             "下周计划啃 Transformer 注意力机制，卡点是 GPU 不够用。要我现在生成周报吗？",
                },
                {  # 编造数字(分支数/完成度均错)
                    "obs": _irrelevant_obs(),
                    "defect": "fabricated_number",
                    "draft": "推荐方向一共 3 个分支：基础层、召回、精排。基础层已全部完成（100%），"
                             "召回和精排里 DSSM、DCN、xDeepFM、DIN 都已经学完。",
                },
            ],
        },
    ]
    return templates


def build_validation_set() -> list[dict]:
    """构造 20 条验证集:每模板 2×S+ + 2×S−。字段 q/observations/draft/label,S− 带 defect_type。"""
    samples: list[dict] = []
    for tpl in _build_templates():
        q = tpl["q"]
        obs_ok = tpl["obs_ok"]
        for draft in tpl["ok"]:
            samples.append({"q": q, "observations": obs_ok, "draft": draft, "label": "ok"})
        for bad in tpl["bad"]:
            samples.append({
                "q": q,
                "observations": bad["obs"],
                "draft": bad["draft"],
                "label": "not_ok",
                "defect_type": bad["defect"],
            })
    return samples


# ══════════════════════════════════════════════════════════════
# 判定 / 统计
# ══════════════════════════════════════════════════════════════
def majority_ok(votes: list[bool]) -> bool:
    """多数投票:>= 半数判 ok → True(平票判 ok,保守避免误触发续跑)。"""
    if not votes:
        raise ValueError("votes 不能为空")
    return 2 * sum(1 for v in votes if v) >= len(votes)


def _try_json(text: str) -> dict | None:
    """整段 json.loads,失败返回 None;非 dict 也返回 None。"""
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _scan_json_object(text: str, start: int) -> dict | None:
    """从 start 处 '{' 起扫描到配对的 '}',返回解析出的 dict(尊重字符串转义)。"""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return _try_json(text[start:i + 1])
    return None


def parse_verdict(text: str) -> bool:
    """解析 judge 回复 → 是否 ok(True=放行)。对齐 loop._reflect 容错:解析失败回退 ok=True。

    规则(由 render_reflect 的 SYS_REFLECT 约定输出 {"ok": true/false, "gap": ...}):
      1) 整段 json.loads 成功 → obj.get("ok", True)
      2) 否则提取首个配对 JSON 对象解析
      3) 否则正则 '"ok": true/false'
      4) 否则关键词兜底("不 ok"/false → 判 not_ok)
      5) 全部失败 → True(与生产一致:不阻塞主流程)
    """
    if not text:
        return True
    t = text.strip()
    obj = _try_json(t)
    if obj is not None:
        return bool(obj.get("ok", True))
    start = t.find("{")
    if start != -1:
        obj = _scan_json_object(t, start)
        if obj is not None:
            return bool(obj.get("ok", True))
    m = re.search(r'"ok"\s*:\s*(true|false)', t, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"
    if "不 ok" in t or "不ok" in t or "false" in t.lower():
        return False
    return True


def tally(labels: list[str], preds: list[bool]) -> dict:
    """混淆矩阵计数。labels: 真值 'ok'/'not_ok';preds: 判 not_ok 的 bool。"""
    if len(labels) != len(preds):
        raise ValueError(f"labels({len(labels)}) 与 preds({len(preds)}) 长度不一致")
    tp = fp = fn = tn = 0
    for label, pred in zip(labels, preds):
        if pred and label == "not_ok":
            tp += 1
        elif pred and label == "ok":
            fp += 1
        elif (not pred) and label == "not_ok":
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ══════════════════════════════════════════════════════════════
# judge 抽象:真实 LLM(ark) / 桩(FakeJudge)
# ══════════════════════════════════════════════════════════════
class ArkJudge:
    """真实 LLM judge:urllib 直调 ark /chat/completions,最小复制 run_eval.CallRecorder 模式。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _chat(self, prompt: str) -> str:
        base = self.cfg["base_url"].rstrip("/")
        url = f"{base}/chat/completions"
        body = {
            "model": self.cfg.get("model") or "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "请输出校验 JSON。"},
            ],
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), method="POST"
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.cfg['api_key']}")
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        msg = data["choices"][0]["message"]
        return msg.get("content") or ""

    def judge_ok(self, q: str, obs: str, draft: str) -> bool:
        prompt = render_reflect(question=q, observations=obs, draft=draft)
        return parse_verdict(self._chat(prompt))


class FakeJudge:
    """桩 judge:不调 LLM,用于 --dry-run 全管道验证。mode ∈ always_ok/always_not_ok/random。"""

    def __init__(self, mode: str, seed: int = 42):
        self.mode = mode
        self.rng = random.Random(seed)

    def judge_ok(self, q: str, obs: str, draft: str) -> bool:
        if self.mode == "always_ok":
            return True
        if self.mode == "always_not_ok":
            return False
        return self.rng.random() < 0.5  # random


def evaluate_set(samples: list[dict], judge, votes: int = 3) -> tuple[dict, list[dict]]:
    """对验证集逐样本多判投票,返回 (tally 四格, 逐样本记录)。"""
    labels: list[str] = []
    preds: list[bool] = []
    records: list[dict] = []
    for s in samples:
        votes_list = [
            judge.judge_ok(s["q"], s["observations"], s["draft"])
            for _ in range(votes)
        ]
        ok = majority_ok(votes_list)
        judged_not_ok = not ok
        labels.append(s["label"])
        preds.append(judged_not_ok)
        records.append({
            "q": s["q"],
            "label": s["label"],
            "defect_type": s.get("defect_type"),
            "votes_ok": votes_list,
            "majority_ok": ok,
        })
    return tally(labels, preds), records


# ══════════════════════════════════════════════════════════════
# 输出
# ══════════════════════════════════════════════════════════════
LIMITATIONS = [
    "验证集人工构造、非盲标(构造者即标注者),存在利于检出的构造偏差",
    "S− 为刻意注入的明显缺陷,非自然错误分布 —— 结论仅限定『对明显缺陷的检出能力』",
    "正类(not_ok)prevalence 固定 50%,与线上真实缺陷率不同,PPV/NPV 不可外推",
    "judge 与被测 agent 同源 LLM 时存在自评偏置风险(本脚本仅提供独立 ground truth,不消除该偏置)",
]


def _fmt_row(name: str, t: dict, metrics: dict) -> str:
    return (
        f"{name:<26} {t['tp']:>3} {t['fp']:>3} {t['fn']:>3} {t['tn']:>3}"
        f"  {metrics['accuracy']:>5.2f}  {metrics['tpr'] if metrics['tpr'] is not None else '  -':>5}"
        f"  {metrics['tnr'] if metrics['tnr'] is not None else '  -':>5}"
        f"  {metrics['kappa']:>5.2f}"
    )


def _run_dry(samples: list[dict], votes: int) -> None:
    """--dry-run:三组 FakeJudge 桩跑通全管道,打印对照表,不调 LLM。"""
    groups = [
        ("恒ok(永远 PASS)", FakeJudge("always_ok")),
        ("恒not_ok(永远 FAIL)", FakeJudge("always_not_ok")),
        ("随机(seed=42)", FakeJudge("random", seed=42)),
    ]
    header = f"{'组别':<28} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}  {'准确率':>5}  {'TPR':>5}  {'TNR':>5}  {'κ':>5}"
    print("=" * 78)
    print("Reflexion 校验器诊断质量 — dry-run(不调 LLM,验证管道)")
    print(f"验证集: {len(samples)} 条 (ok=10 / not_ok=10),每样本 {votes} 次投票多数决")
    print("=" * 78)
    print("混淆矩阵(正类 = not_ok / 判缺陷)")
    print(header)

    results = []
    for name, judge in groups:
        t, records = evaluate_set(samples, judge, votes)
        m = judge_metrics(t["tp"], t["fp"], t["fn"], t["tn"])
        print(_fmt_row(name, t, m))
        results.append({"name": name, **t, "metrics": m, "records": records})

    # 永远-PASS 退化基线对照行
    base_t = {"tp": 0, "fp": 0, "fn": 10, "tn": 10}
    base_m = judge_metrics(0, 0, 10, 10)
    print(_fmt_row("退化基线(永远 PASS)", base_t, base_m))
    print("-" * 78)
    print("解读: 恒ok 组应与退化基线一致(tp=0/fp=0/fn=10/tn=10,准确率=0.5 但 TPR=0 —— 完全抓不住缺陷)。")
    print("       真实 judge 的 TPR 若显著高于基线,才说明它有独立于『瞎判 ok』的诊断能力。")

    _save_json(mode="dry-run", votes=votes, groups=results, baseline=base_m)


def _run_real(samples: list[dict], votes: int) -> None:
    """真实 LLM 跑(后续任务):读 eval/config.local.json,直调 ark。"""
    config_path = EVAL_DIR / "config.local.json"
    if not config_path.exists():
        print("[ERROR] 找不到 eval/config.local.json。请复制 config.local.json.example 并填入真实配置。")
        sys.exit(1)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))["chat"]
    judge = ArkJudge(cfg)
    print("=" * 78)
    print(f"Reflexion 校验器诊断质量 — 真实 LLM(model={cfg.get('model')})")
    print(f"验证集: {len(samples)} 条,每样本 {votes} 次投票多数决")
    print("=" * 78)
    t, records = evaluate_set(samples, judge, votes)
    m = judge_metrics(t["tp"], t["fp"], t["fn"], t["tn"])
    k = t["tp"] + t["tn"]
    n = len(samples)
    lo, hi = wilson_ci(k, n)
    print(_fmt_row("真实 judge", t, m))
    print(f"\n准确率 Wilson 95% CI: {lo:.3f} ~ {hi:.3f} (k={k}/{n})")
    base_m = judge_metrics(0, 0, 10, 10)
    print(_fmt_row("退化基线(永远 PASS)", {"tp": 0, "fp": 0, "fn": 10, "tn": 10}, base_m))
    _save_json(
        mode="real", votes=votes,
        groups=[{"name": "real", **t, "metrics": m, "accuracy_ci": [round(lo, 4), round(hi, 4)],
                 "records": records}],
        baseline=base_m, model=cfg.get("model"),
    )


def _save_json(mode: str, votes: int, groups: list[dict], baseline: dict, model: str | None = None) -> None:
    """结果落盘 eval/results/judge_val_{ts}.json(目录已 gitignore)。"""
    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"judge_val_{ts}.json"
    payload = {
        "mode": mode,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "n_samples": 20,
        "votes_per_sample": votes,
        "groups": groups,
        "baseline_always_pass": baseline,
        "limitations": LIMITATIONS,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] 结果已保存: {out_path}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Reflexion 校验器(judge)诊断质量实测:混淆矩阵/TPR/TNR/κ + 退化基线对照。"
    )
    ap.add_argument("--dry-run", action="store_true", help="用 FakeJudge 桩跑通全管道,不调 LLM")
    ap.add_argument("--votes", type=int, default=3, help="每样本判定次数(多数投票),默认 3")
    args = ap.parse_args(argv)

    samples = build_validation_set()
    if args.dry_run:
        _run_dry(samples, args.votes)
    else:
        _run_real(samples, args.votes)


if __name__ == "__main__":
    main()
