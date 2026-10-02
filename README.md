# 自适应学习 Agent

一个带**评估迭代闭环**的 AI Agent 系统：把学习路径规划、进度追踪、知识检索、文档产出串成「意图分类 → 执行 → 产出」三层流程，用 Reflexion 自校验保障可靠性，用 pytest + 黄金用例集把 Prompt 工程变成可测工程。

> 📄 **[系统设计文档（飞书）](https://my.feishu.cn/docx/M6FGdMITtoZenZxS8X5cRjXXnIQ)** — Agent 三层流程 / RAG 混合检索 / Reflexion 自校验 / 飞书文档产出闭环 / 评估方法论。

## ✨ 核心能力

- **分层 Agent**：意图分类（chat/query/mutate/produce）→ Executor(ReAct) 工具循环 → Writer 文档产出，最大 6 步防发散
- **RAG 混合检索**：源码 AST 切 chunk 建向量索引，向量 + 图谱 + 论文三路融合，带引用编号
- **Reflexion 自校验**：ReAct 草稿后插入自校验发现遗漏续跑（封顶 1 轮防卡死），实测续跑率 87.5%（query/produce 路径）
- **工具协议适配**：混合协议（原生 function calling 优先、指令式 ReAct 回退），一套代码适配多家 OpenAI 兼容供应商
- **飞书产出闭环**：Writer 多模板差异化产出 → 飞书文档 → 知识库归档，端到端可演示
- **评估迭代闭环**：134 个 pytest 用例 + 黄金用例集构成回归网，消融/参数扫描/Top-k 与 MRR 指标

## 📊 实测指标

> 实测数据详见 [`eval/RESULTS.md`](eval/RESULTS.md)，可复现。

**Agent Loop（13 条黄金用例）**

| 指标 | 实测值 |
|------|--------|
| 意图分类准确率 | 13/13（chat/query/mutate/produce 四类全对）|
| Reflexion 续跑率 | 87.5%（query/produce 到达 final 的用例 7/8）|
| 任务完成率 | 84.6%（11/13）|

**RAG 召回（15 条查询集，源码 AST 索引）**

| 指标 | 实测值 |
|------|--------|
| Top-5 召回命中率 | 93.3%（14/15）|
| Top-1 召回命中率 | 66.7%（10/15）|
| MRR | 0.80 |

## 🏗️ 技术栈

**React + FastAPI + docker-compose 全栈应用**

| 层 | 技术 | 说明 |
|----|------|------|
| 前端 | React 18 + TypeScript + Vite | 单页应用，玉青宝石工坊美学，DAG 知识图谱 |
| 后端 | FastAPI (Python) | 单用户本地应用 + AI 生成引擎，零三方依赖(ai/layout/progress 纯标准库) |
| Agent | 意图分类→Executor→Writer | ReAct 循环 + Reflexion 自校验 + 混合协议适配 |
| RAG | 向量 + 图谱 + 论文 | 源码 AST 切 chunk，BM25 + 余弦相似度三路融合 |
| 数据 | JSON 文件 | 单用户 `data/` 目录（桌面应用复制到 `~/.skill-tree/data/`），无数据库 |
| 部署 | docker-compose / Tauri | 一键起 frontend(:5173) + backend(:8001)，或打包桌面应用 |

## 🚀 快速开始

### docker-compose（推荐）
```bash
cd skill-tree
docker-compose up          # 前端 http://localhost:5173  后端 http://localhost:8000
```

### 本地开发
```bash
# 终端 1：后端
cd skill-tree/backend
pip install -r requirements.txt
python -m uvicorn main:app --port 8000 --reload

# 终端 2：前端
cd skill-tree/frontend
npm install
npm run dev                # http://localhost:5173（代理 /api → :8000）
```

首次进入：**新用户自动跳初始化**（建用户 → 配模型 → 贴 JD → AI 生成技能树）。已有用户从侧栏底部「⚙️设置」进入。

**构建 RAG 知识库索引**（让 Agent 能检索你的开源项目源码；配好 LLM 后）：

```bash
curl -X POST http://localhost:8000/api/rag/build-index
curl http://localhost:8000/api/rag/status   # 查索引状态
```

装 numpy 可加速向量检索（可选，不装也能跑，退化为纯标准库）：

```bash
pip install numpy
```

## 🔬 评估体系

把 Prompt 工程变成可测工程——所有 Agent 行为都有对应的评估手段：

- **Agent loop 评估**：13 条黄金用例覆盖 chat/query/mutate/produce 四类意图，跑通即回归
- **RAG 召回评估**：15 条查询集 + Top-k 敏感性扫描 + 分数阈值实验
- **统计口径**：`--repeats k` 多次重复出双口径报告——pass@k（能力，无偏估计）+ pass^k（可靠性，直接频率）+ Wilson 95% CI；judge 验证脚本测 Reflexion 校验器的混淆矩阵 / TPR / TNR / κ（纯标准库 `eval/stats.py`，方法论见[评测指南第九章](https://my.feishu.cn/docx/BDfMdHc11oltMDxr0D6cSYzOnYf)）
- **复现**：`cd skill-tree/backend && python ../../eval/run_eval.py --repeats 5`，原始数据落 `eval/results/`
- 完整结果见 [`eval/RESULTS.md`](eval/RESULTS.md)（k=5 实测：宏 pass@5=1.000 / pass^5=0.615，judge κ=0.60）

## AI Agent 详解

右下角 ✦ 悬浮按钮打开 **AI 学习助手**——它不是单轮生成器，而是一个会调工具、能检索知识、可产文档的 Agent：

- **工具调用**：自主决定读图谱进度 / 查节点 / 检索知识 / 加节点，混合协议（原生 function calling 优先，指令式回退）
- **RAG 知识检索**：把 `../projects` 开源项目源码建成向量索引（AST 切 chunk），混合检索（向量 + 图谱 + 论文），带引用编号
- **ReAct 推理循环**：意图分类分流 → Executor 工具循环 → Writer 文档产出，最大 6 步防发散
- **飞书文档产出**：一键把学习笔记 / 复习卡 / 周报发到飞书

**飞书文档产出配置**（首次使用需登录）：

```bash
lark-cli auth login        # 按提示完成飞书授权
lark-cli --version         # 确认已安装（≥1.0.60）
```

## 📂 目录结构

```
├── skill-tree/              ← 【主体】技能树全栈应用 + AI Agent
│   ├── backend/               FastAPI：API + Agent 内核(agent/) + RAG(rag/) + layout/progress(图谱)
│   ├── frontend/              React+TS：DAG 图谱 + 多会话 AI 对话 + 侧栏四板块
│   ├── desktop/               Tauri 桌面 shell(打包成 .msi/.dmg)
│   ├── data/                  单用户数据：方向树 JSON + profile + 成就 + llm_config
│   └── scripts/               桌面打包脚本
├── resume/                  ← 【果实】模块化 LaTeX 简历(本地,含个人信息,已 gitignore)
│   ├── shared/                素材单一数据源(personal/education/experience/skills)
│   ├── profiles/              岗位 profile(推荐/搜索/广告/agent)
│   ├── templates/             7 套 LaTeX 模板
│   └── build/                 编译脚本 + PDF 输出
├── eval/                    ← Agent + RAG 实测评估（黄金用例 / 召回 / 参数扫描）
├── docs/                    学习笔记 + 设计 spec + 实现计划
│
../projects/                 ← 【果实】搜广推开源项目(已移出本仓库到父目录，与 Resume 同级)
```

## 🌱 学习闭环验证（Agent 吃自己的狗粮）

LearnLoop 的设计目标是解决「学习路径无规划、进度无追踪、产出无沉淀」——本节以作者本人的求职学习为长期真实负载，验证这一闭环（仓库：[interview-prep](https://github.com/Zjrua/interview-prep)，私有）：

| 学习线 | 产出 | 规模 |
|--------|------|------|
| D2L 深度学习 | 教材级 notebook（对齐《动手学深度学习》d2l-zh 原版叙事，实测输出入库） | 31 篇 |
| Agent 技术 | 同风格教材（对话协议/Function Calling/ReAct/MCP/记忆/规划/RAG/评测/毕业项目） | 9 章 |
| 金融量化 | 手写回测引擎与因子模型教材（真实A股数据） | 6 章 |
| 算法 | LeetCode Hot 100（Python） | 17 题 |

- 学习进度由双活文档（`docs/d2l-progress.md`）追踪，每日定时检查点（cron）驱动，费曼式口头自测留痕
- 教材产出遵循统一文风规范（`d2l-study/REWRITE_STYLE.md`），盲审通过后入库

## 🧩 技能沉淀（Skills）

把 Agent 使用中反复出现的工作流封装为可复用 Skill（Markdown 过程性知识，供 Agent 按需加载），已沉淀 4 个：

| Skill | 用途 |
|-------|------|
| `structured-self-study` | 系统化自学陪伴：精读→合卷输出→每日测验→进度档案，防"观光客式学习" |
| `notebook-textbook-generation` | d2l 风格教材 notebook 编写流水线（含盲审验证） |
| `chinese-city-policy-research` | 中国城市社保/公积金/房价参数核查（服务器受限网络的可靠检索路径） |
| `skills/resume-editing` | 简历素材维护规范（数字必须可验证、素材与呈现分离） |

## 核心功能

- **单画布 DAG 知识图谱**：所有方向合并去重成一张图，基础在上向下生长，贝塞尔曲线连线
- **清单驱动掌握度**：每个知识点配验收任务（能默写/讲清/手算），验收全勾才算掌握
- **悬停高亮学习路径**：悬停节点或方向标签 → 高亮上下游路径，其余淡化
- **AI 生成技能树**：支持 DeepSeek/MiMo/智谱/Qwen/Moonshot/自定义，输入 JD 自动生成
- **单用户本地应用**：数据存本地 JSON 目录，API Key 本地存储（gitignore 排除），不打云部署
- **成就系统**：铜/银/金三级，按掌握度/分支完成自动解锁
- **侧栏 SPA**：技能树 / 个人信息 / 简历模板 / 果实展示 四板块 + 底部设置

## 设计原则

- **改数据只动 JSON**：加技能/方向/成就只改 `data/` 对应 JSON，浏览器刷新即生效
- **零依赖后端**：ai/layout/progress 纯 Python 标准库，Windows 直接能跑
- **不填虚构技能**：节点默认 locked，由用户逐步推进
- **平滑演进**：用户抽象层预留鉴权/加密钩子，将来加登录不改业务代码

## 许可

技能树系统（`skill-tree/`）为本人原创；各模板和开源项目遵循其原始许可证。
