# 🌳 技能树系统（Skill Tree）

> 本仓库的**主体**。React + FastAPI 全栈应用 + AI Agent：可视化知识图谱 + 清单驱动掌握度 + 成就反馈 + AI 陪伴学习。
> 简历（`resume/`）和开源项目（`projects/`）是这棵树结出的**果实**，被节点按路径引用。
> 可作本地 web 跑，也可打包成 [Tauri 桌面应用](desktop/README.md)（Windows .msi / macOS .dmg）。

## 技术架构

```
skill-tree/
├── backend/                    FastAPI (Python)
│   ├── main.py                 API: /api/graph, /api/task, /api/agent/chat(SSE), /api/ai/*, /api/rag/*
│   ├── agent/                  AI Agent 内核(意图分类→Executor→Writer + Reflexion)
│   ├── rag/                    混合检索(源码 AST + 论文 + 图谱/简历)
│   ├── larkpub.py              飞书文档产出(lark-cli subprocess)
│   ├── layout.py / progress.py DAG 布局 + 掌握度纯函数
│   ├── entry.py                PyInstaller sidecar 入口(桌面打包)
│   └── requirements.txt        fastapi, uvicorn, pydantic
├── frontend/                   React + TypeScript (Vite)
│   └── src/                    SkillTree 画布 + 多会话 AI 对话(AgentChat) + 侧栏四板块
├── desktop/                    Tauri 桌面 shell(打包成安装包,见 desktop/README.md)
├── data/         ✏️ 数据源（dev 用；桌面应用复制到 ~/.skill-tree/data）
└── docker-compose.yml          dev 一键起 backend(8000) + frontend(5173)
```

## 快速开始

### 方式一：docker-compose（推荐）
```bash
cd skill-tree
docker-compose up              # 前端 http://localhost:5173  后端 http://localhost:8000
```

### 方式二：本地分别启动（开发热更新）
```bash
# 终端 1：后端
cd skill-tree/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 终端 2：前端
cd skill-tree/frontend
npm install
npm run dev                    # 自动代理 /api → :8000
```

## 飞书文档归档(可选)

Agent 产出的学习笔记/复习卡/周报可发布并归档到飞书知识库:

1. 安装并登录 lark-cli:`lark-cli auth login`
2. 在「设置」→「飞书知识库归档」选择一个 wiki 空间(或留空,仅生成单篇文档)
3. 对话里说「整理个 X 的笔记」「生成复习卡」「本周周报」,Agent 产出文档卡片,点「写飞书」即归档

> 文档通过 `docs +create` 生成后用 `wiki +move` 移入选定知识库空间;未配置空间时仅生成单篇飞书文档。

## 桌面应用打包

这是个 personal 应用（依赖本机 lark-cli / 本地源码 RAG，且数据含 api_key），**封装成桌面应用比部署网站更合适**。

```bash
bash scripts/build-desktop.sh      # 一键:前端 build → PyInstaller 冻结后端 → Tauri 打包
```

产出 `desktop/src-tauri/target/release/bundle/` 下的 `.msi`(Windows) / `.dmg`(macOS)。
用户双击安装即用，数据存 `~/.skill-tree/`（升级/卸载不丢）。详见 [desktop/README.md](desktop/README.md)。

## AI Agent

Agent 不只是"加节点"——它有记忆、会短路、能自我校验、产出可沉淀：

- **多轮记忆**：前端发最近对话历史，后端无状态注入
- **意图短路**：闲聊单步直答，不滥用 ReAct
- **提案闭环**：改图谱走"生成→预览→确认"，防误改
- **Reflexion**：ReAct 答完自我校验，遗漏续跑（比基础 ReAct 高一阶）
- **Prompt 工程**：三套分层 prompt 带 few-shot + 回归测试
- **飞书产出**：笔记/复习卡/周报归档到知识库

设计文档：[`docs/design/2026-07-02-agent-depth-design.md`](../docs/design/2026-07-02-agent-depth-design.md)

## 数据源（`data/`）

开发期数据源；桌面应用首次启动会把这里复制到 `~/.skill-tree/data/` 作为 seed（排除 chat_history）。

```
data/
├── recommendation.json   推荐方向节点（含验收子任务）
├── search.json           搜索方向节点
├── ads.json              广告方向节点
├── agent.json            AI Agent 方向节点
├── profile.json          个人信息（⚠️ 须与 resume/shared/*.tex 同步）
├── achievements.json     成就定义
├── llm_config.json       大模型配置（gitignore，含 api_key）
└── lark_config.json      飞书归档配置（wiki_space_id）
```

**知识点与验收**：每个学习任务可带 `verify[]` 子任务。
- 有验收的知识点：**勾完验收才算掌握**（学习任务勾选框置灰，仅清单提示）
- 无验收的知识点：勾学习任务本身即掌握
- 节点点亮(done) = 所有知识点都掌握

```jsonc
{"id":"nn","title":"nn.Module 搭模型 / 训练循环","done":true,
 "verify":[
   {"id":"v1","title":"能默写一个完整训练循环","done":false}
 ]}
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/graph` | 合并去重 + 布局 + 掌握度 + 成就 + 总览 |
| PATCH | `/api/task` | 勾选/取消任务（写回 JSON），返回更新后的图 |
| GET | `/api/profile` | 个人信息 |
| GET | `/api/templates` | 简历模板列表 |
| GET | `/api/fruits` | 果实（简历成品） |
| `/projects/*`, `/resume/*` | 静态托管 | 源码链接、PDF 打开 |

## 泛化与设计

- **目录驱动**：`data/` 丢一个方向 JSON 即自动并入图谱，无需改代码
- **单画布 DAG**：所有方向合并去重成一张图，跨方向同名节点(python/pytorch 等)只画一次
- **基础在上**：depth 0 的根节点在画布顶部，向下生长
- **悬停高亮路径**：鼠标悬停节点 → 上游祖先 + 下游后代高亮，其余淡化
- **展开避让**：点开节点详情 → 下方节点下推，**节点和连线同源于一份 layout → 永不错位**

详见 [CLAUDE.md](../CLAUDE.md)。
