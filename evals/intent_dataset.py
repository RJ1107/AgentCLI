"""Labelled requests for experiment 6 (routing a request to react, /plan or /team).

Each entry: (request, level, mode, note). Labels follow the classifier's own definitions:
  simple    a question, or a change of one or two steps              -> react
  moderate  several steps in one area                                 -> react
  complex   many steps or files, design work, or a large change       -> plan
            ... and it splits into independent parts to run at once   -> team
Several are written to trip keyword rules: short but large, long but simple, scope words in
a plain question, "then" in a two-step change, "each" in work that must run in order.
"""

INTENTS = [
    # ---- simple: questions
    ("这个项目用的是什么 web 框架？", "simple", "react", "scope word 项目 in a question"),
    ("pyproject.toml 里的 requires-python 是多少", "simple", "react", ""),
    ("解释一下 src/agentcli/agent/query.py 里 _arguments_complete 是干什么的", "simple", "react", ""),
    ("为什么 pytest 说找不到 conftest 里的 fixture？", "simple", "react", ""),
    ("git 怎么撤销上一次 commit 但保留改动", "simple", "react", ""),
    ("整个项目有多少个测试文件？", "simple", "react", "scope words, trivial count"),
    ("What does the --read-only flag of mcp serve do?", "simple", "react", ""),
    ("why is the sandbox image 411 MB?", "simple", "react", ""),
    ("这个架构图里 orchestrator 和 worker 是什么关系", "simple", "react", "架构 in a question"),
    ("README 里写的默认模型是哪个", "simple", "react", ""),
    (
        "运行测试报了下面这个错，是什么原因？\n\nTraceback (most recent call last):\n"
        '  File "tests/test_sessions.py", line 88, in test_resume_latest\n'
        "    session = store.latest(project)\n"
        '  File "src/agentcli/session/store.py", line 142, in latest\n'
        "    metas = sorted(self._metas(project), key=lambda m: m.updated, reverse=True)\n"
        '  File "src/agentcli/session/store.py", line 97, in _metas\n'
        "    data = json.loads(path.read_text(encoding=\"utf-8\"))\n"
        "json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)\n"
        "然后第二次跑又好了，我没改任何东西。",
        "simple",
        "react",
        "long pasted log, one question; contains 然后",
    ),
    ("how do I set AGENTCLI_HOME on Windows", "simple", "react", ""),
    ("所有的 memory 都存在哪个目录", "simple", "react", "所有 in a question"),
    ("什么是 RRF", "simple", "react", ""),
    # ---- simple: one or two step changes
    ("把 README 里的版本号改成 1.2.0", "simple", "react", ""),
    ("给 DomainPolicy.remember 加一行注释说明为什么排序", "simple", "react", ""),
    ("把 max_concurrent_read 的默认值从 4 改成 6", "simple", "react", ""),
    ("rename the helper _clip_output to _truncate_output and update its one caller", "simple", "react", ""),
    ("先把 .gitignore 里加上 evals/_home/，然后提交", "simple", "react", "先...然后 but two tiny steps"),
    ("fix the typo 'recieve' in docs/usage.md", "simple", "react", ""),
    ("删掉 config.py 里没用到的 import", "simple", "react", ""),
    ("在 pricing.py 里把 glm-5.3 的输入价格改成 0.6", "simple", "react", ""),
    ("跑一下 tests/test_memory.py 看看过不过", "simple", "react", ""),
    ("add a --version flag that prints agentcli.__version__", "simple", "react", ""),
    ("把这个函数的返回类型标注补上", "simple", "react", ""),
    ("给 web_fetch 的默认白名单加上 docs.python.org", "simple", "react", ""),
    # ---- moderate: several steps, one area
    ("给 /sessions 命令加一个 --limit 参数，并补上对应的测试", "moderate", "react", ""),
    ("session 列表现在按创建时间排序，改成按最后更新时间，picker 和 /sessions 都要改", "moderate", "react", ""),
    ("web_fetch 遇到 429 时现在直接报错，加上重试和 Retry-After 支持，再写个测试", "moderate", "react", ""),
    ("the memory index sometimes lists a deleted memory; find why and fix it, with a regression test", "moderate", "react", ""),
    ("把 save_memory 的 keywords 参数改成可选，没传的时候从标题里自动生成几个", "moderate", "react", ""),
    ("排查一下为什么 /resume 选中会话后提示符没有刷新，修掉", "moderate", "react", ""),
    ("给沙箱加一个 /sandbox status 子命令，显示镜像是否存在、容器是否在跑", "moderate", "react", ""),
    ("make the approval prompt show the full path for write_file and edit_file, and test it", "moderate", "react", ""),
    ("把 L3 截断的头尾比例做成配置项，默认还是 60/40，文档也更新一下", "moderate", "react", ""),
    ("统计每次运行调用了几个 skill，加到运行结束的那行总结里", "moderate", "react", ""),
    ("先读一下 context/manager.py，然后给 _plan 补几个边界情况的测试", "moderate", "react", "先...然后, one area"),
    ("the retry delay ignores Retry-After when it is an HTTP date; parse both forms", "moderate", "react", ""),
    ("每个工具的错误信息格式不统一，统一成 'Tool X failed: 原因' 这种", "moderate", "react", "每个 but small, one pass"),
    ("加一个 /cost 命令，显示当前会话累计花费", "moderate", "react", ""),
    ("MCP 工具列表缓存过期时间现在写死是 1 天，改成可配置并在 /mcp 里显示剩余时间", "moderate", "react", ""),
    ("write a script that converts the old SQLite memory store to the new file layout, with a dry-run flag", "moderate", "react", ""),
    # ---- complex: plan (sequential, design)
    ("重构整个认证模块", "complex", "plan", "short but large"),
    ("把会话存储从 JSON 文件迁移到 SQLite，要兼容旧数据，提供迁移命令，并保证 /resume 等所有命令照常工作", "complex", "plan", ""),
    ("设计并实现一个插件系统：第三方可以通过 entry point 注册工具，要有版本检查、权限声明和加载失败隔离", "complex", "plan", ""),
    ("implement a full undo system: every write snapshots the file, /undo restores the last turn, /undo N goes back N turns, and it survives a restart", "complex", "plan", ""),
    ("从零实现一个 LSP 客户端，让 agent 能拿到跳转定义和查找引用的结果", "complex", "plan", ""),
    ("把 LLM 层从我们自己写的 OpenAI 兼容客户端换成官方 SDK，所有 provider 都要跑通，重试和溢出检测的行为不能变", "complex", "plan", ""),
    (
        "我想给 AgentCLI 加上远程运行：\n1. 起一个 HTTP 服务接收任务\n2. 任务排队执行\n"
        "3. 结果通过 SSE 推回去\n4. 支持取消\n5. 需要 token 鉴权",
        "complex",
        "plan",
        "numbered list",
    ),
    ("redesign the context manager so the protected zone is measured in turns instead of tokens, and rerun the retention experiment", "complex", "plan", ""),
    ("升级到 Python 3.14，把所有不兼容的地方修掉，CI 也要改", "complex", "plan", ""),
    ("给整个项目加上类型检查（mypy strict），修掉所有报错", "complex", "plan", ""),
    ("the orchestrator should support workers on different models; design the config, change the planner prompt, and add evals", "complex", "plan", ""),
    ("实现记忆的冲突检测：新记忆写入前找相似旧记忆，让模型决定新增、更新还是作废，要有版本和回滚", "complex", "plan", ""),
    ("把 bash 沙箱从 Docker 换成可插拔后端，支持 Docker 和 Windows Sandbox 两种", "complex", "plan", ""),
    ("逐个检查每个工具的权限声明是否正确，发现问题就修，修完一个再看下一个", "complex", "plan", "each, but one by one on purpose"),
    ("build a web UI for AgentCLI: chat view, tool approval dialog, session list, streamed output", "complex", "plan", ""),
    # ---- complex: team (independent parts)
    ("分别给 tools、memory、session、sandbox 四个模块补齐单元测试，覆盖率都到 90%", "complex", "team", ""),
    ("把 docs 目录下每个 md 文件分别翻译成英文", "complex", "team", ""),
    ("review every module under src/agentcli for unhandled exceptions and write a report per module", "complex", "team", ""),
    ("同时做三件事：修 web_fetch 的编码问题、给 grep 加 --type 参数、更新 README 的安装说明", "complex", "team", "three unrelated changes"),
    ("为 OpenAI、Anthropic、Gemini、DeepSeek 四个 provider 各写一个集成测试", "complex", "team", ""),
    ("benchmark the five summary models on the retention set, each in its own run, and compare", "complex", "team", ""),
    ("把项目里所有 print 调试语句分别按模块清理掉，每个模块单独提交", "complex", "team", ""),
    ("并行地给 evals 下的四个实验脚本各加一个 --dry-run 选项", "complex", "team", ""),
    ("audit the three MCP servers we ship with (chrome, filesystem, git) for path escapes, one report each", "complex", "team", ""),
    ("给每个内置工具写一份使用文档，放到 docs/tools/ 下，一个工具一个文件", "complex", "team", ""),
    ("迁移 4 个旧实验的结果格式到新 schema：context_retention、workers、models、memory_recall，互不相关", "complex", "team", ""),
    ("port the CLI's colour theme, its key bindings, and its status line to the new renderer; the three are independent", "complex", "team", ""),
]
