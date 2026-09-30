"""Labelled requests for experiment 7: which tool should the agent reach for first?

Each entry: (request, acceptable first tools). Several tools can be right (grep or
search_code to find code; list_dir, glob or directory_tree to look around); "none" means
the agent should answer without a tool. Browser tools come from the Chrome DevTools MCP
server, which AgentCLI keeps deferred until needed.
"""

TOOLS_OK = [
    # ---- reading and finding
    ("看一下 src/agentcli/config.py 里 SandboxConfig 的默认值", ["read_file"]),
    ("pyproject.toml 里依赖了哪些包", ["read_file"]),
    ("项目里哪里调用了 sandbox_for？", ["grep", "search_code"]),
    ("find where the retry delay is computed", ["grep", "search_code"]),
    ("有哪些测试文件是测 memory 的", ["glob", "grep", "list_dir"]),
    ("list all markdown files under docs", ["glob", "list_dir", "directory_tree"]),
    ("src 目录下都有哪些模块", ["list_dir", "directory_tree", "glob"]),
    ("给我看看整个项目的目录结构", ["directory_tree", "list_dir"]),
    ("evals/results/jev_recall.json 有多大，什么时候改的", ["get_file_info"]),
    ("哪段代码负责把旧的工具结果换成存根", ["search_code", "grep"]),
    ("where do we handle the 'd' answer in the approval prompt", ["grep", "search_code"]),
    ("读一下 README 的安装部分", ["read_file"]),
    # ---- editing
    ("把 config.py 里 jev_timeout 的默认值改成 1.0", ["read_file", "edit_file"]),
    ("在 .gitignore 末尾加一行 evals/_home/", ["read_file", "edit_file"]),
    ("新建一个 docs/jev.md，写上这次实验的结论", ["write_file"]),
    ("rename the variable spent to spent_usd in evals/jev_recall.py", ["read_file", "edit_file", "grep"]),
    ("创建一个空的 tests/test_jev.py", ["write_file"]),
    # ---- shell
    ("跑一下全部测试", ["bash"]),
    ("git status 看看改了什么", ["bash"]),
    ("装一下 ruff 然后格式化整个项目", ["bash"]),
    ("run the memory recall eval script", ["bash"]),
    ("看看 docker 里有没有 agentcli-sandbox 镜像", ["bash"]),
    ("提交当前的改动，message 写 'eval: jev recall'", ["bash"]),
    ("统计一下 src 下 python 代码总行数", ["bash"]),
    # ---- web
    ("OpenRouter 现在 gpt-6-luna 的价格是多少", ["web_search", "web_fetch"]),
    ("search for the latest release of httpx", ["web_search"]),
    ("读一下 https://docs.typesafe.ai/api.md 这个页面", ["web_fetch"]),
    ("fetch https://pypi.org/project/mcp/ and tell me the latest version", ["web_fetch"]),
    ("最近有什么关于 prompt caching 的新文章", ["web_search"]),
    # ---- memory
    ("记住：这个项目的测试一律用 .venv 里的 python 跑", ["save_memory"]),
    ("以后提交信息都用英文，帮我记下来", ["save_memory"]),
    ("我之前说过测试数据库的端口是多少来着？", ["search_memory"]),
    ("did I ever tell you which branch we release from?", ["search_memory"]),
    # ---- skills and history
    ("用一下 commit-message 那个 skill", ["load_skill"]),
    ("把刚才这套发布流程保存成一个 skill", ["save_skill"]),
    ("撤销上一轮你对文件做的修改", ["revert_turn"]),
    # ---- browser (deferred MCP tools)
    ("打开 http://localhost:5173 看看首页", ["navigate_page", "new_page"]),
    ("截个图看看现在页面长什么样", ["take_screenshot", "take_snapshot"]),
    ("页面控制台有没有报错", ["list_console_messages"]),
    ("点一下页面上的登录按钮", ["take_snapshot", "click"]),
    ("在搜索框里输入 FinMate 然后回车", ["take_snapshot", "fill", "type_text"]),
    ("看看这个页面发了哪些接口请求", ["list_network_requests"]),
    ("跑一个 Lighthouse 看看无障碍评分", ["lighthouse_audit"]),
    ("check why the page is slow to load, record a performance trace", ["performance_start_trace"]),
    ("用 JS 取一下页面上 document.title", ["evaluate_script"]),
    ("把窗口调成手机尺寸看看布局", ["resize_page", "emulate"]),
    ("fill in the signup form with test data", ["take_snapshot", "fill_form"]),
    ("现在浏览器里开了几个标签页", ["list_pages"]),
    # ---- no tool
    ("解释一下什么是 RRF", ["none"]),
    ("SSE 和 WebSocket 有什么区别", ["none"]),
    ("谢谢，今天就到这", ["none"]),
    ("what's the difference between precision and recall?", ["none"]),
    ("给我写一个快速排序的 Python 示例", ["none"]),
    ("你觉得记忆检索该不该用向量库", ["none"]),
]
