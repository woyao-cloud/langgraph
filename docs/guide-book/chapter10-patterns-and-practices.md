# 第十章：架构模式与最佳实践

在前九章中，我们分别学习了 LangGraph 的核心概念、API 详解、流式输出、检查点、人机交互、高级路由和函数式 API。本章将这些知识融会贯通，总结出实践中最常用的架构模式、需要警惕的反模式、调试技巧和性能优化策略，以及与 LangChain 生态的集成方式。

---

## 10.1 常见架构模式

### 10.1.1 ReAct 代理模式

ReAct（Reasoning + Acting）是最经典的 LLM 代理模式：模型先**思考**需要做什么，然后**行动**调用工具，再**观察**工具返回的结果，循环往复直到得出最终答案。

```python
import operator
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 定义工具
@tool
def search(query: str) -> str:
    """搜索天气信息"""
    return f"查询结果：{query} 今天晴，气温 25 度"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        return f"计算结果：{eval(expression)}"  # 仅示例，生产环境请用安全方式
    except Exception as e:
        return f"计算错误：{e}"

tools = [search, calculate]

# 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 定义代理节点
def agent_node(state: AgentState) -> dict:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)
    system = SystemMessage(content="你是一个有用的助手。使用工具回答问题。")
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}

# 构建图
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()

# 运行
result = graph.invoke({
    "messages": [HumanMessage(content="北京今天天气怎么样？比上海高几度？")]
})
print(result["messages"][-1].content)
```

**模式要点**：

- `tools_condition` 判断模型是否发起了工具调用，有则路由到 `tools` 节点，无则到 `END`
- 工具执行后回到 `agent` 节点，形成思考-行动-观察循环
- 需设置递归限制（默认 25）防止无限循环

---

### 10.1.2 Multi-Agent 协作模式

当单一代理无法胜任复杂任务时，多代理协作是常见解法。LangGraph 支持两种主要协作模式：

**Supervisor 模式**：一个调度节点决定将任务分配给哪个专业节点。

```python
import operator
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    next: str

def supervisor(state: State) -> dict:
    """调度节点：根据最后一条消息决定下一个专业节点"""
    last = state["messages"][-1]
    if isinstance(last, HumanMessage) and "翻译" in last.content:
        return {"next": "translator"}
    elif isinstance(last, HumanMessage) and "代码" in last.content:
        return {"next": "coder"}
    return {"next": "general"}

def translator(state: State) -> dict:
    return {"messages": [{"role": "assistant", "content": "翻译结果：..."}], "next": "supervisor"}

def coder(state: State) -> dict:
    return {"messages": [{"role": "assistant", "content": "代码结果：..."}], "next": "supervisor"}}

def general_assistant(state: State) -> dict:
    return {"messages": [{"role": "assistant", "content": "通用回答：..."}], "next": "end"}

def router(state: State) -> str:
    next_node = state["next"]
    if next_node == "end":
        return END
    return next_node

builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
builder.add_node("translator", translator)
builder.add_node("coder", coder)
builder.add_node("general", general_assistant)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", router, {"translator": "translator", "coder": "coder", "general": "general", END: END})
builder.add_edge("translator", "supervisor")
builder.add_edge("coder", "supervisor")

graph = builder.compile()
result = graph.invoke({"messages": [HumanMessage(content="帮我翻译这段话")], "next": ""})
```

**Swarm 模式**：节点之间通过 `Command` 直接切换控制权，无需中央调度。

```python
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command

class State(TypedDict):
    messages: Annotated[list, add_messages]

def researcher(state: State) -> dict:
    last = state["messages"][-1]
    if "写代码" in (last.content if hasattr(last, "content") else ""):
        # 直接将控制权转给 coder，同时更新状态
        return Command(goto="coder", update={"messages": [{"role": "assistant", "content": "研究完成，转交编码"}]})
    return {"messages": [{"role": "assistant", "content": "研究结论：..."}]}

def coder(state: State) -> dict:
    return {"messages": [{"role": "assistant", "content": "代码编写完成"}]}

builder = StateGraph(State)
builder.add_node("researcher", researcher, destinations=("coder", END))
builder.add_node("coder", coder)
builder.add_edge(START, "researcher")
builder.add_edge("coder", END)

graph = builder.compile()
```

**模式对比**：

| 特性 | Supervisor | Swarm |
|------|-----------|-------|
| 调度方式 | 中央调度 | 节点间直接切换 |
| 耦合度 | 调度逻辑集中 | 路由逻辑分散在各节点 |
| 扩展性 | 新增节点只需修改调度器 | 需要每个相关节点感知新节点 |
| 适用场景 | 节点多、路由规则复杂 | 节点少、切换逻辑简单 |

---

### 10.1.3 Map-Reduce 模式

Map-Reduce 模式使用 `Send` 实现扇出（fan-out）：将一个列表拆分成多个并行任务，各自处理后再汇总结果。

```python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class OverallState(TypedDict):
    topics: list[str]
    summaries: Annotated[list[str], operator.add]

def map_topics(state: OverallState) -> list[Send]:
    """将每个主题发送到 summary 节点并行处理"""
    return [Send("summarize", {"topic": t}) for t in state["topics"]]

def summarize(state: dict) -> dict:
    """对单个主题生成摘要"""
    topic = state["topic"]
    # 实际场景中这里调用 LLM
    return {"summaries": [f"关于{topic}的摘要内容"]}

builder = StateGraph(OverallState)
builder.add_node("summarize", summarize)
builder.add_conditional_edges(START, map_topics)
builder.add_edge("summarize", END)

graph = builder.compile()

result = graph.invoke({"topics": ["人工智能", "量子计算", "区块链"], "summaries": []})
print(result["summaries"])
# ['关于人工智能的摘要内容', '关于量子计算的摘要内容', '关于区块链的摘要内容']
```

**模式要点**：

- `Send` 的第二个参数是目标节点的输入状态，可以与主图状态不同
- 使用 `Annotated[list, operator.add]` 作为 reducer，自动合并多个并行结果
- 所有并行任务在同一个超步中执行，完成后才进入下一步

---

### 10.1.4 审核与纠正模式

生成-审核-纠正是提升 LLM 输出质量的核心模式：先让模型生成内容，再让另一个节点审核，不通过则修正后重新审核，循环直到满意。

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

class ReviewState(TypedDict):
    messages: Annotated[list, add_messages]
    draft: str
    revision_count: int

MAX_REVISIONS = 3

def writer(state: ReviewState) -> dict:
    """生成初稿或根据反馈修改"""
    messages = state["messages"]
    if state["draft"]:
        # 根据审核意见修改
        prompt = f"根据以下意见修改文章：\n当前草稿：{state['draft']}\n审核意见：{messages[-1].content}"
    else:
        prompt = messages[0].content if messages else "写一篇关于AI的文章"

    # 实际场景中调用 LLM
    draft = f"修改后的文章（第{state['revision_count'] + 1}版）：基于{prompt[:20]}..."
    return {"draft": draft, "revision_count": state["revision_count"] + 1}

def reviewer(state: ReviewState) -> dict:
    """审核草稿质量"""
    if state["revision_count"] >= MAX_REVISIONS:
        return {"messages": [AIMessage(content="通过：已达到最大修改次数，接受当前版本")]}
    # 模拟审核逻辑
    quality_score = min(state["revision_count"] / MAX_REVISIONS, 1.0)
    if quality_score >= 0.8:
        return {"messages": [AIMessage(content="通过：文章质量合格")]}
    return {"messages": [AIMessage(content="不通过：请增加更多细节和示例")]}

def should_continue(state: ReviewState) -> str:
    last = state["messages"][-1]
    if "通过" in last.content:
        return END
    return "writer"

builder = StateGraph(ReviewState)
builder.add_node("writer", writer)
builder.add_node("reviewer", reviewer)

builder.add_edge(START, "writer")
builder.add_edge("writer", "reviewer")
builder.add_conditional_edges("reviewer", should_continue, {END: END, "writer": "writer"})

graph = builder.compile()
result = graph.invoke({
    "messages": [HumanMessage(content="写一篇关于大语言模型的技术文章")],
    "draft": "",
    "revision_count": 0
})
print(result["draft"])
```

---

### 10.1.5 状态机模式

状态机模式用条件边实现业务状态转移，每条边对应一次合法的状态变迁，节点函数只处理当前状态的逻辑。

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class OrderState(TypedDict):
    order_id: str
    status: str          # created -> paid -> shipped -> completed | cancelled
    amount: float
    tracking_number: str

# 状态转移规则
VALID_TRANSITIONS = {
    "created": ["paid", "cancelled"],
    "paid": ["shipped", "cancelled"],
    "shipped": ["completed"],
    "cancelled": [],
    "completed": [],
}

def process_payment(state: OrderState) -> dict:
    return {"status": "paid"}

def ship_order(state: OrderState) -> dict:
    return {"status": "shipped", "tracking_number": "SF1234567890"}

def complete_order(state: OrderState) -> dict:
    return {"status": "completed"}

def cancel_order(state: OrderState) -> dict:
    return {"status": "cancelled"}

def route_by_status(state: OrderState) -> str:
    current = state["status"]
    transitions = VALID_TRANSITIONS.get(current, [])
    if current == "created" and "paid" in transitions:
        return "process_payment"
    elif current == "paid" and "shipped" in transitions:
        return "ship_order"
    elif current == "shipped" and "completed" in transitions:
        return "complete_order"
    elif current == "cancelled":
        return END
    elif current == "completed":
        return END
    return END

def validate_transition(state: OrderState) -> str:
    """验证状态转移是否合法"""
    # 可在此加入业务校验，如金额大于0才允许支付
    if state["status"] == "cancelled":
        return END
    return state["status"]

builder = StateGraph(OrderState)
builder.add_node("process_payment", process_payment)
builder.add_node("ship_order", ship_order)
builder.add_node("complete_order", complete_order)
builder.add_node("cancel_order", cancel_order)

builder.add_edge(START, "process_payment")
builder.add_conditional_edges("process_payment", route_by_status, {
    "ship_order": "ship_order",
    END: END,
})
builder.add_conditional_edges("ship_order", route_by_status, {
    "complete_order": "complete_order",
    END: END,
})
builder.add_edge("complete_order", END)

graph = builder.compile()
result = graph.invoke({
    "order_id": "ORD-001",
    "status": "created",
    "amount": 99.9,
    "tracking_number": "",
})
print(result["status"])  # shipped
```

---

### 10.1.6 递归代理模式

递归代理让一个图作为子图嵌入自身，实现自我调用。关键是用 `recursion_limit` 控制深度，防止无限递归。

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

class RecursiveState(TypedDict):
    messages: Annotated[list, add_messages]
    depth: int
    task: str

MAX_DEPTH = 3

def planner(state: RecursiveState) -> dict:
    """规划节点：判断是否需要进一步分解"""
    if state["depth"] >= MAX_DEPTH:
        return {"messages": [AIMessage(content=f"到达最大深度，直接处理：{state['task']}")]}
    return {"messages": [AIMessage(content=f"深度{state['depth']}：需要将任务分解")], "depth": state["depth"] + 1}

def executor(state: RecursiveState) -> dict:
    """执行节点：处理当前层级的任务"""
    return {"messages": [AIMessage(content=f"执行任务：{state['task'][:20]}...")]}

def should_recurse(state: RecursiveState) -> str:
    if state["depth"] >= MAX_DEPTH:
        return "executor"
    return "planner"  # 递归回到规划器

builder = StateGraph(RecursiveState)
builder.add_node("planner", planner)
builder.add_node("executor", executor)

builder.add_edge(START, "planner")
builder.add_conditional_edges("planner", should_recurse)
builder.add_edge("executor", END)

graph = builder.compile()

result = graph.invoke({
    "messages": [HumanMessage(content="设计一个分布式系统架构")],
    "depth": 0,
    "task": "设计一个分布式系统架构",
}, config={"recursion_limit": 20})  # 递归限制需覆盖总步数
print(result["messages"][-1].content)
```

---

## 10.2 反模式

### 10.2.1 在节点间传递大对象

**问题**：将完整的文档内容、图片字节或大数据集放入状态，导致每次检查点保存和状态传递的开销急剧增加。

**正确做法**：状态中只存引用键（如文件 ID、文档 ID），实际数据通过 `BaseStore` 获取。

```python
# 错误：在状态中存大对象
class BadState(TypedDict):
    document: str  # 可能是几 MB 的文本

# 正确：只存引用，运行时从 store 读取
from langgraph.store.base import BaseStore

class GoodState(TypedDict):
    document_id: str  # 只存 ID

def process_node(state: GoodState, *, store: BaseStore) -> dict:
    doc = store.get(("documents",), state["document_id"])
    # 处理文档...
    return {"document_id": state["document_id"]}
```

### 10.2.2 过度使用全局状态

**问题**：所有节点共享一个巨大的状态字典，导致耦合严重、难以理解和测试。

**正确做法**：用子图隔离不同关注域，子图有独立的状态 schema，只在必要时通过输入/输出 schema 与父图交互。

```python
# 错误：一个巨大的全局状态
class MegaState(TypedDict):
    user_input: str
    search_results: list
    translation: str
    code_output: str
    review_comments: str
    # ... 继续膨胀

# 正确：用子图隔离
class SearchState(TypedDict):
    query: str
    results: list

search_graph = StateGraph(SearchState)
# ... search 子图独立定义

class MainState(TypedDict):
    query: str
    search_results: list

main_graph = StateGraph(MainState)
main_graph.add_node("search", search_graph.compile())  # 子图作为节点
```

### 10.2.3 忽略检查点开销

**问题**：对每次执行都使用 PostgresSaver，即使任务只运行几秒且不需要持久化。

**正确做法**：根据场景选择合适的持久化策略。

| Durability 模式 | 行为 | 适用场景 |
|-----------------|------|---------|
| `"sync"` | 每步同步保存 | 关键业务流程，不能丢失任何状态 |
| `"async"` | 异步保存，下一步不等待保存完成 | 大多数生产场景 |
| `"exit"` | 仅在图退出时保存 | 短流程、可容忍少量丢失 |
| 不设 Checkpointer | 不持久化 | 纯计算、一次性任务 |

```python
from langgraph.types import Durability
graph.invoke(input, config={"configurable": {"thread_id": "t1", "durability": "async"}})
```

子图的 Checkpointer 也需要合理选择：`True` 表示独立持久化，`False` 表示禁止持久化（即使父图有），`None` 表示继承父图的 Checkpointer。

### 10.2.4 在条件边中做耗时操作

**问题**：条件边函数中调用 LLM 或执行 IO，导致路由变慢且无法被检查点追踪。

**正确做法**：条件边只做纯逻辑判断，耗时操作放在节点中。

```python
# 错误：条件边中调用 LLM
def bad_router(state):
    llm = ChatOpenAI()
    result = llm.invoke(f"判断走哪条路径：{state['input']}")  # 耗时操作！
    return result

# 正确：将判断逻辑放在节点，条件边只做字符串匹配
def judge_node(state):
    # 在节点中做 LLM 调用
    decision = llm.invoke(f"判断走哪条路径：{state['input']}")
    return {"route": decision.content}

def router(state):
    return state["route"]  # 条件边只做纯映射
```

### 10.2.5 状态中的不可序列化对象

**问题**：状态中存储数据库连接、文件句柄等不可 pickle 的对象，导致检查点保存失败。

**正确做法**：所有状态字段必须可序列化（可 pickle），连接和句柄通过 `context_schema` 或配置注入。

```python
# 错误：状态中存数据库连接
class BadState(TypedDict):
    db_conn: sqlite3.Connection  # 不可 pickle！

# 正确：通过 context_schema 注入
from langgraph.runtime import Runtime

class GoodState(TypedDict):
    user_id: str

class Context(TypedDict):
    db_dsn: str  # 只存连接字符串

def my_node(state: GoodState, runtime: Runtime[Context]) -> dict:
    dsn = runtime.context["db_dsn"]
    # 在节点内创建连接，用完即关
    return {"user_id": state["user_id"]}
```

### 10.2.6 递归限制设置不当

**问题**：默认递归限制为 25 步。图深度为 5 时可能够用，但包含循环的图在长对话中很容易触发 `GraphRecursionError`。

**正确做法**：根据图的实际深度和循环预期次数设置合理的 `recursion_limit`。

```python
# 线性图：5 个节点，设为 10 足够
result = graph.invoke(input, config={"recursion_limit": 10})

# ReAct 循环：每次循环至少 2 步（agent + tools），最多 10 轮
result = graph.invoke(input, config={"recursion_limit": 25})

# 如果确实需要更多步数，显式设置更大值
result = graph.invoke(input, config={"recursion_limit": 50})
```

---

## 10.3 调试技巧

### 10.3.1 使用 interrupt_before 调试

在特定节点前暂停执行，检查当前状态。

```python
graph = builder.compile(
    checkpointer=InMemorySaver(),
    interrupt_before=["reviewer"],  # 在 reviewer 节点前暂停
)
config = {"configurable": {"thread_id": "debug-1"}}
result = graph.invoke(input, config)
# 此时 reviewer 还未执行，可以检查状态
snapshot = graph.get_state(config)
print(snapshot.values)  # 查看暂停时的完整状态
print(snapshot.next)    # 查看下一个要执行的节点
```

### 10.3.2 使用 stream_mode="debug" 查看详细执行过程

`debug` 模式会输出每一步的任务调度和检查点信息，是最详细的调试视图。

```python
for event in graph.stream(input, config, stream_mode="debug"):
    if event["type"] == "task":
        print(f"启动任务：{event['payload']['name']}")
    elif event["type"] == "task_result":
        print(f"任务完成：{event['payload']['name']}, 结果={event['payload']['result']}")
    elif event["type"] == "checkpoint":
        print(f"检查点保存，next={event['payload']['next']}")
```

### 10.3.3 使用 get_state() 检查中间状态

在任何时刻调用 `get_state()` 获取图的当前快照。

```python
snapshot = graph.get_state(config)
print(f"当前状态值: {snapshot.values}")
print(f"下一步要执行的节点: {snapshot.next}")
print(f"待处理的中断: {snapshot.interrupts}")
print(f"元数据: {snapshot.metadata}")
```

### 10.3.4 使用 get_state_history() 回溯执行路径

`get_state_history()` 返回所有历史检查点，帮助你理解执行的每一步。

```python
for state in graph.get_state_history(config):
    print(f"步骤 {state.metadata.get('step', '?')}: next={state.next}")
    print(f"  状态: {state.values}")
    print(f"  父检查点: {state.parent_config}")
```

### 10.3.5 可视化图结构

使用 Mermaid 语法导出图结构，直观理解节点和边的拓扑关系。

```python
# 生成 Mermaid 图描述
print(graph.get_graph().draw_mermaid())

# 如果安装了 pygraphviz，可以导出为 PNG
# graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
```

### 10.3.6 日志配置

设置 LangGraph 的日志级别来查看内部执行细节。

```python
import logging
logging.basicConfig(level=logging.DEBUG)
# 或者只设置 langgraph 的日志
logging.getLogger("langgraph").setLevel(logging.DEBUG)
```

---

## 10.4 性能优化

### 10.4.1 选择合适的 StreamMode

不同流模式的开销和适用场景差异显著：

| StreamMode | 数据量 | 适用场景 |
|------------|--------|---------|
| `"values"` | 大（每步完整状态） | 需要每步完整快照 |
| `"updates"` | 中（每步增量更新） | 监控节点输出 |
| `"messages"` | 小（逐 token） | 聊天界面流式输出 |
| `"custom"` | 由你控制 | 自定义监控指标 |
| `"debug"` | 大 | 仅调试时使用 |

对于聊天场景，优先使用 `"messages"` 模式实现逐字输出：

```python
async for msg, metadata in graph.astream(input, config, stream_mode="messages"):
    if hasattr(msg, "content") and msg.content:
        print(msg.content, end="", flush=True)
```

### 10.4.2 合理使用 cache_policy

对确定性任务（如基于固定输入的摘要生成），启用缓存避免重复计算。

```python
from langgraph.types import CachePolicy

def summarize(state):
    # 确定性操作：相同输入一定产生相同输出
    return {"summary": expensive_llm_call(state["text"])}

builder.add_node("summarize", summarize, cache_policy=CachePolicy(ttl=3600))
```

### 10.4.3 合理使用 retry_policy

对不稳定的操作（如网络请求、LLM API 调用），设置重试策略。

```python
from langgraph.types import RetryPolicy

builder.add_node("call_llm", llm_node, retry_policy=RetryPolicy(
    max_attempts=3,
    initial_interval=0.5,
    backoff_factor=2.0,
    retry_on=(RateLimitError, TimeoutError),  # 仅对特定错误重试
))
```

### 10.4.4 异步优先

LangGraph 的异步 API 性能显著优于同步版本，尤其是在并发场景下。

```python
# 优先使用异步 API
result = await graph.ainvoke(input, config)
async for chunk in graph.astream(input, config):
    process(chunk)
```

### 10.4.5 最小化状态大小

状态越精简，检查点保存和状态传递越快。避免冗余字段。

```python
# 冗余：同时存原文和摘要
class VerboseState(TypedDict):
    original_text: str  # 可能很长
    summary: str        # 可以从 original_text 重新生成

# 精简：只存原文，摘要作为临时结果不进状态
class LeanState(TypedDict):
    original_text: str
```

### 10.4.6 子图的 Checkpointer 选择

- `None`（默认）：继承父图的 Checkpointer，最常用
- `True`：子图独立持久化，有自己的 `thread_id`，适合需要独立恢复的子流程
- `False`：完全禁用子图的检查点，适合纯计算子图

```python
sub_graph = sub_builder.compile(checkpointer=False)  # 子图无需持久化
main_graph = StateGraph(MainState)
main_graph.add_node("sub", sub_graph)  # 子图节点不产生检查点开销
```

---

## 10.5 与 LangChain 生态集成

### 10.5.1 ChatModel 集成

LangGraph 节点中直接使用 LangChain 的 ChatModel，享受统一的模型接口。

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

def chat_node(state):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system = SystemMessage(content="你是一个专业的技术顾问。")
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}
```

### 10.5.2 Tool 集成

使用 `ToolNode` 将 LangChain 工具自动包装为图节点，处理工具调用解析和结果返回。

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

@tool
def lookup_user(user_id: str) -> str:
    """查询用户信息"""
    return f"用户 {user_id}：张三，VIP 用户"

@tool
def query_orders(user_id: str) -> str:
    """查询用户订单"""
    return f"用户 {user_id} 有 3 个订单"

tools = [lookup_user, query_orders]
tool_node = ToolNode(tools)
```

### 10.5.3 Retriever 集成：RAG 模式

将 LangChain 的 Retriever 集成到图中，实现检索增强生成。

```python
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def retrieve_node(state):
    """检索相关文档并添加到消息中"""
    # 实际场景中使用真实向量库
    docs = [{"page_content": "LangGraph 是一个工作流引擎", "metadata": {}}]
    context = "\n".join(d["page_content"] for d in docs)
    return {"messages": [AIMessage(content=f"检索到以下相关内容：\n{context}")]}

def generate_node(state):
    """基于检索结果生成回答"""
    return {"messages": [AIMessage(content="基于检索结果生成的回答...")]}
```

### 10.5.4 Store 集成：跨线程持久化

`BaseStore` 提供跨线程的键值存储，适合存储用户偏好、知识库等需要跨会话共享的数据。

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
# 跨线程存储用户偏好
store.put(("user_prefs",), "user_123", {"language": "zh", "style": "formal"})

def personalized_node(state, *, store: BaseStore):
    prefs = store.get(("user_prefs",), "user_123")
    lang = prefs.value.get("language", "en") if prefs else "en"
    return {"messages": [AIMessage(content=f"使用{lang}回复...")]}

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer, store=store)
```

**Store vs Checkpoint 的区别**：

| 特性 | Checkpoint | Store |
|------|-----------|-------|
| 作用域 | 单线程（`thread_id`） | 跨线程共享 |
| 数据模型 | 图的完整状态快照 | 任意键值对 |
| 更新方式 | 每步自动保存 | 手动 put/get |
| 适用场景 | 流程恢复、回溯 | 用户配置、共享知识库 |

---

## 10.6 小结

本章总结了六种最常见的 LangGraph 架构模式及其完整实现：ReAct 代理、Multi-Agent 协作（Supervisor 与 Swarm）、Map-Reduce、审核纠正、状态机和递归代理。同时列出了六种常见反模式及其修正方案，以及涵盖中断调试、流式调试、状态检查、历史回溯、可视化和日志的调试工具箱。性能优化方面，从流模式选择、缓存策略、重试策略、异步优先、状态精简和子图检查点六个维度给出了具体建议。最后，我们展示了 LangGraph 与 LangChain 生态的四种集成方式：ChatModel、Tool、Retriever 和 Store。

掌握这些模式和最佳实践，你就具备了用 LangGraph 构建生产级 LLM 应用的能力。但请记住：**模式是工具，不是枷锁。** 每个应用都有其独特的需求，选择最适合的架构而不是最复杂的架构，才是工程师的判断力所在。