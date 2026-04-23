# 第四章 StateGraph API 详解

StateGraph 是 LangGraph 中最核心的类，用于构建基于共享状态的有向图。本章将全面讲解 StateGraph 的每个 API 方法，并辅以可运行的代码示例。

## 4.1 StateGraph 类

### 4.1.1 构造函数

```python
class StateGraph(
    state_schema: type[StateT],
    context_schema: type[ContextT] | None = None,
    *,
    input_schema: type[InputT] | None = None,
    output_schema: type[OutputT] | None = None,
)
```

**参数说明：**

- `state_schema`：定义图的核心状态结构。通常使用 `TypedDict` 或 `Pydantic BaseModel`，支持 `Annotated` 标注 reducer 函数。
- `context_schema`：定义运行时不可变上下文的数据结构。与 `state_schema` 不同，context 中的数据在节点间是只读的，适合传递 `user_id`、数据库连接等环境信息。通过 `Runtime[Context]` 在节点中访问。
- `input_schema`：定义图的输入结构。当输入结构与完整状态不同时使用，LangGraph 会在入口处将输入映射为完整状态。默认为 `state_schema`。
- `output_schema`：定义图的输出结构。当输出只是完整状态的子集时使用，只有 `output_schema` 中声明的字段会作为最终结果返回。默认为 `state_schema`。

```python
from typing import Annotated, TypedDict
from typing_extensions import Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime

# 基本状态定义
class State(TypedDict):
    messages: Annotated[list, operator.add]
    count: int

# 带 context_schema 的状态定义
class Context(TypedDict):
    user_id: str
    db_url: str

# 区分输入输出状态
class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

class FullState(TypedDict):
    question: str
    answer: str
    internal_notes: str  # 不会出现在输出中

# 构造图
graph = StateGraph(FullState, input_schema=InputState, output_schema=OutputState)
```

### 4.1.2 add_node()

`add_node()` 有多种签名变体，支持灵活的节点注册方式：

```python
# 方式一：自动推断名称（推荐）
def add_node(self, node: StateNode, *, ...) -> Self

# 方式二：显式指定名称
def add_node(self, node: str, action: StateNode, *, ...) -> Self
```

**关键参数：**

- `node` / `action`：节点函数或 Runnable 对象。节点函数可接受以下签名：
  - `(state) -> dict`：最简形式
  - `(state, config: RunnableConfig) -> dict`：带配置
  - `(state, *, writer: StreamWriter) -> dict`：带自定义流写入器
  - `(state, *, store: BaseStore) -> dict`：带存储
  - `(state, *, runtime: Runtime[Context]) -> dict`：带运行时上下文
- `defer`：是否延迟执行。设为 `True` 时，节点会在当前超步即将结束时才执行，适合需要收集其他节点全部写入后才能运行的聚合节点。
- `retry_policy`：重试策略，类型为 `RetryPolicy` 或 `RetryPolicy` 序列。`RetryPolicy` 参数包括 `initial_interval`（首次重试间隔，默认 0.5s）、`backoff_factor`（退避倍数，默认 2.0）、`max_interval`（最大间隔，默认 128s）、`max_attempts`（最大重试次数，默认 3）、`jitter`（是否添加随机抖动，默认 True）。
- `cache_policy`：缓存策略，类型为 `CachePolicy`。可指定 `key_func`（自定义缓存键生成函数）和 `ttl`（缓存过期时间，秒）。
- `destinations`：声明节点可能路由到的目标节点名。仅用于图可视化，不影响执行逻辑。当节点返回 `Command` 对象时，此参数帮助生成正确的图结构图。

```python
from langgraph.types import RetryPolicy, CachePolicy, StreamWriter
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    value: int
    log: Annotated[list[str], operator.add]

# 基本节点
def step_a(state: State) -> dict:
    return {"value": state["value"] + 1, "log": ["step_a"]}

# 带重试策略的节点
def step_b(state: State) -> dict:
    return {"value": state["value"] * 2, "log": ["step_b"]}

# 带 StreamWriter 的节点（用于 stream_mode="custom"）
def step_c(state: State, *, writer: StreamWriter) -> dict:
    writer(f"处理中：当前值为 {state['value']}")
    return {"value": state["value"] + 10, "log": ["step_c"]}

builder = StateGraph(State)
builder.add_node(step_a)  # 自动推断名称为 "step_a"
builder.add_node("step_b", step_b, retry_policy=RetryPolicy(max_attempts=5))
builder.add_node("step_c", step_c, cache_policy=CachePolicy(ttl=60))
builder.add_edge(START, "step_a")
builder.add_edge("step_a", "step_b")
builder.add_edge("step_b", "step_c")
builder.add_edge("step_c", END)
graph = builder.compile()
result = graph.invoke({"value": 1, "log": []})
# {'value': 14, 'log': ['step_a', 'step_b', 'step_c']}
```

### 4.1.3 add_edge()

```python
def add_edge(self, start_key: str | list[str], end_key: str) -> Self
```

- **普通边**：`start_key` 为单个字符串，表示节点执行完后跳转到 `end_key`。
- **等待边（fan-in）**：`start_key` 为字符串列表，表示图会等待列表中所有节点完成后才执行 `end_key`。这在 map-reduce 模式中非常常用。

```python
from langgraph.types import Send

class State(TypedDict):
    topics: list[str]
    summaries: Annotated[list[str], operator.add]

def summarize(state: State) -> dict:
    return {"summaries": [f"摘要：{state['topics']}"]}

def aggregate(state: State) -> dict:
    return {"summaries": ["聚合完成"]}

builder = StateGraph(State)
builder.add_node("summarize", summarize)
builder.add_node("aggregate", aggregate)
# fan-in 边：等待 summarize 完成后才执行 aggregate
builder.add_edge(["summarize"], "aggregate")
```

### 4.1.4 add_conditional_edges()

```python
def add_conditional_edges(
    self,
    source: str,
    path: Callable | Runnable,
    path_map: dict[Hashable, str] | list[str] | None = None,
) -> Self
```

- `source`：出发节点名称。
- `path`：路由函数，接收状态作为输入，返回目标节点名称（字符串）、节点名称列表或 `Send` 对象。返回 `END` 表示结束。
- `path_map`：可选的路由映射表。键是 `path` 函数的返回值，值是实际的目标节点名称。如果省略，则 `path` 的返回值直接作为目标节点名。也可以用 `list[str]`，等价于 `{name: name for name in list}`。

**Literal 自动推断：** 如果 `path` 函数的返回类型标注为 `Literal["node_a", "node_b"]`，LangGraph 会自动推断出 `path_map`，无需手动指定。

```python
from typing import Literal

class State(TypedDict):
    query: str
    response: str

def router(state: State) -> Literal["chat", "search"]:
    """Literal 类型注解让 LangGraph 自动推断路径映射"""
    if "搜索" in state["query"]:
        return "search"
    return "chat"

def chat_node(state: State) -> dict:
    return {"response": f"聊天回复：{state['query']}"}

def search_node(state: State) -> dict:
    return {"response": f"搜索结果：{state['query']}"}

builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.add_node("search", search_node)
# 无需 path_map，Literal 自动推断
builder.add_conditional_edges(START, router)
builder.add_edge("chat", END)
builder.add_edge("search", END)
graph = builder.compile()
```

使用显式 `path_map` 的示例：

```python
def sentiment_router(state: State) -> str:
    if state["query"].startswith("开心"):
        return "positive"
    elif state["query"].startswith("难过"):
        return "negative"
    return "neutral"

builder.add_conditional_edges(
    "entry",
    sentiment_router,
    path_map={
        "positive": "happy_node",
        "negative": "sad_node",
        "neutral": "neutral_node",
    }
)
```

### 4.1.5 set_entry_point() / set_finish_point()

```python
def set_entry_point(self, key: str) -> Self
def set_finish_point(self, key: str) -> Self
```

等价于 `add_edge(START, key)` 和 `add_edge(key, END)`，提供更直观的语义。

```python
builder = StateGraph(State)
builder.add_node("process", process_fn)
builder.set_entry_point("process")   # START -> process
builder.set_finish_point("process")   # process -> END
```

### 4.1.6 compile()

```python
def compile(
    self,
    checkpointer: Checkpointer = None,
    *,
    cache: BaseCache | None = None,
    store: BaseStore | None = None,
    interrupt_before: All | list[str] | None = None,
    interrupt_after: All | list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
) -> CompiledStateGraph
```

**参数详解：**

- `checkpointer`：检查点持久化器。传入 `InMemorySaver()` 等实例启用状态持久化，支持断点恢复、时间旅行等高级功能。设为 `False` 则显式禁用（即使父图有 checkpointer 也不会继承）。默认 `None` 表示继承父图的设置。
- `store`：跨线程的持久化键值存储，类型为 `BaseStore`。适合在多个会话间共享长期记忆。
- `interrupt_before` / `interrupt_after`：在指定节点执行前/后中断执行，用于人机交互场景。传入 `"*"` 表示在所有节点处中断。
- `debug`：启用调试模式，会输出更详细的执行日志。
- `name`：为编译后的图指定名称，便于在 LangSmith 等追踪工具中识别。

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["approval_node"],  # 在审批节点前暂停
    debug=True,
)
```

## 4.2 CompiledStateGraph

`compile()` 返回的 `CompiledStateGraph` 对象是真正可执行的图，它实现了 LangChain 的 `Runnable` 接口。

### 4.2.1 invoke() / ainvoke()

```python
# 同步执行
result = graph.invoke(
    input,                          # 输入数据
    config=None,                    # RunnableConfig，含 thread_id 等
    *,
    context=None,                   # 运行时上下文（对应 context_schema）
    stream_mode="values",           # 流模式（invoke 中通常默认 "values"）
    durability=None,                # 持久化模式："sync"/"async"/"exit"
    version="v1",                   # 输出格式版本
)

# 异步执行
result = await graph.ainvoke(input, config)
```

`invoke()` 阻塞直到图执行完成，返回最终状态。适合不需要中间步骤的场景。

### 4.2.2 stream() / astream()

```python
# 同步流式
for chunk in graph.stream(input, config, stream_mode="updates"):
    print(chunk)

# 异步流式
async for chunk in graph.astream(input, config, stream_mode="updates"):
    print(chunk)
```

`stream()` 逐步产出中间结果，详见第五章。

### 4.2.3 get_state() / update_state()

```python
# 获取当前状态快照
snapshot = graph.get_state(config)
print(snapshot.values)     # 当前状态值
print(snapshot.next)       # 下一步要执行的节点
print(snapshot.config)     # 配置信息
print(snapshot.metadata)   # 元数据（step、source 等）
print(snapshot.tasks)      # 待执行的任务

# 手动修改状态
graph.update_state(
    config,
    values={"messages": [{"role": "user", "content": "你好"}]},
    as_node="chatbot",  # 指定从哪个节点的视角写入
)

# 异步版本
snapshot = await graph.aget_state(config)
await graph.aupdate_state(config, values, as_node="chatbot")
```

`as_node` 参数非常关键：它指定写入从哪个节点的视角发生。这影响状态的 reducer 计算方式以及图的后续路由逻辑。

### 4.2.4 get_state_history()

```python
# 遍历状态历史
for state in graph.get_state_history(config):
    print(f"步骤 {state.metadata['step']}: {state.values}")
    print(f"下一个节点: {state.next}")

# 从历史检查点恢复执行
config_with_checkpoint = {
    "configurable": {
        "thread_id": "my-thread",
        "checkpoint_id": state.config["configurable"]["checkpoint_id"],
    }
}
# 使用历史检查点继续执行
for chunk in graph.stream(None, config_with_checkpoint):
    print(chunk)
```

### 4.2.5 节点名称获取

```python
# 获取编译图中所有节点名称
node_names = graph.nodes  # 返回 dict，键为节点名

# 获取图的 Mermaid 可视化
print(graph.get_graph().draw_mermaid())
```

## 4.3 完整示例：构建多步骤代理图

下面是一个完整的、可运行的多步骤代理图示例，涵盖本章介绍的所有核心 API：

```python
import operator
from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import StreamWriter, RetryPolicy

# 1. 定义状态
class AgentState(TypedDict):
    query: str
    research_result: str
    draft: str
    final_answer: str
    log: Annotated[list[str], operator.add]

# 2. 定义节点函数
def research(state: AgentState, *, writer: StreamWriter) -> dict:
    """调研节点：根据查询收集信息"""
    writer(f"[调研中] 查询：{state['query']}")
    result = f"关于「{state['query']}」的调研结果"
    return {"research_result": result, "log": ["research"]}

def draft_answer(state: AgentState) -> dict:
    """起草节点：基于调研结果起草回答"""
    draft = f"草稿：基于{state['research_result']}生成回答"
    return {"draft": draft, "log": ["draft"]}

def review(state: AgentState) -> Literal["approve", "revise"]:
    """审查节点：决定是否通过"""
    if len(state["draft"]) > 20:
        return "approve"
    return "revise"

def finalize(state: AgentState) -> dict:
    """终稿节点：输出最终答案"""
    return {"final_answer": state["draft"], "log": ["finalize"]}

def revise(state: AgentState) -> dict:
    """修订节点：改进草稿"""
    return {
        "draft": state["draft"] + "（已修订）",
        "log": ["revise"],
    }

# 3. 构建图
builder = StateGraph(AgentState)

# 添加节点
builder.add_node("research", research)
builder.add_node("draft", draft_answer, retry_policy=RetryPolicy(max_attempts=3))
builder.add_node("review", review)
builder.add_node("finalize", finalize)
builder.add_node("revise", revise)

# 设置边
builder.add_edge(START, "research")
builder.add_edge("research", "draft")
builder.add_conditional_edges("draft", review)
builder.add_edge("revise", "draft")  # 修订后重新起草
builder.add_edge("finalize", END)

# 4. 编译图
checkpointer = InMemorySaver()
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["finalize"],  # 在终稿前暂停，等待人工确认
)

# 5. 执行图
config = {"configurable": {"thread_id": "thread-1"}}

# 第一次执行，会在 finalize 之前中断
for event in graph.stream(
    {"query": "LangGraph 是什么？", "log": []},
    config,
    stream_mode="updates",
):
    print(event)

# 查看当前状态
state = graph.get_state(config)
print(f"下一步将执行：{state.next}")  # ('finalize',)

# 确认后继续执行
for event in graph.stream(None, config, stream_mode="updates"):
    print(event)

# 获取最终结果
final_state = graph.get_state(config)
print(final_state.values["final_answer"])
```

此示例展示了 StateGraph 的完整工作流：状态定义、节点添加、条件边、检查点、中断恢复，以及流式输出。在实际项目中，你可以将 `research`、`draft_answer` 等节点替换为真实的 LLM 调用，构建功能强大的代理系统。