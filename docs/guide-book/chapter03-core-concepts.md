# 第三章：核心概念

第二章我们讨论了 LangGraph 背后的理论——BSP、Actor、FSM、检查点和图论。现在让我们把这些理论落地，逐一解析 LangGraph 的核心概念，并用可运行的代码来理解它们。

## 3.1 State：图的共享数据

State 是 LangGraph 中最基础的概念。它定义了图中的所有节点可以共享读写的数据结构。你可以把 State 想象成一张共享白板——每个节点执行后，可以在白板上更新信息，下一个节点读取白板上的最新内容来决定自己的行为。

### TypedDict 状态

最常用的状态定义方式是 Python 的 `TypedDict`：

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]  # 对话消息列表
    context: str                               # 当前上下文
    score: float                               # 质量评分
    retry_count: int                           # 重试次数
```

这段代码定义了四个状态键。其中 `messages` 使用了 `Annotated[list, add_messages]`，这意味着 `messages` 键有一个特殊的合并策略——`add_messages` 是一个 reducer，它定义了当多个节点同时写入 `messages` 时，如何合并这些写入。

**每个键就是一个通道**——这是理解 State 的关键。State 不是一块扁平的内存，而是一组独立的通道（Channel），每个通道有自己的合并规则。

### Annotated[type, reducer]：如何合并多个写入

在 BSP 模型中，一个超步可能有多个节点并行执行，它们可能同时写入同一个状态键。例如，三个研究节点并行运行，每个都要往 `messages` 里追加一条消息：

```
超步 N：
  节点 A 写入 messages: [{"role": "assistant", "content": "研究结果A"}]
  节点 B 写入 messages: [{"role": "assistant", "content": "研究结果B"}]
  节点 C 写入 messages: [{"role": "assistant", "content": "研究结果C"}]

超步结束时合并：
  messages = reducer(旧messages, [结果A, 结果B, 结果C])
  → 最终 messages = 旧messages + [结果A, 结果B, 结果C]
```

`Annotated[type, reducer]` 就是告诉 LangGraph：当出现多个写入时，用 `reducer` 函数来合并，而不是简单地覆盖。

LangGraph 内置了几个常用 reducer：

| Reducer | 行为 | 典型用途 |
|---------|------|---------|
| `add_messages` | 追加消息到列表，相同 ID 的消息会更新 | 对话历史 |
| 自定义 reducer（如 `operator.add`） | 两个值相加 | 计数器、分数累加 |
| 无 reducer（只有类型） | 新值直接覆盖旧值 | 当前状态、最新配置 |

```python
import operator
from typing import Annotated, TypedDict

class State(TypedDict):
    # 有 reducer：多个写入会合并
    messages: Annotated[list, add_messages]  # 追加消息
    scores: Annotated[list, operator.add]    # 追加分数
    
    # 无 reducer：新值覆盖旧值
    current_topic: str  # 只保留最新值
    retry_count: int    # 只保留最新值
```

### Pydantic 模型状态

除了 `TypedDict`，你还可以用 Pydantic 模型定义状态：

```python
from pydantic import BaseModel, Field
from typing import Annotated
from langgraph.graph.message import add_messages

class State(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    context: str = ""
    score: float = 0.0
    retry_count: int = 0
```

Pydantic 模型的优势：

- **运行时验证**：传入的状态数据会被自动验证，类型不对会立即报错。
- **默认值**：可以为每个字段定义默认值，减少初始化代码。
- **序列化**：Pydantic 模型自带 `model_dump()` 和 `model_dump_json()`，方便序列化。

选择 TypedDict 还是 Pydantic 模型？如果你的图很简单，TypedDict 足够且更轻量。如果你需要严格的运行时验证或复杂的默认值，用 Pydantic。

### 输入/输出模式

默认情况下，图的输入和输出都使用同一个 State 类型。但有时你想区分它们：

```python
from typing import TypedDict

class InputState(TypedDict):
    """用户只需要提供这些字段"""
    user_message: str

class OutputState(TypedDict):
    """用户只需要看到这些字段"""
    assistant_reply: str

class InternalState(TypedDict):
    """内部状态——包含用户不需要看到的中间数据"""
    user_message: str
    assistant_reply: str
    internal_context: str
    retry_count: int

# 构建图时指定输入输出模式
graph = StateGraph(
    InternalState,
    input=InputState,
    output=OutputState,
)
```

这样做的效果是：

- **输入模式**过滤了用户可以提供的字段——用户只需要传 `user_message`，不需要关心内部字段。
- **输出模式**过滤了用户可以看到的字段——用户只看到 `assistant_reply`，看不到内部的 `internal_context` 和 `retry_count`。
- **内部状态**在图的执行过程中完整保留，只是对外接口做了简化。

输入/输出模式是一种**封装**手段——让图的接口更简洁，同时保持内部状态的完整性。

## 3.2 Node：处理状态的函数

节点是图中的"工人"——它们读取状态、执行计算、返回状态更新。

### 节点签名：State → Partial[State]

节点函数的签名非常简单：

```python
def my_node(state: State) -> dict:
    # 读取状态
    current_value = state["some_key"]
    
    # 执行计算
    new_value = do_something(current_value)
    
    # 返回部分状态更新——只需要写你想更新的键
    return {"some_key": new_value}
```

关键点：**节点返回的不是完整的状态，而是需要更新的部分。** 这就是 `Partial[State]` 的含义——你只需要返回变化的部分，LangGraph 会自动合并到完整状态中。

```python
def researcher(state: State) -> dict:
    # 只更新 messages 和 current_topic，其他键不变
    return {
        "messages": [{"role": "assistant", "content": "研究发现..."}],
        "current_topic": "AI 安全"
    }

def evaluator(state: State) -> dict:
    # 只更新 score，其他键不变
    return {"score": 0.92}
```

这种"部分更新"设计有三个好处：

1. **简洁**——不需要写完整的状态对象，只写变化的部分。
2. **并行安全**——当多个节点并行执行时，它们各自只更新自己负责的键，不会互相干扰。
3. **可追踪**——你可以清楚地看到每个节点"改变了什么"。

### 节点可以接收的额外参数

除了状态，节点函数还可以接收一些额外的参数：

```python
from langgraph.config import get_config
from langgraph.types import StreamWriter

def my_node(
    state: State,                  # 必须的第一个参数
    config: RunnableConfig,        # 运行配置（可选）
    writer: StreamWriter,          # 流式输出写入器（可选）
    store: BaseStore,              # 长期存储（可选）
) -> dict:
    # config 包含运行时配置
    thread_id = config["configurable"]["thread_id"]
    
    # writer 用于流式输出中间结果
    writer({"type": "progress", "message": "正在处理..."})
    
    # store 用于跨会话的长期存储
    memory = store.search(thread_id, query=state["current_topic"])
    
    return {"result": "处理完成"}
```

- **config**：包含运行时配置，如 `thread_id`、回调函数等。通过 `get_config()` 也可以在函数内部获取。
- **writer**：流式输出接口，允许节点在执行过程中向客户端推送中间结果，而不需要等到整个图执行完成。
- **store**：跨会话的长期记忆存储，允许节点在不同会话之间共享信息。

### 节点的返回值

节点必须返回一个字典，包含要更新的状态键。但有几个特殊情况：

**返回空字典**——如果节点不需要更新任何状态，可以返回空字典 `{}`。这通常用于"决策节点"——它们通过条件边影响流程，但不修改状态：

```python
def route_decision(state: State) -> dict:
    # 这个节点不修改状态，只是触发条件边的路由逻辑
    return {}
```

**返回 None 或不返回**——这是错误的，节点必须返回一个字典。

**使用 Send 动态路由**——节点可以返回 `Send` 对象来动态决定下一个执行哪些节点：

```python
from langgraph.types import Send

def fan_out_node(state: State) -> list[Send]:
    topics = state["topics"]
    return [Send("researcher", {"topic": t}) for t in topics]
```

## 3.3 Edge：节点之间的连接

边定义了节点之间的转移关系——执行完节点 A 之后，接下来执行哪个节点？

### 普通边：add_edge("A", "B")

最简单的边类型，表示"执行完 A 之后，一定执行 B"：

```python
from langgraph.graph import START, END

graph.add_edge(START, "chatbot")    # 从起点到 chatbot
graph.add_edge("chatbot", "evaluator")  # chatbot 之后一定走 evaluator
graph.add_edge("evaluator", END)    # evaluator 之后到终点
```

普通边定义的是**确定性的、无条件**的转移。流程到这里，一定会走到下一个节点。

### 条件边：add_conditional_edges("A", route_function)

条件边表示"执行完 A 之后，根据某个条件决定走哪条路径"：

```python
def should_continue(state: State) -> str:
    """根据状态决定下一步走哪条边"""
    if state["retry_count"] >= 3:
        return "give_up"
    if state["score"] > 0.8:
        return "success"
    return "retry"

graph.add_conditional_edges(
    "evaluator",           # 源节点
    should_continue,      # 路由函数，返回目标节点名
    {
        "give_up": "give_up_node",    # 路由函数返回 "give_up" 时走这个节点
        "success": END,               # 路由函数返回 "success" 时到终点
        "retry": "chatbot",           # 路由函数返回 "retry" 时回到 chatbot
    }
)
```

路由函数接收当前状态作为参数，返回一个字符串，表示下一个要执行的节点名称。可选的第三个参数是一个映射字典，将路由函数的返回值映射到实际的节点名。

**条件边是 LangGraph 表达复杂逻辑的核心机制。** 它让你可以在运行时动态决定流程走向，而不是在编译时固定。

### 等待边：fan-in 模式

当多个节点需要并行执行，然后汇聚到一个节点时，你使用"等待边"——目标节点会等到所有源节点都完成后才执行：

```python
# 三个研究节点并行执行，然后汇总
graph.add_edge(START, "researcher_a")
graph.add_edge(START, "researcher_b")
graph.add_edge(START, "researcher_c")

# 汇总节点等待所有研究节点完成
graph.add_edge(["researcher_a", "researcher_b", "researcher_c"], "summarizer")
graph.add_edge("summarizer", END)
```

这种模式叫做 **fan-in**：多个并行的执行流汇聚到一个节点。LangGraph 会自动处理等待逻辑——`summarizer` 节点只有在三个 `researcher` 都完成之后才会执行。

**合并状态时的冲突**：当三个 `researcher` 同时写入同一个键（如 `messages`），LangGraph 使用 reducer 来合并这些写入。这就是为什么 `Annotated[list, add_messages]` 如此重要——它确保多个并行节点的输出可以被正确合并。

### START 和 END 常量

- **START**：图的入口点。每个图必须至少有一条从 START 出发的边，否则图无法启动。
- **END**：图的出口点。当执行流到达 END 时，图执行完毕，返回最终状态。

```python
# 合法的图结构
graph.add_edge(START, "node_a")      # 必须有从 START 出发的边
graph.add_edge("node_z", END)        # 必须有到达 END 的边

# 不合法的图结构
# graph.add_edge("node_a", "node_b")  # 如果没有从 START 出发的边，编译会报错
```

**隐式 END**：如果一个节点没有定义任何出边，LangGraph 会隐式地认为它连接到 END。但显式添加到 END 的边更清晰，推荐显式声明。

## 3.4 Channel：状态的底层存储

在前面的内容中，我们多次提到了"通道"（Channel）——现在让我们深入理解它。

State 的每个键背后都是一个 Channel。Channel 定义了三件事：

1. **值类型**——这个通道存储什么类型的数据。
2. **更新策略**——当多个节点同时写入时，如何合并。
3. **生命周期**——值在什么时候被清除。

LangGraph 提供了几种内置 Channel：

### LastValue：单值替换

最简单的通道类型。新值直接替换旧值，不合并。这就是没有 reducer 的状态键的默认行为。

```python
from langgraph.channels import LastValue

# 以下两种写法等价：
class State(TypedDict):
    current_topic: str  # 隐式使用 LastValue 通道

# 当两个节点同时写入 current_topic 时，后写入的值覆盖先写入的值
# 注意：并行写入同一 LastValue 通道会导致不确定性行为，应该避免
```

**使用场景**：配置信息、当前任务描述、重试计数器等——任何"只需要最新值"的场景。

### BinaryOperatorAggregate：reducer 聚合

当状态键使用 `Annotated[type, reducer]` 时，背后使用的是 `BinaryOperatorAggregate` 通道。它通过 reducer 函数来合并多个写入：

```python
from typing import Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, add_messages]    # 追加合并
    scores: Annotated[list, operator.add]         # 追加合并
    total: Annotated[int, operator.add]            # 加法合并
```

**合并过程**：

```
超步 N 开始：
  state["total"] = 10

节点 A 写入 total: 5
节点 B 写入 total: 3

超步 N 结束合并：
  state["total"] = reducer(10, 5) = 15  # 先合并 A
  state["total"] = reducer(15, 3) = 18  # 再合并 B
  
最终 state["total"] = 18
```

**使用场景**：对话历史（消息追加）、分数累加、日志收集等——任何"需要保留历史"或"需要聚合多个来源"的场景。

### EphemeralValue：一步有效

这种通道的值在超步结束后就会被清除。它用于"只需要在当前超步中传递，不需要持久保存"的临时数据。

```python
from langgraph.channels import EphemeralValue

class State(TypedDict):
    # 这个值在下一个超步开始时会被清除
    temp_flag: EphemeralValue[str]
```

**使用场景**：临时标记、一次性的路由信号等——任何"用完即弃"的数据。

### Topic：fan-out 用 Send

Topic 通道是 `Send` 功能的底层支撑。当你使用 `Send` 动态发送数据给多个节点实例时，每个实例收到自己的输入数据，这些数据通过 Topic 通道传递。

```python
from langgraph.types import Send

def route(state: State) -> list[Send]:
    # 动态创建多个并行任务
    return [
        Send("worker", {"task": "任务A"}),
        Send("worker", {"task": "任务B"}),
        Send("worker", {"task": "任务C"}),
    ]

graph.add_node("worker", worker_fn)
graph.add_conditional_edges("dispatcher", route)
```

**使用场景**：动态并行执行——当你不知道运行时需要多少个并行实例时（比如根据用户输入决定并行研究几个主题）。

### 通道总结

| 通道类型 | 合并策略 | 生命周期 | 典型用途 |
|---------|---------|---------|---------|
| LastValue | 覆盖 | 持久 | 最新配置、当前状态 |
| BinaryOperatorAggregate | reducer 合并 | 持久 | 消息列表、分数累加 |
| EphemeralValue | 覆盖 | 一步有效 | 临时标记、路由信号 |
| Topic | 一对一分发 | 一步有效 | Send 动态路由 |

**理解通道的关键**：在并行执行场景中，通道的合并策略决定了多个节点同时写入同一个键时会发生什么。选择正确的通道类型，可以避免并行写入冲突，确保状态的一致性。

## 3.5 Checkpoint：状态快照

检查点是 LangGraph 实现"可中断"和"可恢复"的基石。

### thread_id：会话标识

每个检查点通过 `thread_id` 来标识它属于哪个会话（对话、工作流实例等）：

```python
# 同一个 thread_id 的调用共享检查点历史
config = {"configurable": {"thread_id": "user-123-conversation-1"}}

# 第一次调用——保存检查点
result1 = app.invoke({"messages": [("user", "你好")]}, config)

# 第二次调用——从上次检查点继续
result2 = app.invoke({"messages": [("user", "帮我查天气")]}, config)

# 不同的 thread_id 是独立的会话
config2 = {"configurable": {"thread_id": "user-123-conversation-2"}}
result3 = app.invoke({"messages": [("user", "你好")]}, config2)
```

`thread_id` 是你管理多用户、多对话场景的核心机制。每个 `thread_id` 对应一条完整的检查点链。

### 检查点保存时机

检查点在每个**超步结束时**自动保存。这意味着：

```python
# 假设图结构是：START → A → B → C → END

result = app.invoke(input, config)

# 执行过程中，检查点保存的时机：
# 超步 1：执行 START → A，保存检查点 #1
# 超步 2：执行 A → B，保存检查点 #2
# 超步 3：执行 B → C，保存检查点 #3
# 超步 4：执行 C → END，保存检查点 #4
```

每次检查点保存了完整的 State 快照，包括：

- 所有通道的当前值
- 当前执行到哪个节点
- 下一个要执行的节点
- 元数据（时间戳、步骤编号等）

### 状态恢复：从检查点继续

检查点的核心价值在于**状态恢复**——你可以从任意检查点继续执行：

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

# 创建带检查点的图
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "my-conversation"}}

# 第一次调用
result1 = app.invoke(
    {"messages": [("user", "帮我分析这段代码")]},
    config
)
# 结果：助手分析了代码，并在某个节点暂停等待人工确认

# 人工确认后，继续执行
result2 = app.invoke(
    {"messages": [("user", "确认，继续")]},
    config  # 相同的 thread_id，自动从上次检查点继续
)
```

**检查点存储后端**：LangGraph 提供了多种检查点存储后端：

| 后端 | 用途 | 持久性 |
|------|------|--------|
| `MemorySaver` | 开发和测试 | 进程内存在，重启丢失 |
| `SqliteSaver` | 本地部署 | SQLite 文件，持久 |
| `PostgresSaver` | 生产环境 | PostgreSQL 数据库，高可用 |

```python
# 内存检查点——开发时用
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# SQLite 检查点——本地持久化
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# PostgreSQL 检查点——生产环境
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(DATABASE_URL)
```

### 检查点与中断的配合

检查点和中断（interrupt）是 LangGraph 实现人机协作的黄金组合：

```python
from langgraph.graph import StateGraph, START, END

def human_review_node(state: State) -> dict:
    # 这个节点什么都不做——它只是一个中断点
    return {}

# 构建图时，在 human_review 节点前设置中断
graph = StateGraph(State)
graph.add_node("ai_generate", ai_generate_fn)
graph.add_node("human_review", human_review_fn)
graph.add_node("final_output", final_output_fn)

graph.add_edge(START, "ai_generate")
graph.add_edge("ai_generate", "human_review")
graph.add_edge("human_review", "final_output")
graph.add_edge("final_output", END)

# 编译时设置中断点
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_review"]  # 在 human_review 执行前中断
)

# 第一次调用——执行到 human_review 前中断
result1 = app.invoke({"messages": [...]}, config)
# result1 包含 ai_generate 的输出，但 human_review 还没执行

# 人工审核后，继续执行
result2 = app.invoke(None, config)  # 传 None 表示继续执行
# result2 包含 human_review 和 final_output 的输出
```

**工作原理**：

1. 执行到 `human_review` 之前，LangGraph 保存检查点并暂停。
2. 应用程序收到中断信号，把 AI 生成的结果展示给人类审核者。
3. 人类审核者做出决定后，应用程序再次调用 `app.invoke(None, config)`。
4. LangGraph 从检查点恢复，继续执行 `human_review` 和后续节点。

这就是检查点的真正威力——它让 LLM 工作流可以像数据库事务一样，在任意点暂停和恢复，实现真正的人机协作。

## 3.6 概念全图

让我们把所有核心概念串起来，看看一个完整的图是如何工作的：

```
┌──────────────────────────────────────────────────────────────────────┐
│                           StateGraph                                  │
│                                                                      │
│  State (共享数据)                                                     │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  messages: Annotated[list, add_messages]  ← Channel         │     │
│  │  current_topic: str                       ← LastValue      │     │
│  │  scores: Annotated[list, operator.add]    ← Channel         │     │
│  │  retry_count: int                         ← LastValue      │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  Nodes (处理函数)        Edges (转移关系)                             │
│  ┌──────────┐          ┌──────────────────┐                         │
│  │ chatbot  │ ←─────── │ START            │                         │
│  └────┬─────┘          └──────────────────┘                         │
│       │                                                            │
│       │  conditional_edge(route_by_topic)                          │
│       ├──→ "research" → researcher ──┐                            │
│       ├──→ "code"      → coder ──────┤ fan-in                     │
│       └──→ "general"    → responder ─┘                            │
│                                      │                            │
│                              ┌───────┴───────┐                    │
│                              │  synthesizer  │                     │
│                              └───────┬───────┘                    │
│                                      │                            │
│                              ┌───────┴───────┐                    │
│                              │     END       │                     │
│                              └───────────────┘                    │
│                                                                      │
│  Checkpoint (每个超步保存)                                            │
│  ┌─────────────────────────────────────┐                            │
│  │  Step 1: {state after chatbot}      │                            │
│  │  Step 2: {state after researcher}   │                            │
│  │  Step 3: {state after synthesizer}  │                            │
│  └─────────────────────────────────────┘                            │
└──────────────────────────────────────────────────────────────────────┘
```

**执行流程**：

1. 用户输入通过 `START` 进入图。
2. `chatbot` 节点读取 State，执行计算，返回部分状态更新。
3. 条件边根据 `current_topic` 决定走 `researcher`、`coder` 还是 `responder`。
4. 并行节点执行完毕后，通过 Channel 的 reducer 合并写入 State。
5. 超步结束，保存检查点。
6. `synthesizer` 节点读取合并后的 State，执行汇总。
7. 超步结束，再次保存检查点。
8. 到达 `END`，返回最终 State。

**核心概念之间的关系**：

- **State** 是数据，**Node** 是计算，**Edge** 是控制流——三者共同定义了图的逻辑。
- **Channel** 是 State 的底层实现——它决定了数据的合并策略和生命周期。
- **Checkpoint** 是执行过程的快照——它让图可以在任意点暂停和恢复。
- **BSP 模型**是执行框架——它定义了 Node 何时执行、Channel 何时合并、Checkpoint 何时保存。

理解了这五个核心概念及其关系，你就掌握了 LangGraph 的全部基础。在后续章节中，我们将基于这些概念构建更复杂的工作流——子图、人机协作、多 Agent 协同等。