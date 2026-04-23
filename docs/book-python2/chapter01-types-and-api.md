# 第一章：核心类型与公共 API — types.py, constants.py, errors.py

## 1. 功能概览

LangGraph 的公共 API 表面主要由三个文件定义：`types.py` 提供用户直接交互的所有类型与函数，`constants.py` 定义图结构的保留字常量，`errors.py` 声明框架可能抛出的异常层次。

从用户视角看，这三个文件构成了 LangGraph 的"契约层"——它们回答了三个核心问题：

- **我能用什么？** `Send`、`Command`、`interrupt()`、`Overwrite`、`StreamMode`、`RetryPolicy`、`CachePolicy`、`GraphOutput` 等类型和函数。
- **图怎么接？** `START`、`END` 是虚拟节点标记，决定了边连接的入口和出口；`TAG_NOSTREAM` 和 `TAG_HIDDEN` 控制运行时行为。
- **出了什么错？** `GraphRecursionError`、`InvalidUpdateError`、`GraphInterrupt` 等异常类型，让用户能精确捕获和处理不同类别的失败。

这三个文件不包含业务逻辑，但它们是理解整个框架行为的入口。本章将逐一拆解每个定义，揭示其设计意图和内部机制。

## 2. 应用场景

### 2.1 用 Send 实现 Map-Reduce 工作流

当你的图需要对一组数据并行处理、再汇总结果时，`Send` 是核心原语。它允许条件边返回多个 `Send` 对象，每个对象以独立状态触发同一个节点——这就是经典的 map-reduce 模式。

```python
from typing import Annotated, TypedDict
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class OverallState(TypedDict):
    documents: list[str]          # 待处理的文档列表
    summaries: Annotated[list[str], operator.add]  # 汇总的结果

def route_to_summarizer(state: OverallState):
    """条件边：为每个文档创建一个并行任务"""
    return [Send("summarize", {"doc": doc}) for doc in state["documents"]]

def summarize(state: dict):
    """每个并行任务独立执行"""
    doc = state["doc"]
    return {"summaries": [f"摘要: {doc[:50]}..."]}

builder = StateGraph(OverallState)
builder.add_node("summarize", summarize)
builder.add_conditional_edges(START, route_to_summarizer)
builder.add_edge("summarize", END)
graph = builder.compile()

result = graph.invoke({"documents": ["报告A...", "报告B...", "报告C..."]})
# summaries 将包含三个并行处理的结果
```

关键点：`Send("summarize", {"doc": doc})` 中第二个参数是**独立状态**，不与主图状态共享。这意味着每个并行任务看到的只是 `{"doc": doc}`，而非整个 `OverallState`。汇总阶段通过 `operator.add` 聚合器自动合并所有并行结果。

### 2.2 用 Command 实现状态更新 + 路由 + 中断恢复

`Command` 是 LangGraph 的"三合一"操作原语——一个对象同时完成状态更新、路由控制和中断恢复。这三种操作在传统框架中需要三个独立接口，而 `Command` 将它们统一。

```python
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START

class State(TypedDict):
    query: str
    approved: bool
    result: str

def review_node(state: State):
    # 请求人工审批
    decision = interrupt("请审批此查询：" + state["query"])
    # 用 Command 同时完成：1) 更新状态  2) 路由到下一节点
    return Command(
        update={"approved": decision == "yes"},
        goto="execute" if decision == "yes" else "reject"
    )

builder = StateGraph(State)
builder.add_node("review", review_node)
builder.add_node("execute", lambda s: {"result": f"执行: {s['query']}"})
builder.add_node("reject", lambda s: {"result": "已拒绝"})
builder.add_edge(START, "review")
graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "1"}}
# 第一次运行，图在中断处暂停
graph.invoke({"query": "删除所有数据"}, config)

# 恢复执行：Command(resume=...) 提供人工输入
graph.invoke(Command(resume="no"), config)  # → 路由到 reject
```

`Command` 还支持跨图操作：`Command(graph=Command.PARENT, update=...)` 允许子图修改父图状态，这是实现嵌套工作流中"向上汇报"的关键机制。

### 2.3 用 interrupt() 实现人在环中

`interrupt()` 是 LangGraph 的人机协作原语。它会在节点内抛出 `GraphInterrupt` 异常，暂停图执行，将值传递给客户端。恢复时，图**重新执行**该节点，但 `interrupt()` 不再抛出异常，而是返回客户端提供的恢复值。

```python
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    question: str
    answer: str
    feedback: str

def generate_answer(state: State):
    answer = "根据分析，建议方案 A"
    # 中断请求人工反馈
    feedback = interrupt(f"生成的答案：{answer}\n请确认或修改：")
    return {"answer": answer, "feedback": feedback}

builder = StateGraph(State)
builder.add_node("generate", generate_answer)
builder.add_edge(START, "generate")
builder.add_edge("generate", END)
graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"question": "如何优化系统?"}, config)

# 恢复并提供反馈
result = graph.invoke(Command(resume="修改为方案 B"), config)
# answer = "根据分析，建议方案 A", feedback = "修改为方案 B"
```

节点中可以有多个 `interrupt()` 调用，恢复值按顺序匹配——第一个 `interrupt` 得到第一个恢复值，第二个得到第二个。

### 2.4 StreamMode 的七种选择

LangGraph 提供 7 种流式输出模式，每种对应不同的使用场景：

| StreamMode | 适用场景 | 输出内容 |
|---|---|---|
| `"values"` | 前端实时展示完整状态 | 每步后的完整状态 |
| `"updates"` | 日志/审计追踪 | 节点名和返回的增量更新 |
| `"messages"` | 聊天界面的逐 token 流式 | LLM 的 token 级输出 + 元数据 |
| `"custom"` | 自定义监控面板 | 节点内通过 `StreamWriter` 发出的数据 |
| `"checkpoints"` | 调试/回放 | 检查点创建事件 |
| `"tasks"` | 任务调度监控 | 任务开始/结束事件 |
| `"debug"` | 开发调试 | 等于 `checkpoints + tasks` |

```python
# 聊天场景：逐 token 流式
async for msg, meta in graph.astream(input, stream_mode="messages"):
    print(msg.content, end="", flush=True)

# 审计场景：记录每个节点的输出
async for update in graph.astream(input, stream_mode="updates"):
    print(f"节点 {list(update.keys())[0]} 完成")

# 自定义监控：节点内部发出进度
async for data in graph.astream(input, stream_mode="custom"):
    print(f"进度: {data}")
```

### 2.5 RetryPolicy 和 CachePolicy 调优

**RetryPolicy** 控制节点失败后的重试策略，采用指数退避加抖动的经典算法：

```python
from langgraph.types import RetryPolicy

# 对调用外部 API 的节点使用宽松重试
api_retry = RetryPolicy(
    initial_interval=1.0,    # 首次重试等 1 秒
    backoff_factor=3.0,      # 间隔倍增：1s → 3s → 9s
    max_interval=60.0,       # 最大间隔 60 秒
    max_attempts=5,          # 最多 5 次（含首次执行）
    jitter=True,            # 加入随机抖动防止惊群
    retry_on=[ConnectionError, TimeoutError]  # 仅重试网络错误
)

# 对确定性计算节点禁用重试
no_retry = RetryPolicy(max_attempts=1)
```

**CachePolicy** 缓存节点的计算结果，避免重复调用 LLM：

```python
from langgraph.types import CachePolicy

# LLM 调用缓存 10 分钟
llm_cache = CachePolicy(ttl=600)

# 自定义缓存键（基于查询内容而非完整输入）
def query_cache_key(*args, **kwargs):
    return kwargs.get("query", "")
custom_cache = CachePolicy(key_func=query_cache_key, ttl=3600)
```

### 2.6 GraphOutput vs 原始字典输出

当使用 `version="v2"` 调用 `invoke()`/`ainvoke()` 时，返回值从原始字典升级为 `GraphOutput` 对象，提供类型安全的访问方式：

```python
# v1 风格：原始字典，中断信息混在 __interrupt__ 键中
result = graph.invoke(input)
if "__interrupt__" in result:
    ...

# v2 风格：类型安全的 GraphOutput
result = graph.invoke(input, version="v2")
if result.interrupts:
    for intr in result.interrupts:
        print(f"中断: {intr.value}")
print(result.value)  # 图的输出值
```

### 2.7 用 Overwrite 重置累积状态

当状态通道使用了聚合器（如 `operator.add`），常规更新是追加操作。`Overwrite` 允许绕过聚合器，直接替换整个值：

```python
from typing import Annotated
import operator
from langgraph.types import Overwrite
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    messages: Annotated[list[str], operator.add]

def reset_messages(state: State):
    # 不是追加，而是完全替换
    return {"messages": Overwrite(value=["系统重置"])}

builder = StateGraph(State)
builder.add_node("reset", reset_messages)
builder.add_edge(START, "reset")
builder.add_edge("reset", END)
graph = builder.compile()

result = graph.invoke({"messages": ["旧消息1", "旧消息2"]})
# messages = ["系统重置"]，而非 ["旧消息1", "旧消息2", "系统重置"]
```

## 3. Python 进阶

### 3.1 TypedDict vs dataclass vs NamedTuple 的取舍

LangGraph 在三种数据容器之间做了明确分工：

- **NamedTuple**：用于不可变的轻量记录——`RetryPolicy`、`StateUpdate`、`PregelTask`、`StateSnapshot`。这些类型创建后不需要修改，且需要元组语义（可解包、可哈希）。
- **dataclass(frozen=True, slots=True)**：用于需要默认值、继承或更复杂初始化的对象——`Command`、`CachePolicy`、`Interrupt`、`GraphOutput`。`slots=True` 减少内存开销，`frozen=True` 保证不可变。
- **TypedDict**：用于 JSON 风格的流式数据结构——`TaskPayload`、`CheckpointPayload`、`StreamPart` 系列。这些类型不需要实例化，只用于类型标注。

```python
# NamedTuple：不可变、可哈希、轻量
class RetryPolicy(NamedTuple):
    initial_interval: float = 0.5
    max_attempts: int = 3

# dataclass：默认值、泛型、@property
@dataclass(kw_only=True, slots=True, frozen=True)
class Command(Generic[N]):
    update: Any | None = None
    goto: Send | Sequence[Send | N] | N = ()

# TypedDict：仅类型标注，不实例化
class TaskPayload(TypedDict):
    id: str
    name: str
    input: Any
```

### 3.2 `__slots__` 的使用

`Send`、`Interrupt` 等高频创建的对象使用 `__slots__`：

```python
class Send:
    __slots__ = ("node", "arg")
```

`__slots__` 阻止动态属性创建，减少约 40% 内存占用并加速属性访问。在 Pregel 执行器每步可能创建成百上千个 `Send` 对象的场景下，这个优化有实际意义。

### 3.3 `sys.intern()` 字符串驻留

`constants.py` 中所有常量使用 `sys.intern()` 驻留：

```python
START = sys.intern("__start__")
END = sys.intern("__end__")
TAG_NOSTREAM = sys.intern("nostream")
```

驻留保证相同字符串共享同一内存对象，使 `is` 比较和字典查找从 O(n) 降为 O(1)。在 Pregel 每步遍历成千上万个写操作时，字符串比较的效率至关重要。

### 3.4 Literal 作为轻量级枚举

`StreamMode` 和 `Durability` 使用 `Literal` 而非 `Enum`：

```python
StreamMode = Literal["values", "updates", "checkpoints", "tasks", "debug", "messages", "custom"]
Durability = Literal["sync", "async", "exit"]
```

`Literal` 的优势：与 JSON 字符串直接兼容（无需 `.value`），IDE 自动补全同样有效，类型检查器能进行穷举检查。LangGraph 选择 `Literal` 的核心理由是——这些值在 API 层面就是字符串，没必要引入额外的 `Enum` 转换层。

### 3.5 TypeVar、Generic 与 @final

```python
StateT = TypeVar("StateT")
OutputT = TypeVar("OutputT")
N = TypeVar("N", bound=Hashable)

@dataclass(**_DC_KWARGS)
class Command(Generic[N], ToolOutputMixin): ...

@final
@dataclass(init=False, slots=True)
class Interrupt: ...
```

`Generic[N]` 使 `Command` 能携带类型信息（如 `Command[str]` 表示 `goto` 中的节点名类型）。`@final` 阻止继承，这是实现"密封类"的 Python 方式——`Interrupt` 的行为与 Pregel 引擎深度绑定，子类化可能导致不可预测的行为。

### 3.6 `kw_only`、TypeAliasType、Unpack

```python
_DC_KWARGS = {"kw_only": True, "slots": True, "frozen": True}

@dataclass(**_DC_KWARGS)
class CachePolicy(Generic[KeyFuncT]): ...

StreamPart = TypeAliasType(
    "StreamPart",
    ValuesStreamPart[OutputT] | UpdatesStreamPart | ...,
    type_params=(StateT, OutputT),
)
```

`kw_only=True` 强制所有参数以关键字形式传入，避免位置参数歧义。`TypeAliasType` 创建可在运行时使用的类型别名（比 `TypeAlias` 更强），`type_params` 使其支持泛型。`Unpack[DeprecatedKwargs]` 则用于 `Interrupt.__init__` 的 `**kwargs` 类型标注，在类型层面标记这些参数已弃用。

### 3.7 `from __future__ import annotations` 与 `__all__`

文件顶部的 `from __future__ import annotations` 启用延迟注解求值，使前向引用无需引号。`__all__` 元组严格控制 `from langgraph.types import *` 的导出范围，同时向类型检查器声明公共 API 表面。

### 3.8 `deprecated()` 装饰器

```python
@property
@deprecated("`interrupt_id` is deprecated. Use `id` instead.", category=None)
def interrupt_id(self) -> str:
    return self.id
```

`typing_extensions.deprecated` 是 Python 3.13+ 标准库 `warnings.deprecated` 的回移版本。`category=None` 使其不触发 `DeprecationWarning`（由框架自己的 `warn()` 处理），但 IDE 和类型检查器仍会标记为弃用。

## 4. 代码走读

### 4.1 Send — 动态路由消息包

```python
class Send:
    __slots__ = ("node", "arg")       # 只有两个字段，内存紧凑

    def __init__(self, /, node: str, arg: Any) -> None:
        # / 强制 node 为位置参数，防止误写 Send(node="foo", arg=bar)
        self.node = node              # 目标节点名
        self.arg = arg                # 传递给节点的独立状态

    def __hash__(self) -> int:
        return hash((self.node, self.arg))  # 使 Send 可放入 set/dict

    def __eq__(self, value: object) -> bool:
        return (isinstance(value, Send)
                and self.node == value.node
                and self.arg == value.arg)  # 精确比较，非子类
```

`Send` 的设计极简：两个属性、四个方法。`__slots__` 和手动实现 `__eq__`/`__hash__` 表明这是一个高频创建、需要快速比较的对象。在 map-reduce 场景中，条件边可能返回数百个 `Send`，Pregel 引擎用 `set` 去重，所以哈希和相等性必须高效。

### 4.2 Command — 三合一操作原语

```python
@dataclass(**_DC_KWARGS)  # kw_only=True, slots=True, frozen=True
class Command(Generic[N], ToolOutputMixin):
    graph: str | None = None        # None=当前图, Command.PARENT=父图
    update: Any | None = None       # 状态更新（字典或对象）
    resume: dict[str, Any] | Any | None = None  # 中断恢复值
    goto: Send | Sequence[Send | N] | N = ()    # 路由目标

    def _update_as_tuples(self) -> Sequence[tuple[str, Any]]:
        # 将 update 转换为 (key, value) 元组序列
        if isinstance(self.update, dict):
            return list(self.update.items())          # 字典 → items
        elif isinstance(self.update, (list, tuple)) and all(...):
            return self.update                        # 已经是元组序列
        elif keys := get_cached_annotated_keys(type(self.update)):
            return get_update_as_tuples(self.update, keys)  # dataclass/pydantic
        elif self.update is not None:
            return [("__root__", self.update)]        # 标量值

    PARENT: ClassVar[Literal["__parent__"]] = "__parent__"  # 跨图常量
```

`_update_as_tuples` 是 Pregel 引擎消费 `Command.update` 的桥梁：它将不同格式的更新统一为 `(channel_name, value)` 元组序列。`PARENT` 作为类变量定义跨图通信的保留字。

`ToolOutputMixin` 使 `Command` 可作为 LLM 工具调用的返回值——当 LLM 的工具需要中断执行时，直接返回 `Command(resume=...)` 即可。

### 4.3 Interrupt — 中断信息载体

```python
@final
@dataclass(init=False, slots=True)
class Interrupt:
    value: Any    # 传递给客户端的中断值
    id: str       # 中断标识，用于精确定位恢复

    def __init__(self, value: Any, id: str = _DEFAULT_INTERRUPT_ID, **kwargs):
        self.value = value
        if (ns := kwargs.get("ns", MISSING)) is not MISSING and id == _DEFAULT_INTERRUPT_ID:
            self.id = xxh3_128_hexdigest("|".join(ns).encode())  # 从命名空间生成
        else:
            self.id = id

    @classmethod
    def from_ns(cls, value: Any, ns: str) -> Interrupt:
        return cls(value=value, id=xxh3_128_hexdigest(ns.encode()))
```

`init=False` 覆盖 dataclass 自动生成的 `__init__`，因为需要处理弃用的 `ns` 参数和 xxhash ID 生成逻辑。`@final` 阻止子类化——`Interrupt` 的 ID 生成算法与 Pregel 的检查点命名空间紧密耦合。

### 4.4 interrupt() — 中断函数

```python
def interrupt(value: Any) -> Any:
    conf = get_config()["configurable"]
    scratchpad = conf[CONFIG_KEY_SCRATCHPAD]
    idx = scratchpad.interrupt_counter()     # 递增中断索引

    # 情况1：节点内有多个interrupt，已有对应的resume值
    if scratchpad.resume and idx < len(scratchpad.resume):
        conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
        return scratchpad.resume[idx]

    # 情况2：本次恢复的null_resume值
    v = scratchpad.get_null_resume(True)
    if v is not None:
        scratchpad.resume.append(v)
        conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
        return v

    # 情况3：没有恢复值 → 抛出GraphInterrupt
    raise GraphInterrupt((Interrupt.from_ns(value, conf[CONFIG_KEY_CHECKPOINT_NS]),))
```

这是 interrupt/resume 的核心状态机。`scratchpad` 是 Pregel 为每个任务维护的临时存储，`interrupt_counter` 跟踪当前节点内已调用了几次 `interrupt()`。三种分支分别处理：多中断恢复、单值恢复和首次中断。

### 4.5 StateSnapshot — 状态快照

```python
class StateSnapshot(NamedTuple):
    values: dict[str, Any] | Any         # 通道当前值
    next: tuple[str, ...]               # 下一步待执行的节点
    config: RunnableConfig              # 此快照的配置
    metadata: CheckpointMetadata | None # 元数据
    created_at: str | None             # 创建时间
    parent_config: RunnableConfig | None  # 父检查点配置（用于时间旅行）
    tasks: tuple[PregelTask, ...]       # 当前步的任务列表
    interrupts: tuple[Interrupt, ...]   # 待处理的中断
```

`StateSnapshot` 是 `graph.get_state()` 的返回类型。`parent_config` 支持时间旅行——通过它可以回溯到任意历史检查点。

### 4.6 StreamMode 和 StreamPart 类型

```python
StreamMode = Literal["values", "updates", "checkpoints", "tasks", "debug", "messages", "custom"]

# v2 流式 API 的类型化事件
class ValuesStreamPart(TypedDict, Generic[OutputT]):
    type: Literal["values"]             # 鉴别字段
    ns: tuple[str, ...]                 # 命名空间（子图层级）
    data: OutputT                       # 完整状态
    interrupts: tuple[Interrupt, ...]   # 中断信息

StreamPart = TypeAliasType(
    "StreamPart",
    ValuesStreamPart[OutputT] | UpdatesStreamPart | MessagesStreamPart | ...,
    type_params=(StateT, OutputT),
)
```

v2 流式 API 使用鉴别联合（discriminated union）模式：每个 `StreamPart` 都有 `type` 字段，用户通过 `part["type"]` 窄化类型。`ns` 字段标识事件来源的子图层级，这对嵌套图的调试至关重要。

### 4.7 GraphOutput — 类型安全的输出容器

```python
@dataclass(frozen=True)
class GraphOutput(Generic[OutputT]):
    value: OutputT                          # 图的输出值
    interrupts: tuple[Interrupt, ...] = ()  # 中断列表

    def __getitem__(self, key: str) -> Any:
        # 向后兼容：支持 result["__interrupt__"] 和 dict 键访问
        warn("Accessing GraphOutput via `result[key]` is deprecated. ...")
        if key == _INTERRUPT_KEY:
            return self.interrupts
        if isinstance(self.value, dict):
            return self.value[key]
```

`__getitem__` 和 `__contains__` 方法提供向后兼容，同时发出弃用警告。这确保从 v1 字典输出迁移到 v2 `GraphOutput` 的过程不会破坏现有代码。

### 4.8 Overwrite — 绕过聚合器

```python
@dataclass(slots=True)
class Overwrite:
    value: Any  # 直接写入通道的值
```

仅两个字段，但语义强大：当 `BinaryOperatorAggregate` 通道收到 `Overwrite` 时，跳过聚合操作，直接用 `value` 替换当前值。注意 `Overwrite` 不冻结（非 `frozen=True`），因为 Pregel 引擎需要解包并修改其内容。

### 4.9 constants.py — 保留字常量

```python
START = sys.intern("__start__")   # 图的入口虚拟节点
END = sys.intern("__end__")       # 图的出口虚拟节点
TAG_NOSTREAM = sys.intern("nostream")    # 禁用流式标记
TAG_HIDDEN = sys.intern("langsmith:hidden")  # 隐藏追踪标记

def __getattr__(name: str) -> Any:
    # 模块级 __getattr__ 处理弃用导入
    if name in ["Send", "Interrupt"]:
        warn(f"Importing {name} from langgraph.constants is deprecated. ...")
        from importlib import import_module
        module = import_module("langgraph.types")
        return getattr(module, name)
```

模块级 `__getattr__` 是 PEP 562 引入的特性，允许在属性查找失败时动态处理。LangGraph 用它实现"软迁移"：旧代码 `from langgraph.constants import Send` 仍然可用，但会收到弃用警告。

### 4.10 errors.py — 异常层次

```python
class GraphRecursionError(RecursionError):
    """图超过最大步数限制"""
    pass

class InvalidUpdateError(Exception):
    """无效的通道更新（如 LastValue 收到多个值）"""
    pass

class GraphBubbleUp(Exception):
    """内部基类：需要冒泡到父图的异常"""
    pass

class GraphInterrupt(GraphBubbleUp):
    """节点中断，被根图抑制"""
    def __init__(self, interrupts: Sequence[Interrupt] = ()):
        super().__init__(interrupts)

class ParentCommand(GraphBubbleUp):
    """子图向父图发送 Command"""
    def __init__(self, command: Command):
        super().__init__(command)
```

`GraphBubbleUp` 是关键设计：子图中的中断和父图命令需要"冒泡"到根图处理。`GraphInterrupt` 和 `ParentCommand` 继承自它，Pregel 引擎通过 `isinstance(exc, GraphBubbleUp)` 统一捕获并向上传播。

`ErrorCode` 枚举和 `create_error_message` 函数生成带排查链接的错误消息，指向 LangGraph 官方文档的具体错误页面。

## 5. 实现原理

### 5.1 interrupt/resume 的数据级机制

interrupt/resume 不依赖信号或线程中断，而是通过异常和检查点实现：

1. **首次调用 `interrupt(value)`**：构造 `Interrupt` 对象，抛出 `GraphInterrupt` 异常。
2. **Pregel 捕获异常**：将 `Interrupt` 写入检查点的 `tasks` 列表，持久化当前状态。
3. **客户端恢复**：调用 `graph.invoke(Command(resume="answer"), config)`，Pregel 从检查点恢复。
4. **节点重新执行**：`interrupt()` 发现 `scratchpad.resume` 有值，返回恢复值而非抛出异常。

`scratchpad` 是每个任务的生命周期存储，`interrupt_counter()` 是递增计数器，确保同一节点内的多个 `interrupt()` 按顺序匹配恢复值。

### 5.2 Send 创建 PUSH 任务

`Send` 对象由条件边返回，Pregel 引擎将其转换为 PUSH 类型的任务：

1. 条件边函数返回 `[Send("node_a", state1), Send("node_b", state2)]`。
2. Pregel 在 `_internal._constants.TASKS` 保留键中记录这些 `Send` 对象。
3. 下一步调度时，为每个 `Send` 创建独立的 PUSH 任务（区别于边触发的 PULL 任务）。
4. PUSH 任务的输入是 `Send.arg`，而非从通道读取。

这使得 map-reduce 成为可能：一个条件边可以动态生成任意数量的并行任务。

### 5.3 Command 三合一操作的处理流程

当节点返回 `Command` 时，Pregel 按以下顺序处理：

1. **resume 处理**：如果 `Command.resume` 有值，将其写入 `scratchpad.resume`，供同节点的 `interrupt()` 消费。
2. **update 处理**：`_update_as_tuples()` 将更新转为 `(channel, value)` 元组，写入对应通道。
3. **goto 处理**：`goto` 中的节点名或 `Send` 对象被注册为下一步的任务。

三者在同一步内完成，保证了状态更新和路由的原子性——不存在"更新了但路由没生效"的中间状态。

### 5.4 xxhash 中断 ID 生成

```python
Interrupt.from_ns(value, conf[CONFIG_KEY_CHECKPOINT_NS])
# → Interrupt(value=..., id=xxh3_128_hexdigest(ns.encode()))
```

`xxh3_128_hexdigest` 是 128 位 xxHash，速度远快于 SHA-256。中断 ID 由检查点命名空间（如 `"graph|subgraph:task_id"`）的哈希生成，确保同一层级的中断有唯一且确定性的标识。这对恢复操作至关重要——客户端需要通过 ID 精确定位要恢复的中断。

## 6. 动手实验

### 实验 1：Send 的 Map-Reduce 模式

```python
from typing import Annotated, TypedDict
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class State(TypedDict):
    numbers: list[int]
    results: Annotated[list[int], operator.add]

def scatter(state: State):
    """将每个数字发送到独立任务"""
    return [Send("square", {"n": n}) for n in state["numbers"]]

def square(state: dict):
    n = state["n"]
    print(f"  处理 {n}^2 = {n**2}")
    return {"results": [n ** 2]}

builder = StateGraph(State)
builder.add_node("square", square)
builder.add_conditional_edges(START, scatter)
builder.add_edge("square", END)
graph = builder.compile()

result = graph.invoke({"numbers": [1, 2, 3, 4, 5], "results": []})
print(f"结果: {result['results']}")  # [1, 4, 9, 16, 25]
```

### 实验 2：Command 的状态更新 + 路由 + 中断恢复

```python
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import uuid

class State(TypedDict):
    order: str
    confirmed: bool
    status: str

def confirm_order(state: State):
    approval = interrupt(f"订单确认：{state['order']}，是否继续？(yes/no)")
    if approval == "yes":
        return Command(update={"confirmed": True}, goto="process")
    else:
        return Command(update={"confirmed": False, "status": "已取消"}, goto=END)

def process_order(state: State):
    return {"status": f"已处理: {state['order']}"}

builder = StateGraph(State)
builder.add_node("confirm", confirm_order)
builder.add_node("process", process_order)
builder.add_edge(START, "confirm")
graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# 第一步：触发中断
graph.invoke({"order": "100台服务器", "confirmed": False, "status": "待确认"}, config)
print("图已暂停，等待审批...")

# 第二步：恢复执行
result = graph.invoke(Command(resume="yes"), config)
print(f"最终状态: {result}")
# {'order': '100台服务器', 'confirmed': True, 'status': '已处理: 100台服务器'}
```

### 实验 3：多个 interrupt 的顺序匹配

```python
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import uuid

class State(TypedDict):
    data: str

def multi_interrupt(state: State):
    name = interrupt("请输入姓名：")
    age = interrupt("请输入年龄：")
    return {"data": f"{name}, {age}岁"}

builder = StateGraph(State)
builder.add_node("ask", multi_interrupt)
builder.add_edge(START, "ask")
builder.add_edge("ask", END)
graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# 第一次运行：第一个 interrupt 抛出
graph.invoke({"data": ""}, config)

# 恢复：提供两个值（按顺序匹配）
result = graph.invoke(Command(resume=["张三", "28"]), config)
print(result)  # {'data': '张三, 28岁'}
```

注意：当 `Command.resume` 是列表时，第一个 `interrupt` 得到 `"张三"`，第二个得到 `"28"`。这验证了 `scratchpad.interrupt_counter()` 的顺序匹配机制。