# 第十章 配置注入与运行时 — config.py, runtime.py, managed/

## 10.1 功能概览

LangGraph 的节点函数不是孤立运行的——它们需要访问线程 ID、持久化存储、流式输出通道等运行时资源。这些资源不是通过参数逐层传递，而是通过一套精心设计的上下文注入机制自动送达节点内部。本章涉及四个核心模块：

| 模块 | 职责 |
|------|------|
| `config.py` | `get_config()`、`get_store()`、`get_stream_writer()`——从 ContextVar 读取运行时配置 |
| `runtime.py` | `Runtime[T]` 类——捆绑 context/store/writer/previous/execution_info 的容器 |
| `managed/base.py` | `ManagedValue` 抽象基类——"按需计算、不持久化"的状态值声明模式 |
| `managed/is_last_step.py` | `IsLastStep`/`RemainingSteps`——检测递归限制边界的托管值 |

核心设计思想：**配置通过 ContextVar 隐式传播，托管值通过 Annotated 声明式注入**。开发者只需在节点函数签名中声明所需依赖，框架自动注入。

## 10.2 应用场景

### 场景一：用 get_config() 获取 thread_id 和运行时信息

在聊天机器人、多轮对话等场景中，thread_id 是区分不同会话的关键标识。你可以用 `get_config()` 在节点内部获取它，实现会话级隔离：

```python
from langgraph.config import get_config
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict

class State(TypedDict):
    message: str

def chat_node(state: State):
    config = get_config()
    thread_id = config["configurable"]["thread_id"]
    checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

    # 根据 thread_id 加载会话历史、实现会话级缓存
    print(f"Processing thread: {thread_id}, namespace: {checkpoint_ns}")
    return {"message": f"Hello from thread {thread_id}"}

graph = (
    StateGraph(State)
    .add_node("chat", chat_node)
    .add_edge(START, "chat")
    .set_finish_point("chat")
    .compile()
)

# 不同 thread_id 代表不同会话
graph.invoke({"message": "hi"}, config={"configurable": {"thread_id": "session-001"}})
graph.invoke({"message": "hi"}, config={"configurable": {"thread_id": "session-002"}})
```

### 场景二：用 get_stream_writer() 输出自定义流数据

在长时间运行的任务中，用户需要实时反馈——进度条、中间结果、日志信息。`StreamWriter` 让你在节点执行过程中向客户端推送自定义数据，而不必等到节点完成：

```python
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict
import time

class State(TypedDict):
    result: str

def analyze_node(state: State):
    writer = get_stream_writer()

    steps = ["loading_data", "preprocessing", "modeling", "generating_report"]
    for i, step in enumerate(steps):
        # 推送进度信息
        writer({"progress": (i + 1) / len(steps), "current_step": step})
        time.sleep(0.5)  # 模拟耗时操作

    return {"result": "Analysis complete"}

graph = (
    StateGraph(State)
    .add_node("analyze", analyze_node)
    .add_edge(START, "analyze")
    .set_finish_point("analyze")
    .compile()
)

# 使用 stream_mode="custom" 接收自定义流数据
for chunk in graph.stream({"result": ""}, stream_mode="custom"):
    print(chunk)
# {"progress": 0.25, "current_step": "loading_data"}
# {"progress": 0.5, "current_step": "preprocessing"}
# {"progress": 0.75, "current_step": "modeling"}
# {"progress": 1.0, "current_step": "generating_report"}
```

### 场景三：用 Runtime.context 传递运行作用域的不可变数据

在多租户系统、用户个性化服务中，每个图运行需要关联一个用户身份、数据库连接或请求上下文。`Runtime.context` 提供了类型安全的运行作用域数据注入：

```python
from dataclasses import dataclass
from langgraph.runtime import Runtime, get_runtime
from langgraph.graph import StateGraph, START
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

@dataclass
class AppContext:
    user_id: str
    tenant_id: str
    db_name: str

class State(TypedDict):
    response: str

store = InMemoryStore()

def personalized_node(state: State, runtime: Runtime[AppContext]):
    user_id = runtime.context.user_id
    tenant_id = runtime.context.tenant_id

    # 从 store 加载用户偏好
    if runtime.store:
        user_data = runtime.store.get(("preferences", tenant_id), user_id)
        preferences = user_data.value if user_data else {}

    return {"response": f"Hello {user_id} from tenant {tenant_id}"}

graph = (
    StateGraph(state_schema=State, context_schema=AppContext)
    .add_node("personalize", personalized_node)
    .add_edge(START, "personalize")
    .set_finish_point("personalize")
    .compile(store=store)
)

result = graph.invoke(
    {"response": ""},
    context=AppContext(user_id="alice", tenant_id="acme", db_name="prod")
)
```

### 场景四：用 IsLastStep 检测递归限制临近

在具有循环的图中，节点可能被反复执行。当递归步数接近限制时，你需要切换行为——跳过昂贵操作、保存中间结果、或优雅终止。`IsLastStep` 就是为此设计的托管值：

```python
from typing import Annotated
from langgraph.graph import StateGraph, START
from langgraph.managed.is_last_step import IsLastStep
from typing_extensions import TypedDict

class State(TypedDict):
    data: str
    is_last: Annotated[bool, IsLastStep]  # 声明托管值

def process_node(state: State):
    if state["is_last"]:
        # 最后一步——跳过耗时操作，直接保存当前进度
        return {"data": state["data"] + " [saved at last step]"}

    # 正常处理
    return {"data": state["data"] + " -> processed"}

graph = (
    StateGraph(State)
    .add_node("process", process_node)
    .add_edge(START, "process")
    .add_edge("process", "process")  # 自循环
    .set_finish_point("process")
    .compile()
)

result = graph.invoke({"data": "start", "is_last": False})
# 当递归步数接近限制时，is_last 自动变为 True
```

### 场景五：用 RemainingSteps 实现自适应行为

`RemainingSteps` 比 `IsLastStep` 更精细——它告诉你还剩多少步，让你根据剩余步数动态调整策略：

```python
from typing import Annotated
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    analysis: str
    steps_left: Annotated[int, RemainingSteps]

def deep_analysis(state: State):
    remaining = state["steps_left"]

    if remaining <= 1:
        return {"analysis": state["analysis"] + " [quick summary]"}
    elif remaining <= 3:
        return {"analysis": state["analysis"] + " [medium analysis]"}
    else:
        # 还有充足步数，执行深度分析
        return {"analysis": state["analysis"] + " [deep analysis]"}

graph = (
    StateGraph(State)
    .add_node("analyze", deep_analysis)
    .add_edge(START, "analyze")
    .add_conditional_edges("analyze", lambda s: "analyze" if s["steps_left"] > 1 else END)
    .compile()
)
```

### 场景六：托管值与普通 State 键的区别

理解这个区别至关重要：托管值（如 `IsLastStep`）**不持久化**到 checkpoint，它们在每一步都从 scratchpad 重新计算。这意味着：

1. 中断（interrupt）恢复后，托管值自动反映恢复时的真实步数
2. 从 checkpoint 恢复时，`IsLastStep` 不会保留中断前的旧值
3. 托管值不会出现在 checkpoint 的 channel_values 中

```python
# 托管值声明方式——通过 Annotated
class State(TypedDict):
    messages: list           # 普通状态键：持久化、跨步保留
    is_last: Annotated[bool, IsLastStep]  # 托管值：不持久化、每步重新计算
```

### 场景七：用 store 实现跨线程持久数据

`store` 与 checkpoint 不同——checkpoint 是线程内的状态快照，而 store 是跨线程的持久化键值存储，适合保存用户画像、文档库、共享配置等：

```python
from langgraph.config import get_store
from langgraph.graph import StateGraph, START
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

class State(TypedDict):
    query: str
    answer: str

store = InMemoryStore()
# 预存用户偏好
store.put(("user_prefs",), "alice", {"language": "zh", "style": "concise"})

def qa_node(state: State):
    s = get_store()
    # 从跨线程 store 读取用户偏好
    pref = s.get(("user_prefs",), "alice")
    lang = pref.value["language"] if pref else "en"

    answer = f"[{lang}] Answer to: {state['query']}"
    return {"answer": answer}

graph = (
    StateGraph(State)
    .add_node("qa", qa_node)
    .add_edge(START, "qa")
    .set_finish_point("qa")
    .compile(store=store)
)

# 不同线程可以访问同一个 store
graph.invoke({"query": "What is AI?", "answer": ""},
             config={"configurable": {"thread_id": "t1"}})
```

## 10.3 Python 进阶

### contextvars.ContextVar 与 ThreadLocal 对比

`ContextVar` 是 Python 3.7 引入的上下文变量，相比 `threading.local` 有两个关键优势：

1. **asyncio 安全**：ContextVar 在协程间正确传播，ThreadLocal 在异步代码中会混乱
2. **显式复制**：`copy_context()` 创建上下文快照，可以安全传递给子任务

```python
import contextvars

# ContextVar 声明
var = contextvars.ContextVar("my_var", default=None)

# 设置值并获取 token
token = var.set("hello")
value = var.get()  # "hello"
var.reset(token)   # 恢复到设置前的状态
```

LangGraph 中 `var_child_runnable_config` 就是一个 `ContextVar[RunnableConfig | None]`，在 `set_config_context` 上下文管理器中设置和恢复。

### Generic 与 bound TypeVar

`Runtime[ContextT]` 使用泛型参数 `ContextT`，让类型检查器知道 `runtime.context` 的具体类型。`ContextT` 在 `langgraph.typing` 中被定义为 `TypeVar("ContextT", bound=StateLike | None, default=None)`，bound 约束确保 context 只能是类 TypedDict 结构或 None。

### dataclass(frozen=True) 与 slots=True

`ExecutionInfo` 和 `ServerInfo` 都使用了 `@dataclass(frozen=True, slots=True)`：
- `frozen=True`：实例不可变，防止意外修改运行时元数据
- `slots=True`：用 `__slots__` 替代 `__dict__`，减少内存占用

### dataclasses.replace()

`ExecutionInfo.patch()` 和 `Runtime.override()` 使用 `dataclasses.replace(self, **overrides)` 创建新实例而非修改原实例——这是不可变模式的标准实现。

### TypeGuard

`is_managed_value(value)` 返回 `TypeGuard[ManagedValueSpec]`，告诉类型检查器当此函数返回 True 时，`value` 的类型可以缩窄为 `ManagedValueSpec`。这比简单的 `isinstance` 更精确，因为它处理的是类类型检查。

### Annotated[T, Manager] 模式

`IsLastStep = Annotated[bool, IsLastStepManager]` 是 LangGraph 最优雅的设计模式之一：用 `Annotated` 的第二个参数声明值的计算方式（Manager），而第一个参数声明值的类型（bool）。编译时，框架扫描 State 的类型注解，发现 `IsLastStepManager` 是 `ManagedValue` 子类后，将其注册为托管值；运行时，每一步调用 `IsLastStepManager.get(scratchpad)` 计算当前值。

### @abstractmethod + ABC

`ManagedValue` 继承 `ABC` 并声明 `get` 为抽象静态方法，强制所有托管值管理器必须实现 `get(scratchpad)` 方法。这是经典模板方法模式的 Python 实现。

### @staticmethod

`IsLastStepManager.get` 和 `RemainingStepsManager.get` 都是 `@staticmethod`，因为它们不需要实例状态——它们只从 scratchpad 参数计算结果。这比 `@classmethod` 更明确：方法不依赖类也不依赖实例。

## 10.4 代码走读

### config.py — 配置访问入口

`get_config()` 是最基础也最常用的函数：

```python
def get_config() -> RunnableConfig:
    if sys.version_info < (3, 11):
        try:
            if asyncio.current_task():
                raise RuntimeError(
                    "Python 3.11 or later required to use this in an async context"
                )
        except RuntimeError:
            pass
    if var_config := var_child_runnable_config.get():
        return var_config
    else:
        raise RuntimeError("Called get_config outside of a runnable context")
```

关键点：Python 3.11 之前，`asyncio.create_task` 不支持 `context` 参数，ContextVar 无法在异步任务间传播。因此旧版 Python 在异步上下文中直接报错。

`get_store()` 一行代码从 config 中提取 store：

```python
def get_store() -> BaseStore:
    return get_config()[CONF][CONFIG_KEY_RUNTIME].store
```

`get_stream_writer()` 同理：

```python
def get_stream_writer() -> StreamWriter:
    runtime = get_config()[CONF][CONFIG_KEY_RUNTIME]
    return runtime.stream_writer
```

两者的共性：通过 `get_config()` 获取当前配置，从 `configurable.__pregel_runtime` 提取 Runtime 对象，再访问其属性。

### runtime.py — 运行时容器

`Runtime` 是一个泛型 dataclass，捆绑了运行时可用的所有资源：

```python
@dataclass(**_DC_KWARGS)
class Runtime(Generic[ContextT]):
    context: ContextT = field(default=None)
    store: BaseStore | None = field(default=None)
    stream_writer: StreamWriter = field(default=_no_op_stream_writer)
    previous: Any = field(default=None)
    execution_info: ExecutionInfo | None = field(default=None)
    server_info: ServerInfo | None = field(default=None)
```

`_no_op_stream_writer` 是默认的空操作 writer——当用户没有使用 `stream_mode="custom"` 时，写入操作静默忽略，避免节点代码需要条件判断。

`merge(other)` 方法实现两个 Runtime 的合并，后者覆盖前者：

```python
def merge(self, other: Runtime[ContextT]) -> Runtime[ContextT]:
    return Runtime(
        context=other.context or self.context,
        store=other.store or self.store,
        stream_writer=other.stream_writer if other.stream_writer is not _no_op_stream_writer else self.stream_writer,
        previous=self.previous if other.previous is None else other.previous,
        execution_info=other.execution_info or self.execution_info,
        server_info=other.server_info or self.server_info,
    )
```

注意 `stream_writer` 的合并逻辑：只有当 `other.stream_writer` 不是默认的 `_no_op_stream_writer` 时才覆盖——通过 `is` 判断哨兵值。

`get_runtime(context_schema)` 是获取当前 Runtime 的便捷函数，内部实现与 `get_config()` 一致：

```python
def get_runtime(context_schema=None) -> Runtime[ContextT]:
    runtime = cast(Runtime[ContextT], get_config()[CONF].get(CONFIG_KEY_RUNTIME))
    return runtime
```

`ExecutionInfo` 是 frozen dataclass，提供只读的执行元数据（checkpoint_id、task_id、thread_id、node_attempt 等）。`patch(**overrides)` 方法用 `dataclasses.replace` 创建修改后的新实例。

### managed/base.py — 托管值抽象

```python
class ManagedValue(ABC, Generic[V]):
    @staticmethod
    @abstractmethod
    def get(scratchpad: PregelScratchpad) -> V: ...

ManagedValueSpec = type[ManagedValue]  # ManagedValue 的类类型

def is_managed_value(value: Any) -> TypeGuard[ManagedValueSpec]:
    return isclass(value) and issubclass(value, ManagedValue)

ManagedValueMapping = dict[str, ManagedValueSpec]
```

`ManagedValue` 的设计极简：只有一个抽象静态方法 `get(scratchpad)`。所有托管值管理器只需实现这个方法，从 scratchpad 计算出当前值。`is_managed_value` 作为类型守卫函数，在编译时被用来判断 `Annotated[T, metadata]` 中的 metadata 是否是托管值。

### managed/is_last_step.py — 步数感知托管值

```python
class IsLastStepManager(ManagedValue[bool]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> bool:
        return scratchpad.step == scratchpad.stop - 1

IsLastStep = Annotated[bool, IsLastStepManager]

class RemainingStepsManager(ManagedValue[int]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> int:
        return scratchpad.stop - scratchpad.step

RemainingSteps = Annotated[int, RemainingStepsManager]
```

`IsLastStepManager.get` 比较当前步数 `step` 与终止步数 `stop - 1`；`RemainingStepsManager.get` 计算剩余步数。`scratchpad` 是 Pregel 在每个 superstep 开始时创建的临时存储，包含当前执行步的元信息。

声明方式是 `Annotated[bool, IsLastStepManager]`——在 State 的类型注解中使用，框架编译时自动识别并注册为托管值通道。

## 10.5 实现原理

### 上下文传播穿越异步边界

LangGraph 的核心挑战之一是在异步执行环境中正确传播 ContextVar。整个传播链路如下：

```
graph.invoke(input, config)
    → Pregel.stream() 调用 ensure_config() 创建完整 config
    → 调用 _set_config_context(config) 设置 var_child_runnable_config
    → 对每个节点调用 RunnableCallable.invoke()
    → RunnableCallable 使用 copy_context() 创建上下文快照
    → 在快照中执行 step.invoke(input, config)
    → Python 3.11+: asyncio.create_task(coro, context=ctx) 传播上下文
    → 节点内部调用 get_config() → var_child_runnable_config.get() 返回当前配置
```

Python 3.11 的 `asyncio.create_task(coro, context=ctx)` 是关键改进——它确保新创建的协程任务继承父任务的 ContextVar 快照。3.11 之前，异步任务无法获得正确的 ContextVar 值，因此 `get_config()` 在异步上下文中会报错。

### 托管值从 scratchpad 计算而非持久化

托管值与普通状态键的根本区别在于数据来源：

- **普通状态键**：值存储在 checkpoint 的 channels 中，跨步持久化，通过 reducer 更新
- **托管值**：值不存储在任何 channel 中，每一步都调用 `Manager.get(scratchpad)` 从 Pregel 的执行状态实时计算

当 Pregel 在一个 superstep 中准备任务时，它会创建 `PregelScratchpad` 对象，填入当前步数 `step`、终止步数 `stop`、中断计数器等信息。对于声明了 `Annotated[bool, IsLastStepManager]` 的 State 键，Pregel 调用 `IsLastStepManager.get(scratchpad)` 获取当前值并写入节点的输入状态。节点看到的 `state["is_last"]` 就是这样计算出来的——它从未被写入 checkpoint。

### Annotated[T, Manager] 声明模式的编译时处理

当 `StateGraph(state_schema=MyState)` 被创建时，编译过程会扫描 `MyState` 的所有类型注解：

1. `get_type_hints(MyState, include_extras=True)` 提取所有 Annotated 类型
2. 对每个 `Annotated[T, metadata]`，检查 `metadata` 是否是 `ManagedValue` 子类
3. 如果是，将其注册为托管值通道，在运行时每步调用 `metadata.get(scratchpad)` 计算值
4. 如果不是（如 `Annotated[list, add]` 中的 `add`），将其视为 reducer 函数

这个判断逻辑由 `is_managed_value(metadata)` 完成——一个返回 `TypeGuard` 的辅助函数。

## 10.6 动手实验

### 实验一：在节点中使用 get_config()

```python
from langgraph.config import get_config
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    result: str

def my_node(state: State):
    config = get_config()
    thread_id = config["configurable"].get("thread_id", "unknown")
    checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

    return {"result": f"thread={thread_id}, ns={checkpoint_ns}"}

graph = (
    StateGraph(State)
    .add_node("node", my_node)
    .add_edge(START, "node")
    .set_finish_point("node")
    .compile()
)

# 测试不同 thread_id
result1 = graph.invoke({"result": ""}, config={"configurable": {"thread_id": "t-001"}})
print(result1["result"])  # thread=t-001, ns=
```

### 实验二：创建自定义托管值

```python
from typing import Annotated
from langgraph.managed.base import ManagedValue
from langgraph._internal._scratchpad import PregelScratchpad
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# 自定义托管值：当前步数
class CurrentStepManager(ManagedValue[int]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> int:
        return scratchpad.step

CurrentStep = Annotated[int, CurrentStepManager]

class State(TypedDict):
    value: str
    step_num: Annotated[int, CurrentStep]  # 托管值

def process(state: State):
    return {"value": f"Step {state['step_num']}: processed"}

graph = (
    StateGraph(State)
    .add_node("process", process)
    .add_edge(START, "process")
    .add_conditional_edges("process", lambda s: "process" if s["step_num"] < 3 else END)
    .set_finish_point("process")
    .compile()
)

# step_num 每步自动更新，不需要手动维护
result = graph.invoke({"value": "start", "step_num": 0})
```

### 实验三：测试 ContextVar 上下文传播

```python
import asyncio
import contextvars

my_var = contextvars.ContextVar("my_var", default="unset")

async def inner_task():
    value = my_var.get()
    print(f"Inner task sees: {value}")

async def outer_task():
    token = my_var.set("hello_from_outer")
    try:
        # Python 3.11+: 上下文自动传播
        task = asyncio.create_task(inner_task())
        await task
    finally:
        my_var.reset(token)

asyncio.run(outer_task())
# Python 3.11+: Inner task sees: hello_from_outer
# Python 3.10-: Inner task sees: unset (上下文未传播)
```

### 实验四：Runtime.merge 的行为

```python
from langgraph.runtime import Runtime

runtime1 = Runtime(context={"user_id": "alice"}, store=None)
runtime2 = Runtime(context=None, store="my_store")

merged = runtime1.merge(runtime2)
print(merged.context)   # {'user_id': 'alice'} — 来自 runtime1（runtime2.context 为 None 不覆盖）
print(merged.store)     # 'my_store' — 来自 runtime2

# 测试 override
runtime3 = runtime1.override(context={"user_id": "bob"})
print(runtime3.context)  # {'user_id': 'bob'}
print(runtime1.context)  # {'user_id': 'alice'} — 原实例不变（不可变模式）
```

### 实验五：ExecutionInfo.patch 的不可变更新

```python
from langgraph.runtime import ExecutionInfo

info = ExecutionInfo(
    checkpoint_id="cp-001",
    checkpoint_ns="graph|subgraph",
    task_id="task-001",
    thread_id="thread-001"
)

# 不可变更新
info2 = info.patch(task_id="task-002", node_attempt=2)
print(info2.task_id)        # 'task-002'
print(info2.node_attempt)    # 2
print(info.task_id)          # 'task-001' — 原实例不变
print(info.node_attempt)     # 1
```