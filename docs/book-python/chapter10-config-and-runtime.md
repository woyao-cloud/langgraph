# 第十章：配置注入与运行时 — config.py, runtime.py, managed/

LangGraph 的图执行涉及多层上下文传播：配置从调用者流向每个节点，运行时状态在异步任务间传递，托管值按需计算。本章深入分析 `config.py`、`runtime.py` 和 `managed/` 模块如何协同工作，构建一个类型安全、异步友好的上下文传播系统。

---

## 1. Python 进阶

### 1.1 `contextvars.ContextVar` — 异步安全的上下文传播

`ContextVar` 是 Python 3.7 引入的异步安全上下文变量。与 `threading.local()` 不同，`ContextVar` 在 `asyncio` 任务间自动隔离——每个 `Task` 获得独立的变量副本。Python 3.11 的 `asyncio.create_task(coro, context=ctx)` 进一步允许显式地传播上下文到子任务，解决了异步场景下上下文丢失的问题。

### 1.2 `Generic` + `TypeVar` — 泛型容器类型安全

`Runtime[ContextT]` 使用 `Generic[ContextT]` 声明泛型类。当用户写 `Runtime[MyContext]` 时，类型检查器知道 `runtime.context` 返回 `MyContext` 而非 `Any`。`bound` 约束（如 `TypeVar("V", bound=ManagedValue)`）限制类型参数必须是指定类的子类。

### 1.3 `dataclass(frozen=True)` — 不可变配置

`ExecutionInfo` 使用 `@dataclass(frozen=True)` 声明不可变数据类。`frozen=True` 使得实例化后无法修改属性（尝试赋值会抛出 `FrozenInstanceError`），并通过 `__hash__` 使实例可哈希。`dataclasses.replace()` 创建修改了部分字段的新实例，遵循不可变模式。

### 1.4 `TypeGuard` — 类型缩窄

`is_managed_value` 返回 `TypeGuard[ManagedValueSpec]`，告诉类型检查器：如果此函数返回 `True`，则参数类型从 `Any` 缩窄为 `ManagedValueSpec`。这比 `isinstance` 更灵活，因为 `ManagedValueSpec` 是一个 `type` 别名，不是运行时可检查的类型。

### 1.5 `Annotated[T, Manager]` — 声明式托管值

`IsLastStep = Annotated[bool, IsLastStepManager]` 将"如何计算此值"的元数据编码在类型标注中。类型检查器只看到 `bool`，而 LangGraph 在运行时通过 `get_origin()/get_args()` 提取 `IsLastStepManager` 并调用其 `get()` 方法。这是类型驱动框架设计的典范。

### 1.6 `@staticmethod` + `@abstractmethod` — 无实例依赖的抽象方法

`ManagedValue.get(scratchpad)` 声明为 `@staticmethod @abstractmethod`，意味着子类必须实现此方法但不需要实例。这与常规 `@abstractmethod` 不同——后者隐含 `self` 参数，而静态方法允许不创建 `IsLastStepManager` 实例就调用 `get()`。

---

## 2. 代码走读

### 2.1 `config.py` — 配置获取与上下文提取

```python
from langchain_core.runnables.config import var_child_runnable_config
from langgraph._internal._constants import CONF, CONFIG_KEY_RUNTIME

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

`get_config()` 是用户在节点内部获取运行时配置的入口。它首先检查 Python 版本——在 3.11 之前的版本中，`asyncio.create_task` 不支持 `context` 参数，`ContextVar` 无法自动传播到子任务。如果检测到异步上下文但在旧版 Python 上运行，抛出运行时错误。

核心机制是通过 `var_child_runnable_config.get()` 从 `ContextVar` 获取当前配置。这个变量由 `_set_config_context()` 在节点执行前设置，确保每个节点能获取到正确的配置快照。

```python
def get_store() -> BaseStore:
    return get_config()[CONF][CONFIG_KEY_RUNTIME].store

def get_stream_writer() -> StreamWriter:
    runtime = get_config()[CONF][CONFIG_KEY_RUNTIME]
    return runtime.stream_writer
```

`get_store()` 和 `get_stream_writer()` 是便捷函数，通过配置链路提取 `Runtime` 对象的属性。`get_config()` 返回 `config` → `config["configurable"]` → `config["configurable"]["__pregel_runtime"]` → `.store` / `.stream_writer`。这三层间接访问体现了"配置即依赖注入容器"的设计思想。

### 2.2 `runtime.py` — 运行时容器与上下文管理

```python
@dataclass(frozen=True, slots=True)
class ExecutionInfo:
    checkpoint_id: str
    checkpoint_ns: str
    task_id: str
    thread_id: str | None = None
    run_id: str | None = None
    node_attempt: int = 1
    node_first_attempt_time: float | None = None

    def patch(self, **overrides: Any) -> ExecutionInfo:
        return replace(self, **overrides)
```

`ExecutionInfo` 是一个 `frozen=True` 的 dataclass，使用 `slots=True`（Python 3.10+）减少内存占用。`patch()` 方法用 `dataclasses.replace()` 创建修改了部分字段的新实例，遵循不可变模式。`node_attempt` 默认为 1（从 1 开始计数），`node_first_attempt_time` 记录首次尝试的时间戳，用于重试场景。

```python
@dataclass(frozen=True, slots=True)
class ServerInfo:
    assistant_id: str
    graph_id: str
    user: BaseUser | None = None
```

`ServerInfo` 封装了 LangGraph Server 注入的元数据。`user` 字段实现了 `BaseUser` 协议（支持属性访问和字典访问两种方式），在开源部署中为 `None`。

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

`Runtime` 是运行时上下文的容器。`ContextT` 泛型参数让类型检查器推断 `context` 的类型。`stream_writer` 默认为空操作函数 `_no_op_stream_writer`——这是一个惰性默认值，避免在没有流式输出需求时创建无用的流写入器。

```python
def merge(self, other: Runtime[ContextT]) -> Runtime[ContextT]:
    return Runtime(
        context=other.context or self.context,
        store=other.store or self.store,
        stream_writer=other.stream_writer
        if other.stream_writer is not _no_op_stream_writer
        else self.stream_writer,
        previous=self.previous if other.previous is None else other.previous,
        execution_info=other.execution_info or self.execution_info,
        server_info=other.server_info or self.server_info,
    )
```

`merge()` 的合并策略值得注意：
- **context/store/execution_info/server_info**：`or` 运算符优先使用 `other` 的值，`None` 时回退到 `self`
- **stream_writer**：`is not _no_op_stream_writer` 判断是否为有效写入器，而非简单的真值检查（因为函数对象总是真值）
- **previous**：`None` 检查而非 `or`，因为 `previous` 可以是 `0` 或空列表等假值但合法的返回值

```python
def override(self, **overrides: Unpack[_RuntimeOverrides[ContextT]]) -> Runtime[ContextT]:
    return replace(self, **overrides)
```

`override()` 使用 `Unpack[_RuntimeOverrides]` 将 TypedDict 作为 `**kwargs` 的类型约束——只允许传入 `Runtime` 已定义的字段。`_RuntimeOverrides[ContextT]` 是一个 `TypedDict(total=False)`，所有字段都是可选的。

```python
DEFAULT_RUNTIME = Runtime(
    context=None,
    store=None,
    stream_writer=_no_op_stream_writer,
    previous=None,
    execution_info=None,
)
```

`DEFAULT_RUNTIME` 是一个模块级单例，提供"什么都不做"的默认运行时。`_no_op_stream_writer` 是一个 `lambda _: None` 的空操作函数，确保在没有显式配置流写入器时调用 `stream_writer()` 不会报错。

```python
def get_runtime(context_schema: type[ContextT] | None = None) -> Runtime[ContextT]:
    runtime = cast(Runtime[ContextT], get_config()[CONF].get(CONFIG_KEY_RUNTIME))
    return runtime
```

`get_runtime()` 用 `cast()` 进行纯类型级别转换，运行时不做任何检查。`context_schema` 参数仅用于类型提示——它不参与运行时逻辑，但告诉类型检查器返回的 `Runtime[ContextT]` 的 `context` 属性是 `ContextT` 类型。

### 2.3 `managed/base.py` — 托管值抽象框架

```python
V = TypeVar("V")

class ManagedValue(ABC, Generic[V]):
    @staticmethod
    @abstractmethod
    def get(scratchpad: PregelScratchpad) -> V: ...

ManagedValueSpec = type[ManagedValue]

def is_managed_value(value: Any) -> TypeGuard[ManagedValueSpec]:
    return isclass(value) and issubclass(value, ManagedValue)

ManagedValueMapping = dict[str, ManagedValueSpec]
```

`ManagedValue` 是一个抽象基类，`Generic[V]` 声明返回值类型。`get()` 方法的 `@staticmethod` 装饰器意味着它不需要实例——调用时直接传 `scratchpad` 参数，而不是通过 `self`。`ManagedValueSpec` 是类型别名，指代 `ManagedValue` 的子类本身（而非子类实例）。

`is_managed_value()` 使用 `TypeGuard` 实现类型缩窄：如果返回 `True`，类型检查器将参数类型从 `Any` 缩窄为 `ManagedValueSpec`。`isclass()` 检查是必需的，因为 `issubclass()` 的第一个参数必须是类。

### 2.4 `managed/is_last_step.py` — 托管值的具体实现

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

`IsLastStepManager.get()` 从 `scratchpad` 读取 `step` 和 `stop`，计算当前是否为最后一步。`step` 是当前步骤编号（从 1 开始），`stop` 是总步数限制。当 `step == stop - 1` 时返回 `True`。

`IsLastStep = Annotated[bool, IsLastStepManager]` 是类型级别的声明——类型检查器看到 `bool`，而 LangGraph 在运行时通过 `get_origin()` 和 `get_args()` 提取 `IsLastStepManager` 并调用其 `get()` 方法。用户在状态模式中写 `is_last_step: IsLastStep`，框架自动注入正确值。

`RemainingStepsManager` 类似，返回剩余步数 `stop - step`。

### 2.5 `_scratchpad.py` — 临时计算便笺

```python
@dataclasses.dataclass(**_DC_KWARGS)
class PregelScratchpad:
    step: int
    stop: int
    call_counter: Callable[[], int]
    interrupt_counter: Callable[[], int]
    get_null_resume: Callable[[bool], Any]
    resume: list[Any]
    subgraph_counter: Callable[[], int]
```

`PregelScratchpad` 是每个任务执行期间的临时存储。它不持久化到检查点——只在当前步骤内有效。`step` 和 `stop` 是核心步进信息，供 `IsLastStepManager` 和 `RemainingStepsManager` 使用。`call_counter`、`interrupt_counter` 和 `subgraph_counter` 是惰性计数器函数，每次调用返回递增的 ID。`resume` 存储恢复值列表。

---

## 3. 运行原理

### 3.1 配置传播全流程

```
graph.invoke(input, config)
       │
       ▼
  Pregel.prepare()
  创建 Runtime 对象，注入 send/read/call/runtime
       │
       ▼
  patch_config(config, configurable={
      "__pregel_send": send_fn,
      "__pregel_read": read_fn,
      "__pregel_runtime": Runtime(store=..., stream_writer=...),
      ...
  })
       │
       ▼
  RunnableCallable.invoke(input, config)
  检查 func_accepts，从 config["configurable"]["__pregel_runtime"]
  提取 store/stream_writer/previous 等注入为 kwargs
       │
       ▼
  用户节点函数接收注入的参数
  def my_node(state, store: BaseStore):
      # store 已自动注入
```

配置注入的关键在于 `Runtime` 对象作为配置值存储在 `config["configurable"]["__pregel_runtime"]` 中。`RunnableCallable` 在初始化时通过 `inspect.signature` 确定函数需要哪些参数，运行时从 `Runtime` 提取对应属性注入。

### 3.2 托管值 vs 通道：两种状态管理方式

```
通道（Channel）：
  ┌──────────────────────────────────────────────────┐
  │ 持久化到检查点                                    │
  │ 节点通过返回值写入                                 │
  │ 下一步读取上一步写入的值                             │
  │ 例：messages: Annotated[list, add]               │
  └──────────────────────────────────────────────────┘

托管值（Managed Value）：
  ┌──────────────────────────────────────────────────┐
  │ 不持久化，每次从 scratchpad/config 动态计算          │
  │ 节点通过函数参数自动注入                              │
  │ 例：IsLastStep = Annotated[bool, IsLastStepManager] │
  └──────────────────────────────────────────────────┘
```

通道和托管值的根本区别在于持久化。通道存储在检查点中，跨步骤持久化，适合累积型状态（如消息列表）。托管值是瞬态的，每次从 `scratchpad` 重新计算，适合派生型状态（如"是否最后一步"）。这种设计避免了将可推导的信息存储到检查点中的冗余。

### 3.3 IsLastStep 计算流程

```
PregelScratchpad(step=3, stop=5)
       │
       ▼
  IsLastStepManager.get(scratchpad)
  scratchpad.step == scratchpad.stop - 1
  3 == 5 - 1 → 3 == 4 → False

  RemainingStepsManager.get(scratchpad)
  scratchpad.stop - scratchpad.step
  5 - 3 → 2
```

当 `step` 达到 `stop - 1`（即倒数第一步）时，`IsLastStep` 为 `True`。`RemainingSteps` 直接计算差值。节点函数可以这样使用：

```python
def my_node(state: State, is_last_step: IsLastStep) -> State:
    if is_last_step:
        return {"response": "I've reached the step limit."}
    # ...
```

### 3.4 ContextVar 在异步边界的行为

```
主任务 (Task A)                    子任务 (Task B)
    │                                    │
    var_child_runnable_config.set(config) │
    │                                    │
    │  ─── asyncio.create_task(coro, ──► │
    │       context=ctx)  ──────────────► │
    │                                    │
    │       [Python 3.11+]               │ var_child_runnable_config.get()
    │       context 自动传播              │ → 正确获取到 config
    │                                    │
    │  ─── asyncio.create_task(coro) ──► │
    │       [Python < 3.11]              │ var_child_runnable_config.get()
    │       context 不传播                │ → RuntimeError!
```

Python 3.11 之前，`asyncio.create_task()` 不接受 `context` 参数，新任务继承当前上下文——但在复杂异步场景中上下文可能丢失。`_runnable.py` 中的 `ASYNCIO_ACCEPTS_CONTEXT = sys.version_info >= (3, 11)` 检查决定了是否使用 `asyncio.create_task(coro, context=context)` 来显式传播上下文。

---

## 4. 实现细节

### 4.1 get_config 的 Python 版本兼容

```python
if sys.version_info < (3, 11):
    try:
        if asyncio.current_task():
            raise RuntimeError(
                "Python 3.11 or later required to use this in an async context"
            )
    except RuntimeError:
        pass
```

这段代码的逻辑微妙：在 Python 3.11 以下，检测到异步任务时应该报错，但 `asyncio.current_task()` 在非异步上下文中抛出 `RuntimeError`，这里用 `except RuntimeError: pass` 吞掉。只有当前确实在异步任务中（`current_task()` 返回非 `None`）时，才真正抛出用户可见的错误。这是一个防御性设计——宁可忽略检测，也不要在非异步场景中误报。

### 4.2 _RuntimeOverrides 的 TypedDict 类型约束

```python
class _RuntimeOverrides(TypedDict, Generic[ContextT], total=False):
    context: ContextT
    store: BaseStore | None
    stream_writer: StreamWriter
    previous: Any
    execution_info: ExecutionInfo
    server_info: ServerInfo | None
```

`_RuntimeOverrides` 用 `TypedDict(total=False)` 声明所有字段可选，配合 `Unpack` 在 `override()` 方法中约束 `**kwargs` 的键名和类型。`Generic[ContextT]` 使 `context` 字段的类型随 `Runtime[ContextT]` 的类型参数变化。这是 TypedDict + Generic 的高级组合，在 Python 3.12+ 才完全支持。

### 4.3 merge() 的 stream_writer 特殊处理

```python
stream_writer=other.stream_writer
if other.stream_writer is not _no_op_stream_writer
else self.stream_writer,
```

`stream_writer` 是一个函数对象，函数总是真值，所以不能用 `or` 运算符。`is not _no_op_stream_writer` 精确判断是否为有效的流写入器——只有当 `other` 显式提供了非默认写入器时才使用 `other` 的值，否则保留 `self` 的。

### 4.4 previous 字段的 None 检查

```python
previous=self.previous if other.previous is None else other.previous,
```

`previous` 用 `is None` 而非 `or`，因为 `previous` 可以是 `0`、`""`、`[]` 等假值但合法的返回值。如果用 `other.previous or self.previous`，当 `other.previous` 为 `0` 时会错误地回退到 `self.previous`。

### 4.5 ManagedValue.get 的静态方法设计

`ManagedValue.get()` 声明为 `@staticmethod` 而非实例方法，原因是托管值的计算不需要状态——它只需要 `scratchpad` 参数。如果用实例方法，框架需要为每次节点执行创建 `IsLastStepManager` 实例，增加不必要的对象分配开销。静态方法允许零实例化的值计算。

### 4.6 PregelScratchpad 的计数器闭包

```python
call_counter: Callable[[], int]
interrupt_counter: Callable[[], int]
subgraph_counter: Callable[[], int]
```

这些字段是闭包函数（非普通整数），因为 Pregel 在准备 scratchpad 时需要为每个计数器维护独立的递增状态。闭包捕获了可变状态（通常是一个 `itertools.count()` 或类似机制），每次调用返回递增的 ID。

---

## 5. 动手实验

### 实验 1：在自定义节点中使用 get_config()

```python
from langgraph.graph import StateGraph, START
from langgraph.config import get_config
from typing_extensions import TypedDict

class State(TypedDict):
    count: int

def my_node(state: State):
    config = get_config()
    # 访问 thread_id（用户提供的配置）
    thread_id = config.get("configurable", {}).get("thread_id")
    # 访问任务 ID（框架注入的配置）
    task_id = config.get("configurable", {}).get("__pregel_task_id")
    print(f"Thread: {thread_id}, Task: {task_id}")
    return {"count": state["count"] + 1}

graph = (
    StateGraph(State)
    .add_node("counter", my_node)
    .add_edge(START, "counter")
    .add_edge("counter", "__end__")
    .compile()
)

result = graph.invoke({"count": 0}, config={"configurable": {"thread_id": "t-001"}})
print(result)  # {"count": 1}
```

### 实验 2：创建自定义托管值

```python
from langgraph.managed.base import ManagedValue
from langgraph._internal._scratchpad import PregelScratchpad
from typing import Annotated

class StepProgressManager(ManagedValue[str]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> str:
        return f"Step {scratchpad.step} of {scratchpad.stop}"

StepProgress = Annotated[str, StepProgressManager]

# 在状态模式中使用
from typing_extensions import TypedDict

class MyState(TypedDict):
    messages: list[str]
    progress: StepProgress  # 自动注入为 "Step N of M"
```

### 实验 3：异步上下文传播验证

```python
import asyncio
from contextvars import ContextVar

# 模拟 ContextVar 的跨任务传播
my_var: ContextVar[str] = ContextVar("my_var")

async def child_task():
    # 在 Python 3.11+ 中，create_task 可以传入 context
    value = my_var.get("default")
    print(f"Child task sees: {value}")

async def main():
    token = my_var.set("hello from parent")
    try:
        # Python 3.11+ 方式
        if hasattr(asyncio, "create_task"):
            ctx = asyncio.current_context() if hasattr(asyncio, "current_context") else None
            task = asyncio.create_task(child_task())
            await task
    finally:
        my_var.reset(token)

asyncio.run(main())
# 输出取决于 Python 版本和上下文传播方式
```

### 实验 4：Runtime.merge() 行为测试

```python
from langgraph.runtime import Runtime, _no_op_stream_writer

def custom_writer(data):
    print(f"Writing: {data}")

# 基础 Runtime
base = Runtime(
    context={"user": "alice"},
    store=None,
    stream_writer=_no_op_stream_writer,
)

# 覆盖 Runtime
override = Runtime(
    context=None,  # 不覆盖 context
    store=None,
    stream_writer=custom_writer,  # 覆盖 stream_writer
    previous="last result",
)

merged = base.merge(override)
print(f"context: {merged.context}")          # {'user': 'alice'} (保留 base)
print(f"stream_writer: {merged.stream_writer}")  # custom_writer (使用 override)
print(f"previous: {merged.previous}")        # 'last result' (使用 override)

# 测试 stream_writer 的 _no_op 检测
override2 = Runtime(stream_writer=_no_op_stream_writer)
merged2 = base.merge(override2)
print(f"stream_writer preserved: {merged2.stream_writer is base.stream_writer}")  # True
```

### 实验 5：ExecutionInfo.patch() 不可变更新

```python
from dataclasses import replace
from langgraph.runtime import ExecutionInfo

info = ExecutionInfo(
    checkpoint_id="cp-001",
    checkpoint_ns="",
    task_id="task-001",
    thread_id="thread-001",
    node_attempt=1,
)

# 使用 patch() 创建修改了部分字段的新实例
patched = info.patch(node_attempt=2, node_first_attempt_time=1714000000.0)

print(f"Original node_attempt: {info.node_attempt}")    # 1 (未改变)
print(f"Patched node_attempt: {patched.node_attempt}")   # 2 (新值)
print(f"Patched is new object: {info is not patched}")   # True

# frozen=True 使得直接赋值抛出异常
try:
    info.node_attempt = 3  # FrozenInstanceError!
except AttributeError as e:
    print(f"Cannot modify frozen dataclass: {e}")
```

### 实验 6：inspect.signature 参数注入模拟

```python
import inspect
from typing import get_type_hints

def my_node(state: dict, config, store, writer):
    """模拟 LangGraph 节点函数。"""
    pass

# 检查函数签名
sig = inspect.signature(my_node)
for name, param in sig.parameters.items():
    print(f"  {name}: kind={param.kind.name}, annotation={param.annotation}")

# 输出:
#   state: kind=POSITIONAL_OR_KEYWORD, annotation=<class 'inspect._empty'>
#   config: kind=POSITIONAL_OR_KEYWORD, annotation=<class 'inspect._empty'>
#   store: kind=POSITIONAL_OR_KEYWORD, annotation=<class 'inspect._empty'>
#   writer: kind=POSITIONAL_OR_KEYWORD, annotation=<class 'inspect._empty'>

# LangGraph 的实际注入逻辑会检查 annotation 是否匹配已知类型
# 如 RunnableConfig, BaseStore, StreamWriter 等
```