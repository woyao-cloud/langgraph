# 第十章 配置与运行时 — config.py, runtime.py, managed/

## Java 桥梁

Java 开发者对 `ThreadLocal<T>` 非常熟悉——它为每个线程保存独立的上下文副本，常用于传递 Request ID、用户身份、数据库连接等。在 Spring 中，`RequestContextHolder` 就是基于 `ThreadLocal` 实现的。

LangGraph 面对的是同样的问题：图的节点函数可能需要访问当前配置、存储后端、流式写入器等运行时依赖，但不想通过参数层层传递。Python 的 `contextvars` 模块正是 `ThreadLocal` 的升级版——它同时支持线程和协程，而 Java 的 `ThreadLocal` 在异步场景下需要手动传递。

本章介绍三个层次的基础设施：`config.py` 提供配置的上下文访问入口，`runtime.py` 封装运行时依赖容器，`managed/` 模块实现一种特殊的"虚拟状态"机制——它看起来像 State 的键，但值是动态计算的，不会持久化到检查点中。

## Python 概念速查

| Python 概念 | Java 对应 | 说明 |
|---|---|---|
| `contextvars.ContextVar` | `ThreadLocal<T>` | 协程安全的上下文变量 |
| `contextvars.copy_context()` | 无直接对应 | 复制整个上下文快照 |
| `Generic[T]` | `Class<T>` 泛型类 | Python 泛型类声明 |
| `dataclass(frozen=True)` | `@Value` / 不可变类 | 编译器生成的不可变数据类 |
| `dataclasses.replace()` | `withX()` 链式调用 | 创建修改部分字段的新实例 |
| `@abstractmethod` + `ABC` | `abstract` 方法 + `abstract class` | 抽象方法和抽象类 |
| `Annotated[type, metadata]` | Java 注解 | 用元数据装饰类型 |
| `Protocol`（结构化子类型） | Java `interface` | 鸭子类型协议 |
| `TypeGuard` | `instanceof` + 类型窄化 | 类型检查器的类型窄化提示 |
| `TypedDict(total=False)` | Lombok `@Builder.Default` | 所有字段可选的结构化字典 |
| `field(default=None)` | 字段默认值 | dataclass 字段默认值 |
| `@staticmethod` | `static` 方法 | 不接收 self 参数的方法 |

## 代码走读

### 10.1 config.py — 配置的上下文访问

`config.py` 是用户侧的入口文件，提供了两个核心函数：`get_config()` 和 `get_stream_writer()`。

**get_config()** 从上下文变量获取当前配置：

```python
from langchain_core.runnables.config import var_child_runnable_config

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

`var_child_runnable_config` 是 LangChain 定义的 `ContextVar[RunnableConfig | None]`。Java 的 `ThreadLocal` 在异步框架（如 WebFlux）中经常出问题——不同线程之间不会自动传播。Python 3.11 的 `asyncio.create_task()` 支持了 `context` 参数，使 `ContextVar` 在协程间自动传播。版本检查代码确保在 3.11 以下版本异步环境中给出明确错误，而不是静默返回 `None` 导致难以调试的 bug。

**get_stream_writer()** 获取流式写入器：

```python
def get_stream_writer() -> StreamWriter:
    runtime = get_config()[CONF][CONFIG_KEY_RUNTIME]
    return runtime.stream_writer
```

通过 `get_config()` → `config["configurable"]["__pregel_runtime"]` → `runtime.stream_writer` 的链式访问获取写入器。Java 开发者可以类比为从 Spring 的 `ApplicationContext` 中按名称获取 Bean——不过这里用的是字典索引而非类型查找。

**get_store()** 获取存储后端：

```python
def get_store() -> BaseStore:
    return get_config()[CONF][CONFIG_KEY_RUNTIME].store
```

这三个函数的共同模式是：通过上下文变量获取配置 → 从配置的 `configurable` 子字典中提取 Runtime 对象 → 访问 Runtime 的属性。这种"魔法"访问在 Java 社区通常不被推荐（倾向于显式注入），但在 Python 生态中，上下文变量 + 约定键名的模式已广为接受。

### 10.2 runtime.py — 运行时容器

`Runtime[T]` 是一个泛型数据类，封装了一次图执行的所有运行时依赖。

**ExecutionInfo** — 执行元信息：

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

`frozen=True` 使实例不可变——类似 Java 的 `record` 或 `@Value` 注解类。`slots=True` 使用 `__slots__` 替代 `__dict__`，节省内存并禁止动态添加属性——Java 类默认就是这种行为（字段固定），Python 默认允许动态添加属性。`replace()` 函数创建修改部分字段的新实例，等价于 Java 的不可变对象模式 `new ExecutionInfo(this.checkpointId, this.checkpointNs, newTaskId, ...)`。

**ServerInfo** — 服务端注入信息：

```python
@dataclass(frozen=True, slots=True)
class ServerInfo:
    assistant_id: str
    graph_id: str
    user: BaseUser | None = None
```

仅在 LangGraph Server（LangSmith 托管版）中非 None，开源部署时为 None。

**Runtime[T]** — 泛型运行时容器：

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

`Generic[ContextT]` 使 `Runtime` 成为泛型类。Java 开发者熟悉 `Runtime<MyContext>` 的写法，但 Python 的泛型是"类型提示层"的——运行时不会强制类型约束。`context` 字段是用户自定义上下文（如 `user_id`、`db_conn`），在 `StateGraph(context_schema=MyContext)` 中声明后，运行时注入。

`merge()` 方法合并两个 Runtime：

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

注意 `stream_writer` 的合并逻辑——用 `_no_op_stream_writer` 函数对象作为哨兵，检查 `is` 而非 `==`。这和 `_typing.py` 中的 `MISSING` 哨兵是同一模式。Java 中通常用 `Objects.equals()` 或 `== null` 判断，Python 中用 `is` 判断哨兵对象身份是标准做法。

`override()` 方法用 `replace()` 创建新实例：

```python
def override(self, **overrides: Unpack[_RuntimeOverrides[ContextT]]) -> Runtime[ContextT]:
    return replace(self, **overrides)
```

`Unpack[_RuntimeOverrides[ContextT]]` 是 Python 3.12+ 的 TypedDict 解包语法，让 `**overrides` 的键名和类型受到约束——类似 Java 的 Builder 模式，但利用类型系统在调用时检查。

**DEFAULT_RUNTIME** 是模块级单例：

```python
DEFAULT_RUNTIME = Runtime(
    context=None, store=None, stream_writer=_no_op_stream_writer,
    previous=None, execution_info=None,
)
```

Python 的模块级变量天然是单例——模块只加载一次。Java 要实现单例需要额外的模式（`enum` 单例、双重检查锁等），Python 则靠解释器保证。

**get_runtime()** 从配置中获取当前 Runtime：

```python
def get_runtime(context_schema: type[ContextT] | None = None) -> Runtime[ContextT]:
    runtime = cast(Runtime[ContextT], get_config()[CONF].get(CONFIG_KEY_RUNTIME))
    return runtime
```

`context_schema` 参数纯粹用于类型提示——让 IDE 知道返回的 `Runtime[ContextT]` 中 `context` 字段的具体类型。`cast()` 不做运行时检查，只影响类型检查器的推断——Java 不需要 `cast`，因为泛型在编译时已确定。

### 10.3 managed/ — 托管值：不持久化的"虚拟状态"

Managed values 是 LangGraph 中一个优雅的设计模式：它们在 State 定义中看起来像普通键，但值是运行时动态计算的，不参与检查点持久化。

**managed/base.py** — 抽象基类与类型定义：

```python
class ManagedValue(ABC, Generic[V]):
    @staticmethod
    @abstractmethod
    def get(scratchpad: PregelScratchpad) -> V: ...

ManagedValueSpec = type[ManagedValue]
ManagedValueMapping = dict[str, ManagedValueSpec]
```

`ManagedValue` 是泛型抽象类，唯一的抽象方法 `get()` 接收 `PregelScratchpad` 参数并返回计算值。`PregelScratchpad` 是一个 dataclass，保存当前执行步数（`step`）、递归上限（`stop`）、中断计数器等临时数据——这些数据不会持久化到检查点，仅在当次执行的生命周期内存在。

`ManagedValueSpec` 是 `type[ManagedValue]` 的别名——它代表类本身（而非实例），类似 Java 的 `Class<? extends ManagedValue>`。`ManagedValueMapping` 将字符串键名映射到 ManagedValue 子类。

**is_managed_value()** 类型守卫函数：

```python
def is_managed_value(value: Any) -> TypeGuard[ManagedValueSpec]:
    return isclass(value) and issubclass(value, ManagedValue)
```

`TypeGuard` 是 Python 类型系统的特殊注解——当此函数返回 `True` 时，类型检查器会将 `value` 的类型窄化为 `ManagedValueSpec`。Java 中等价的模式是 `instanceof` 检查后编译器自动窄化类型，但 Java 不需要显式声明"类型守卫"。

**managed/is_last_step.py** — 具体实现：

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

`IsLastStep` 的定义 `Annotated[bool, IsLastStepManager]` 是关键的"魔法"——它在类型层面是 `bool`，但附加了 `IsLastStepManager` 作为元数据。用户在 State 中声明：

```python
class MyState(TypedDict):
    messages: list[BaseMessage]
    is_last_step: IsLastStep       # 看起来是 bool
    remaining_steps: RemainingSteps  # 看起来是 int
```

运行时，LangGraph 的模式解析器（`_fields.py` 中的函数）发现 `Annotated[bool, IsLastStepManager]` 的元数据是 `ManagedValueSpec` 子类，就将其标记为托管值而非普通 Channel。读取时调用 `IsLastStepManager.get(scratchpad)` 获取当前计算结果，写入时忽略（托管值不接受写入）。

`scratchpad.step == scratchpad.stop - 1` 的逻辑：Pregel 引擎每执行一个 superstep，`step` 递增 1。当 `step == stop - 1` 时，下一个 superstep 就会触发递归限制。`stop` 的值来自 `config["recursion_limit"]`，默认 10007——这是一个看似奇怪的数字，选择它是因为足够大（实际应用很少达到）同时不是 2 的幂次（避免与位运算冲突）。

### 10.4 托管值 vs Channel：两种状态机制

理解托管值的关键是区分它与普通 Channel 的差异：

| 特性 | Channel（普通状态键） | Managed Value（托管值） |
|---|---|---|
| 存储 | 持久化在检查点中 | 不持久化，每次从 scratchpad 计算 |
| 更新 | 节点写入时由 reducer 更新 | 只读，不接受节点写入 |
| 声明方式 | `Annotated[type, reducer]` | `Annotated[type, ManagedValue子类]` |
| 子图传播 | 通过检查点命名空间传播 | 在子图中重新计算 |
| 恢复行为 | 从检查点恢复 | 从恢复点的 scratchpad 重新计算 |
| 生命周期 | 跨检查点持久存在 | 仅在当前 superstep 内有效 |

用一个 Java 风格的类比来理解：Channel 像数据库中的列（持久化、可更新），Managed Value 像数据库中的计算列 / 虚拟列（`GENERATED ALWAYS AS ...`，每次查询时计算，不占存储）。

`IsLastStep` 和 `RemainingSteps` 的典型用途是在节点中做条件判断：

```python
def my_node(state: MyState):
    if state["is_last_step"]:
        # 递归限制快到了，做收尾工作
        return {"messages": [...], "final_answer": "..."}
    # 正常处理
    return {"messages": [...]}
```

Java 开发者可能觉得这种"状态键其实是计算属性"的设计有些隐晦——在 Java 中更常见的做法是在方法参数中显式传入 `StepContext` 对象。但 Python 社区偏好"约定优于显式"的风格，`Annotated[type, manager]` 的声明方式让 State 定义保持简洁，同时运行时自动注入计算值。

## 运行原理

配置与运行时的完整协作流程：

1. **图编译时**：`StateGraph.__init__()` 解析 State schema，`get_cached_annotated_keys()` 提取所有键，对每个键检查 `Annotated` 元数据——如果元数据是 `ManagedValueSpec` 子类则注册为托管值，否则注册为 Channel。

2. **图调用时**：`Pregel.invoke()` 调用 `ensure_config()` 合并配置，创建 `Runtime` 实例，通过 `patch_config()` 注入到 `config["configurable"]["__pregel_runtime"]`。同时创建 `PregelScratchpad(step=0, stop=recursion_limit, ...)`。

3. **superstep 执行时**：对每个待执行任务，引擎将 `scratchpad` 存入 `config["configurable"]["__pregel_scratchpad"]`。节点函数读取状态时，托管值通过 `ManagedValue.get(scratchpad)` 动态计算。

4. **superstep 结束时**：`scratchpad.step += 1`。Channel 写入由 reducer 处理并持久化到检查点。托管值不参与此过程。

5. **节点函数中调用 `get_config()`**：从 `var_child_runnable_config` 上下文变量获取配置 → 从配置中提取 Runtime → 访问 `store`、`stream_writer` 等属性。如果节点声明了 `config: RunnableConfig` 参数，`RunnableCallable` 会自动注入，无需显式调用 `get_config()`。

6. **上下文传播**：`RunnableCallable.invoke()` 和 `RunnableSeq.invoke()` 通过 `set_config_context()` 上下文管理器设置 `var_child_runnable_config`，确保嵌套调用中 `get_config()` 能正确返回当前配置。在 Python 3.11+ 中，使用 `copy_context()` + `context.run()` 确保协程间的上下文传播。

## 动手实验

1. **验证上下文变量行为**：
   ```python
   import contextvars
   var = contextvars.ContextVar("test", default=None)
   var.set("hello")
   print(var.get())  # "hello"
   # 在协程中自动传播（Python 3.11+）
   import asyncio
   async def inner():
       print(var.get())  # "hello" — 协程继承了上下文
   asyncio.run(inner())
   ```

2. **体验 Runtime 的不可变模式**：
   ```python
   from langgraph.runtime import Runtime, ExecutionInfo
   from dataclasses import replace
   info = ExecutionInfo(
       checkpoint_id="cp-1", checkpoint_ns="", task_id="t-1"
   )
   # 尝试修改
   try:
       info.checkpoint_id = "cp-2"  # FrozenInstanceError!
   except AttributeError as e:
       print(f"不可变: {e}")
   # 正确方式：创建新实例
   new_info = info.patch(task_id="t-2")
   print(new_info.task_id)       # "t-2"
   print(info.task_id)           # "t-1" — 原对象不变
   ```

3. **观察托管值的计算性质**：
   ```python
   from langgraph.managed.is_last_step import IsLastStep, RemainingSteps
   from typing import get_args, get_origin
   import typing
   # 解析 Annotated 类型
   origin = get_origin(IsLastStep)  # Annotated
   args = get_args(IsLastStep)      # (bool, IsLastStepManager)
   print(args[0])        # <class 'bool'>
   print(args[1])        # <class 'IsLastStepManager'>
   # 验证 IsLastStepManager 是 ManagedValue 子类
   from langgraph.managed.base import ManagedValue
   print(issubclass(args[1], ManagedValue))  # True
   ```

4. **模拟托管值的计算**：
   ```python
   from langgraph._internal._scratchpad import PregelScratchpad
   from langgraph.managed.is_last_step import IsLastStepManager, RemainingStepsManager
   # 创建 scratchpad（简化版，只填必要字段）
   scratchpad = PregelScratchpad(
       step=9999, stop=10007,
       call_counter=lambda: 0,
       interrupt_counter=lambda: 0,
       get_null_resume=lambda _: None,
       resume=[],
       subgraph_counter=lambda: 0,
   )
   print(IsLastStepManager.get(scratchpad))       # False
   print(RemainingStepsManager.get(scratchpad))   # 8
   # 修改 step 模拟最后一步
   from dataclasses import replace
   last_step = replace(scratchpad, step=10006)
   print(IsLastStepManager.get(last_step))        # True
   print(RemainingStepsManager.get(last_step))    # 1
   ```

5. **理解 Runtime.merge() 的语义**：
   ```python
   from langgraph.runtime import Runtime
   r1 = Runtime(context={"user": "alice"}, store=None)
   r2 = Runtime(context=None, store="my_store")
   merged = r1.merge(r2)
   print(merged.context)  # {"user": "alice"} — r2.context 为 None，用 r1
   print(merged.store)    # "my_store" — r2.store 有值，用 r2
   ```