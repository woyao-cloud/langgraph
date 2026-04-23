# 第 1 章：入口与类型系统

> 源码文件：`types.py`, `constants.py`, `errors.py`

## Java 桥梁

如果你来自 Java 世界，你已经习惯了用 `interface` 定义契约、用 `enum` 定义常量、用 `record` 定义不可变数据类、用泛型参数化类型。Python 有对应的机制，但形式不同。本章通过走读 LangGraph 的类型定义文件，帮你理解这些 Python 特性。

## Python 概念速查

### `TypedDict` — Python 版的 Java Record

Java 17+ 的 `record`：

```java
public record TaskPayload(String id, String name, Object input) {}
```

Python 的 `TypedDict`：

```python
class TaskPayload(TypedDict):
    id: str
    name: str
    input: Any
```

关键区别：
- `TypedDict` 是**运行时字典**，类型提示仅用于静态检查，运行时不强制约束
- `TypedDict` 的字段本质是字典的键值对，可以用 `task["id"]` 访问
- `NotRequired[T]` 标记可选字段，类似 Java record 中可为 null 的字段

### `NamedTuple` — 不可变元组 + 命名字段

Java 的 record：

```java
public record RetryPolicy(double initialInterval, double backoffFactor) {}
```

Python 的 `NamedTuple`：

```python
class RetryPolicy(NamedTuple):
    initial_interval: float = 0.5
    backoff_factor: float = 2.0
```

关键区别：
- `NamedTuple` 实例是**不可变**的（像 Java record）
- 支持默认值
- 可以用 `policy.initial_interval` 或 `policy[0]` 访问

### `dataclass` — Python 版的 Java data class

```python
@dataclass(frozen=True, slots=True)
class Interrupt:
    value: Any
    id: str = _DEFAULT_INTERRUPT_ID
```

Java 对应：

```java
record Interrupt(Object value, String id) {}
```

关键区别：
- `frozen=True` 使实例不可变（类似 Java record 的不可变性）
- `slots=True` 使用 `__slots__` 替代 `__dict__`，节省内存（类似 Java 的字段布局而非 HashMap）
- Python `dataclass` 的 `kw_only=True` 强制关键字传参，类似 Java 的 Builder 模式

### `__slots__` — Python 的内存优化

Java 中，类的字段在编译期就确定了内存布局。Python 默认用 `__dict__`（一个 HashMap）存储实例属性，这很灵活但内存开销大。`__slots__` 告诉 Python 提前声明所有字段名，使 Python 使用类似 Java 的固定内存布局：

```python
class LastValue(Generic[Value], BaseChannel[Value, Value, Value]):
    __slots__ = ("value",)  # 声明所有字段，替代 __dict__

    def __init__(self, typ, key=""):
        super().__init__(typ, key)
        self.value = MISSING  # 只能用 __slots__ 中声明的字段名
```

好处：省内存、访问更快、防止拼写错误的属性赋值。

### `Literal` — Python 版的枚举子集

Java 中：

```java
public enum StreamMode { VALUES, UPDATES, MESSAGES, CUSTOM, CHECKPOINTS, TASKS, DEBUG }
```

Python 中：

```python
StreamMode = Literal["values", "updates", "messages", "custom", "checkpoints", "tasks", "debug"]
```

`Literal` 是类型层面的约束——变量只能是这几个字符串值之一。运行时它就是普通字符串，但 mypy/pyright 等静态检查工具会验证。这比 Java enum 更轻量：不需要定义新类，直接约束已有类型。

### `TypeVar` 和 `Generic` — Python 泛型

Java：

```java
public class BaseChannel<Value, Update, Checkpoint> { ... }
```

Python：

```python
Value = TypeVar("Value")
Update = TypeVar("Update")
Checkpoint = TypeVar("Checkpoint")

class BaseChannel(Generic[Value, Update, Checkpoint], ABC):
    ...
```

关键区别：
- Python 的泛型是**声明时**的注解，运行时类型擦除比 Java 更彻底
- `TypeVar` 可以定义约束：`TypeVar("T", bound=BaseClass)` 类似 Java 的 `<T extends BaseClass>`
- `Generic[A, B]` 对应 Java 的 `class MyClass<A, B>`

### `from __future__ import annotations` — 延迟求值

Python 的类型注解默认在定义时就要求引用的类型已存在。`from __future__ import annotations` 让所有注解变成字符串，延迟到实际使用时才求值。这类似 Java 的前向引用，但更灵活：

```python
from __future__ import annotations  # 文件顶部

class Node:
    def process(self) -> Node:  # 没有 future annotations，这里 Node 还没定义完会报错
        ...
```

### `@final` — Python 版的 final 类/方法

Java：

```java
public final class Interrupt { ... }
```

Python：

```python
from typing import final

@final
@dataclass(init=False, slots=True)
class Interrupt:
    value: Any
    id: str
```

`@final` 是类型检查器（mypy/pyright）的提示，运行时不强制，但 IDE 和静态检查会阻止子类化。

### `__all__` — 模块的公共 API 声明

Java 用 `module-info.java` 声明模块导出：

```java
module langgraph.types {
    exports Send, Command, Interrupt;
}
```

Python 用 `__all__`：

```python
__all__ = (
    "Send",
    "Command",
    "Interrupt",
    "RetryPolicy",
    ...
)
```

`from langgraph.types import *` 只会导入 `__all__` 列出的名称。这也是文档工具识别公共 API 的依据。

## 代码走读

### 1. 常量定义 — `constants.py`

```python
# langgraph/constants.py

TAG_NOSTREAM = sys.intern("nostream")    # 禁止流式输出的标记
TAG_HIDDEN = sys.intern("langsmith:hidden")  # 隐藏节点/边的标记
END = sys.intern("__end__")     # 图的终止节点
START = sys.intern("__start__") # 图的起始节点
```

**`sys.intern()` 是什么？** 这是 Python 的字符串驻留机制。Java 也有字符串常量池（String pool），`sys.intern()` 的效果完全一样：相同内容的字符串只存储一份，后续用 `==` 比较时直接比较内存地址（`is`），而非逐字符比较。

Java 对比：

```java
// Java 字符串常量池
String s1 = "__end__";
String s2 = "__end__";
s1 == s2;  // true，因为字面量自动驻留

// Python sys.intern
s1 = sys.intern("__end__")
s2 = sys.intern("__end__")
s1 is s2;  // true，intern 保证同一引用
```

**为什么用 `sys.intern`？** `START` 和 `END` 在整个框架中频繁用作字典键和比较值。驻留后 `is` 比较比 `==` 快，而且保证身份比较的可靠性。

### 2. 错误层次结构 — `errors.py`

```python
class GraphRecursionError(RecursionError):
    """图超过最大步数时抛出。"""

class InvalidUpdateError(Exception):
    """通道更新无效时抛出。"""

class GraphBubbleUp(Exception):
    """子图异常冒泡的基类。"""

class GraphInterrupt(GraphBubbleUp):
    """图中断时抛出，携带 Interrupt 对象列表。"""

class ParentCommand(GraphBubbleUp):
    """子图向父图发送 Command 时的异常包装。"""

class EmptyInputError(Exception):
    """图接收到空输入时抛出。"""

class TaskNotFound(Exception):
    """分布式执行中找不到任务时抛出。"""
```

**Java 对比：** 这个错误层次结构类似 Java 的受检异常 vs 非受检异常设计：

- `GraphBubbleUp` 是一个**控制流异常**（类似 Java 的 `InterruptedException`），不是真正的错误
- `GraphInterrupt` 继承自 `GraphBubbleUp`，在子图中抛出、在根图被捕获处理
- `GraphRecursionError` 继承 `RecursionError`（Python 内置），类似 Java 的 `StackOverflowError`

Python 中异常类继承体系的设计思路和 Java 完全一致：**用继承关系表达"可以被同一种 catch 块处理"**。

注意 `@deprecated` 装饰器的用法：

```python
@deprecated("NodeInterrupt is deprecated. Please use `interrupt` instead.", category=None)
class NodeInterrupt(GraphInterrupt):
    ...
```

这类似 Java 的 `@Deprecated` 注解，但 Python 的 `deprecated` 来自 `typing_extensions`，能在运行时发出警告。

### 3. 核心类型 — `types.py`

#### Send — 动态路由消息

```python
class Send:
    __slots__ = ("node", "arg")

    node: str  # 目标节点名
    arg: Any   # 发送给该节点的参数

    def __init__(self, /, node: str, arg: Any) -> None:
        self.node = node
        self.arg = arg
```

**Java 对比：** `Send` 类似 Akka 的 `ActorRef.tell(message)` —— 向特定节点发送消息。但 LangGraph 的 `Send` 不是异步消息，而是在当前超步（superstep）中创建新任务。

**`__slots__` 详解：** `__slots__ = ("node", "arg")` 声明了该类的所有实例属性。效果：
1. 禁止动态添加属性（类似 Java 的固定字段列表）
2. 省去 `__dict__` 的内存开销（每个实例省约 56 字节）
3. 属性访问更快（用数组偏移而非哈希查找）

**`/` 参数语法：** `def __init__(self, /, node: str, arg: Any)` 中的 `/` 是 Python 3.8+ 的仅限位置参数标记。`/` 之前的参数只能按位置传递：

```python
Send("my_node", data)      # 正确
Send(node="my_node", arg=data)  # TypeError! / 之后的才能用关键字
```

这类似 Java 的构造函数——参数顺序固定，不能按名传递。

#### Command — 多功能控制原语

```python
@dataclass(**_DC_KWARGS)  # _DC_KWARGS = {"kw_only": True, "slots": True, "frozen": True}
class Command(Generic[N], ToolOutputMixin):
    graph: str | None = None
    update: Any | None = None
    resume: dict[str, Any] | Any | None = None
    goto: Send | Sequence[Send | N] | N = ()

    PARENT: ClassVar[Literal["__parent__"]] = "__parent__"
```

**Java 对比：** `Command` 类似一个命令模式（Command Pattern）的复合对象：

```java
// Java 等价概念
public record Command<N>(
    Optional<String> graph,      // 目标子图
    Optional<Object> update,     // 状态更新
    Optional<Object> resume,     // 中断恢复值
    Object goto                   // 跳转目标
) {}
```

**`Generic[N]`** — `N` 是节点名称的类型参数。`Command[str]` 表示节点名是字符串。

**`ClassVar`** — `PARENT` 是类变量，不属于实例。类似 Java 的 `static final`：

```python
Command.PARENT  # 访问类变量
cmd = Command(update={"key": "val"})
cmd.PARENT      # 也可以通过实例访问，但不存储在实例中
```

**`kw_only=True`** — 强制所有参数用关键字传递，类似 Java 的 Builder 模式：

```python
Command(update={"x": 1}, resume="answer")  # 正确
Command({"x": 1}, "answer")                  # TypeError! 必须用关键字
```

#### Interrupt — 中断执行

```python
@final
@dataclass(init=False, slots=True)
class Interrupt:
    value: Any
    id: str

    def __init__(self, value: Any, id: str = _DEFAULT_INTERRUPT_ID, **deprecated_kwargs):
        self.value = value
        # 处理废弃的 ns 参数...
        if (ns := deprecated_kwargs.get("ns", MISSING)) is not MISSING and id == _DEFAULT_INTERRUPT_ID:
            self.id = xxh3_128_hexdigest("|".join(ns).encode())
        else:
            self.id = id
```

**`@final`** — 表示这个类不能被继承。Java 的 `final class` 完全等价。

**`init=False`** — 告诉 dataclass 不要自动生成 `__init__`，因为这里自定义了构造函数。Java 中没有这种需要——Java 构造函数总是自定义的。

**`xxh3_128_hexdigest`** — xxHash 的 128 位哈希。类似 Java 的 `MessageDigest.getInstance("SHA-256")`，但更快（非加密用途）。

#### interrupt() — 中断执行函数

```python
def interrupt(value: Any) -> Any:
    """从节点内中断图执行，将值返回给客户端。"""
    conf = get_config()["configurable"]
    scratchpad = conf[CONFIG_KEY_SCRATCHPAD]
    idx = scratchpad.interrupt_counter()

    if scratchpad.resume:
        if idx < len(scratchpad.resume):
            conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
            return scratchpad.resume[idx]

    v = scratchpad.get_null_resume(True)
    if v is not None:
        scratchpad.resume.append(v)
        conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
        return v

    raise GraphInterrupt((Interrupt.from_ns(value=value, ns=conf[CONFIG_KEY_CHECKPOINT_NS]),))
```

**这个函数的设计模式很有意思：** 同一个函数在第一次调用时抛异常，在恢复执行时返回值。这类似 Java 的"异常流控"模式（如 `Thread.interrupt()`），但更精巧：

- **首次调用**：`raise GraphInterrupt(...)` 中断执行
- **恢复调用**：从 `scratchpad.resume` 中取出恢复值并返回

**Python 语法提示：**

- `conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])` — 调用从配置中注入的函数，参数是一个元组列表
- `scratchpad.resume[idx]` — 列表索引访问
- `if (ns := deprecated_kwargs.get("ns", MISSING)) is not MISSING` — 海象运算符 `:=` 同时赋值和判断

#### StreamMode — 流式输出模式

```python
StreamMode = Literal[
    "values", "updates", "checkpoints", "tasks", "debug", "messages", "custom"
]
```

**Java 对比：** 这比 Java enum 轻量得多。在 Java 中你会写：

```java
public enum StreamMode { VALUES, UPDATES, CHECKPOINTS, TASKS, DEBUG, MESSAGES, CUSTOM }
```

Python 的 `Literal` 是类型层面的约束，运行时值就是普通字符串。好处：
1. 不需要定义新类
2. 用户可以直接传字符串，无需 `.VALUES`
3. 类型检查器仍然能验证

#### StateSnapshot — 状态快照

```python
class StateSnapshot(NamedTuple):
    values: dict[str, Any] | Any
    next: tuple[str, ...]
    config: RunnableConfig
    metadata: CheckpointMetadata | None
    created_at: str | None
    parent_config: RunnableConfig | None
    tasks: tuple[PregelTask, ...]
    interrupts: tuple[Interrupt, ...]
```

**Java 对比：** `NamedTuple` 创建的既是元组又是类。你可以用 `snapshot.values` 或 `snapshot[0]` 两种方式访问。类似 Java record：

```java
public record StateSnapshot(
    Object values,
    List<String> next,
    RunnableConfig config,
    CheckpointMetadata metadata,
    String createdAt,
    RunnableConfig parentConfig,
    List<PregelTask> tasks,
    List<Interrupt> interrupts
) {}
```

注意 `tuple[str, ...]` 语法——`...` 表示任意长度的同类型元组，类似 Java 的 `String[]` 但不可变。

#### RetryPolicy — 重试策略

```python
class RetryPolicy(NamedTuple):
    initial_interval: float = 0.5
    backoff_factor: float = 2.0
    max_interval: float = 128.0
    max_attempts: int = 3
    jitter: bool = True
    retry_on: type[Exception] | Sequence[type[Exception]] | Callable[[Exception], bool] = default_retry_on
```

**Java 对比：** 这类似 resilience4j 的 `RetryConfig`：

```java
RetryConfig config = RetryConfig.custom()
    .maxAttempts(3)
    .intervalFunction(IntervalFunction.ofExponentialBackoff(Duration.ofMillis(500), 2.0))
    .retryOnException(e -> e instanceof RuntimeException)
    .build();
```

Python 的 `NamedTuple` 比 dataclass 更轻量，适合这种纯数据容器。

**`retry_on` 字段类型：** `type[Exception] | Sequence[type[Exception]] | Callable[[Exception], bool]` — 这是一个联合类型（Union type），类似 Java 的：

```java
// Java 中没有直接对应的联合类型，通常用泛型或重载
Object retryOn; // 可以是 Class<? extends Exception>，或 List<Class<? extends Exception>>，或 Predicate<Exception>
```

Python 的联合类型在静态检查时能精确定义，运行时仍然是普通对象。

## 运行原理

### 类型系统的设计哲学

LangGraph 的类型系统遵循以下原则：

1. **`NamedTuple` 用于不可变数据容器**：`RetryPolicy`, `CachePolicy`, `StateSnapshot`, `StateUpdate` —— 这些类型只承载数据，不需要方法。类似 Java record。

2. **`dataclass` 用于需要自定义行为的类**：`Interrupt`, `Command` —— 需要自定义 `__init__`、`__eq__` 等方法时使用。类似 Java class with Lombok `@Value`。

3. **`TypedDict` 用于流式数据传输**：`TaskPayload`, `CheckpointPayload`, 各种 `StreamPart` —— 这些是 JSON 兼容的字典结构。类似 Java 的 DTO (Data Transfer Object)。

4. **`Literal` 用于枚举式约束**：`StreamMode`, `Durability` —— 不需要方法或行为的枚举值。

5. **`TypeVar` + `Generic` 用于参数化类型**：`BaseChannel[Value, Update, Checkpoint]`, `Command[N]` —— 和 Java 泛型完全一致的用途。

### 中断-恢复机制的数据流

```
首次调用 interrupt(value)
  ┌─────────────────────────────────────────┐
  │ 1. scratchpad.interrupt_counter() → idx  │
  │ 2. scratchpad.resume → 空               │
  │ 3. raise GraphInterrupt(Interrupt(value)) │
  └─────────────────────────────────────────┘
           ↓ 图被中断，Interrupt 值传递给客户端

恢复调用 interrupt(value)（第二次执行同一节点）
  ┌─────────────────────────────────────────┐
  │ 1. scratchpad.interrupt_counter() → idx  │
  │ 2. scratchpad.resume → 有值              │
  │ 3. idx < len(resume) → True              │
  │ 4. return resume[idx]  ← 返回恢复值      │
  └─────────────────────────────────────────┘
```

## 动手实验

```python
# 实验 1：理解 TypedDict vs dataclass vs NamedTuple 的区别
from typing import TypedDict, NamedTuple
from dataclasses import dataclass

# TypedDict — 运行时是字典
class TaskInfo(TypedDict):
    id: str
    name: str

task: TaskInfo = {"id": "1", "name": "process"}  # 字面量写法
print(task["id"])    # "1" — 像字典一样访问
print(task.get("name"))  # "process"

# NamedTuple — 运行时是元组 + 属性
class RetryConfig(NamedTuple):
    max_attempts: int = 3
    backoff: float = 2.0

config = RetryConfig()
print(config.max_attempts)  # 3 — 像对象一样访问属性
print(config[0])            # 3 — 也可以用索引访问

# dataclass — 运行时是对象
@dataclass(frozen=True)
class Command:
    update: dict | None = None
    goto: str | None = None

cmd = Command(update={"x": 1})
print(cmd.update)  # {'x': 1}

# 实验 2：理解 __slots__ 的内存效果
import sys

class WithoutSlots:
    def __init__(self):
        self.value = 1
        self.key = "a"

class WithSlots:
    __slots__ = ("value", "key")
    def __init__(self):
        self.value = 1
        self.key = "a"

print(sys.getsizeof(WithoutSlots()))  # ~56 + __dict__ 开销
print(sys.getsizeof(WithSlots()))     # ~48，更小

# 实验 3：理解 interrupt 的行为
# 注意：需要完整的 LangGraph 环境
from langgraph.types import Interrupt

# 创建 Interrupt 对象
intr = Interrupt(value="请确认这个操作", id="confirm-001")
print(intr.value)  # "请确认这个操作"
print(intr.id)     # "confirm-001"

# 使用默认 id（会自动从 ns 生成哈希）
intr2 = Interrupt(value="请输入年龄")
print(intr2.id)  # 默认值 "placeholder-id"
```