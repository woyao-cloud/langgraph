# 第一章 类型系统与常量定义 — types.py, constants.py, errors.py

## Python 进阶

### 1. TypedDict vs dataclass vs NamedTuple：三路设计抉择

LangGraph 的类型定义同时使用了 `TypedDict`、`dataclass` 和 `NamedTuple`，每种选择背后都有明确的工程意图：

- **TypedDict**（如 `TaskPayload`、`ValuesStreamPart`）：用于描述 JSON 风格的字典结构，零运行时开销，仅在类型检查时生效。适合流式输出的序列化场景——这些数据最终会以 JSON 形式传输。
- **NamedTuple**（如 `RetryPolicy`、`PregelTask`、`StateSnapshot`）：不可变、内存紧凑、支持位置解包。用于"值对象"——一旦构造就不应改变的数据。
- **dataclass**（如 `Command`、`Interrupt`、`GraphOutput`）：需要 `__slots__`、`frozen`、`kw_only` 等细粒度控制时使用。适合有复杂初始化逻辑或需要方法的对象。

### 2. `__slots__` 优化

```python
class Send:
    __slots__ = ("node", "arg")
```

`__slots__` 阻止动态属性创建，将属性存储从 `__dict__` 移到预分配的描述符槽位。效果：每个实例节省约 40-50 字节内存，属性访问速度提升约 20%。在 Pregel 引擎同时管理数千个 `Send`/`PregelTask` 对象时，这个优化显著减少 GC 压力。

### 3. `sys.intern()` 字符串驻留

```python
START = sys.intern("__start__")
END = sys.intern("__end__")
```

`sys.intern()` 将字符串注册到 Python 解释器的驻留表，使相同内容的字符串共享同一对象。这不仅是内存优化——更重要的是让 `is` 比较等价于 `==` 比较，在 Pregel 调度循环中频繁比较节点名称时，`is` 比 `==` 快约 2-3 倍。

### 4. Literal 作为轻量枚举

```python
StreamMode = Literal["values", "updates", "checkpoints", "tasks", "debug", "messages", "custom"]
Durability = Literal["sync", "async", "exit"]
```

`Literal` 比 `Enum` 更轻量：零运行时开销，类型检查器直接约束合法值，且与 JSON 序列化天然兼容（不需要 `.value`）。LangGraph 中所有"有限选项"类型都采用此模式。

### 5. TypeVar bounds 与 Generic 多参数

```python
StateT = TypeVar("StateT")
OutputT = TypeVar("OutputT")
N = TypeVar("N", bound=Hashable)
KeyFuncT = TypeVar("KeyFuncT", bound=Callable[..., str | bytes])
```

`bound=Hashable` 限制 `N` 必须是可哈希类型——因为 `Command[N]` 的 `goto` 字段中 `N` 会作为字典键或集合元素。`KeyFuncT` 的 bound 确保缓存键函数返回 `str | bytes`。

### 6. TypeAliasType 与泛型类型别名

```python
StreamPart = TypeAliasType(
    "StreamPart",
    ValuesStreamPart[OutputT] | UpdatesStreamPart | ... | DebugStreamPart[StateT],
    type_params=(StateT, OutputT),
)
```

`TypeAliasType`（PEP 695）创建真正的泛型类型别名，`type_params` 声明别名自身的类型参数。与旧式 `StreamPart = Union[...]` 不同，它在运行时是一个真正的类型对象，可被 `isinstance` 和类型检查器正确处理。

### 7. `@final` 装饰器与 `kw_only` dataclass 参数

```python
@final
@dataclass(init=False, slots=True)
class Interrupt:
    ...
```

`@final` 告知类型检查器此类不可被继承。`init=False` 禁止自动生成 `__init__`，因为 `Interrupt` 需要自定义初始化逻辑（兼容废弃的 `ns` 参数）。

```python
_DC_KWARGS = {"kw_only": True, "slots": True, "frozen": True}
@dataclass(**_DC_KWARGS)
class Command(Generic[N], ToolOutputMixin):
```

`kw_only=True` 强制所有参数必须以关键字形式传入，防止位置参数混乱。

### 8. `from __future__ import annotations` 与 `deprecated()`

文件首行的 `from __future__ import annotations`（PEP 563）延迟所有注解求值，避免循环引用，并允许在运行时使用尚未定义的类型。`typing_extensions.deprecated()` 则在类型层面标记废弃，类型检查器会在使用处发出警告。

---

## 代码走读

### types.py 核心类型定义

**文件头部（第 1-30 行）**

```python
from __future__ import annotations          # 延迟注解求值
import sys
from collections import deque               # PregelExecutableTask.writes 使用 deque
from collections.abc import Callable, Hashable, Sequence
from dataclasses import asdict, dataclass
from typing import (
    TYPE_CHECKING,                           # 条件导入守卫
    Any, ClassVar, Generic, Literal,
    NamedTuple, TypeVar, final,
)
from warnings import warn
from xxhash import xxh3_128_hexdigest        # 高性能哈希，用于 interrupt ID 生成
```

第 39-41 行的条件导入：

```python
if TYPE_CHECKING:
    from langgraph.pregel.protocol import PregelProtocol
```

仅在类型检查时导入 `PregelProtocol`，运行时不触发，避免循环依赖。这是 Python 中处理循环引用的标准模式。

第 43-48 行的可选导入：

```python
try:
    from langchain_core.messages.tool import ToolOutputMixin
except ImportError:
    class ToolOutputMixin: pass
```

`ToolOutputMixin` 来自 `langchain-core` 的工具子系统。若未安装则定义空类作为占位——这是优雅的降级策略，让 `Command` 在不依赖工具功能的场景下仍可正常使用。

**`__all__` 导出控制（第 51-83 行）**

```python
__all__ = (
    "All", "Checkpointer", "StreamMode", "StreamWriter", "StreamPart",
    ...
    "Overwrite", "GraphOutput", "ensure_valid_checkpointer",
)
```

显式声明模块的公开 API。`from langgraph.types import *` 只会导入这些名称。注意 `CacheKey`、`_DEFAULT_INTERRUPT_ID`、`_DC_KWARGS`、`_T_DC_KWARGS` 等以下划线开头的内部符号不在其中。

**Literal 类型（第 85-132 行）**

```python
Durability = Literal["sync", "async", "exit"]
All = Literal["*"]
Checkpointer = None | bool | BaseCheckpointSaver
StreamMode = Literal["values", "updates", "checkpoints", "tasks", "debug", "messages", "custom"]
```

`Checkpointer` 使用 `None | bool | BaseCheckpointSaver` 联合类型，三值语义：`True` 启用、`False` 禁用、`None` 继承父图。`All` 仅含 `"*"` 通配值，用于中断配置。

**ensure_valid_checkpointer 验证函数（第 105-115 行）**

```python
def ensure_valid_checkpointer(checkpointer: Checkpointer) -> Checkpointer:
    if checkpointer not in (None, True, False) and not isinstance(
        checkpointer, BaseCheckpointSaver
    ):
        raise TypeError(...)
    return checkpointer
```

先检查三个特殊值（`in` 比较快），再检查实例类型。这是一种优化：`in` 对小元组是 O(n) 但 n=3 且全都是单例对象，`is` 比较极快。

**流式 TypedDict 类型族（第 140-353 行）**

这是 types.py 中最大的类型定义区域。核心设计模式是"判别联合"（discriminated union）——每个流部分都有 `type: Literal["xxx"]` 字段作为判别器：

```python
class ValuesStreamPart(TypedDict, Generic[OutputT]):
    type: Literal["values"]
    ns: tuple[str, ...]
    data: OutputT
    interrupts: tuple[Interrupt, ...]

class UpdatesStreamPart(TypedDict):
    type: Literal["updates"]
    ns: tuple[str, ...]
    data: dict[str, Any]
```

`ns` 字段（namespace）使用 `tuple[str, ...]` 表示图的层级路径，如 `("graph", "subgraph")`。所有流部分共享 `type` 和 `ns` 结构，但 `data` 的类型各异。

`StreamPart` 用 `TypeAliasType` 统合所有变体：

```python
StreamPart = TypeAliasType(
    "StreamPart",
    ValuesStreamPart[OutputT] | UpdatesStreamPart | ... | DebugStreamPart[StateT],
    type_params=(StateT, OutputT),
)
```

注意 `DebugPayload` 也使用了同样的模式：

```python
DebugPayload = TypeAliasType(
    "DebugPayload",
    _DebugCheckpointPayload[StateT] | _DebugTaskPayload | _DebugTaskResultPayload,
    type_params=(StateT,),
)
```

**GraphOutput：向后兼容的 dataclass（第 356-398 行）**

```python
@dataclass(frozen=True)
class GraphOutput(Generic[OutputT]):
    value: OutputT
    interrupts: tuple[Interrupt, ...] = ()

    def __getitem__(self, key: str) -> Any:
        warn("Accessing GraphOutput via `result[key]` is deprecated. ...")
        if key == _INTERRUPT_KEY:
            return self.interrupts
        if isinstance(self.value, dict):
            return self.value[key]
        try:
            return getattr(self.value, key)
        except AttributeError:
            raise KeyError(key)
```

`__getitem__` 实现了三层回退：先检查中断键，再尝试字典键，最后尝试属性访问。这让旧代码 `result["__interrupt__"]` 和 `result["some_key"]` 仍能工作，同时发出废弃警告。

**Send 类（第 574-647 行）**

```python
class Send:
    __slots__ = ("node", "arg")

    node: str
    arg: Any

    def __init__(self, /, node: str, arg: Any) -> None:
        self.node = node
        self.arg = arg

    def __hash__(self) -> int:
        return hash((self.node, self.arg))

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, Send)
            and self.node == value.node
            and self.arg == value.arg
        )
```

`/` 位置限定符使 `node` 必须以位置参数传入。`__hash__` 和 `__eq__` 的手动实现确保 `Send` 可作为字典键和集合元素——这是 Pregel 去重逻辑的基础。注意：`__slots__` 不自动提供 `__hash__`，所以必须手动实现。

**Command 类（第 652-702 行）**

```python
@dataclass(**_DC_KWARGS)
class Command(Generic[N], ToolOutputMixin):
    graph: str | None = None
    update: Any | None = None
    resume: dict[str, Any] | Any | None = None
    goto: Send | Sequence[Send | N] | N = ()

    PARENT: ClassVar[Literal["__parent__"]] = "__parent__"
```

`_DC_KWARGS = {"kw_only": True, "slots": True, "frozen": True}` 统一了 dataclass 配置。`PARENT` 是 `ClassVar`，不参与实例数据，而是类级别常量。`resume` 字段的三态类型 `dict[str, Any] | Any | None` 支持：按 ID 映射恢复、单值恢复、无恢复值。

`_update_as_tuples` 方法（第 687-700 行）展示了多态状态更新解析：

```python
def _update_as_tuples(self) -> Sequence[tuple[str, Any]]:
    if isinstance(self.update, dict):
        return list(self.update.items())
    elif isinstance(self.update, (list, tuple)) and all(
        isinstance(t, tuple) and len(t) == 2 and isinstance(t[0], str) for t in self.update
    ):
        return self.update
    elif keys := get_cached_annotated_keys(type(self.update)):
        return get_update_as_tuples(self.update, keys)
    elif self.update is not None:
        return [("__root__", self.update)]
    else:
        return []
```

四层解析：字典 -> 键值对列表 -> Annotated 注解对象 -> 根值包裹 -> 空列表。这确保了无论用户传入什么格式的状态更新，都能统一为 `(key, value)` 元组序列。

**Interrupt 类与 interrupt() 函数（第 441-828 行）**

```python
_DEFAULT_INTERRUPT_ID = "placeholder-id"

@final
@dataclass(init=False, slots=True)
class Interrupt:
    value: Any
    id: str

    def __init__(self, value: Any, id: str = _DEFAULT_INTERRUPT_ID, **deprecated_kwargs):
        self.value = value
        if ((ns := deprecated_kwargs.get("ns", MISSING)) is not MISSING
            and (id == _DEFAULT_INTERRUPT_ID)
            and (isinstance(ns, Sequence))):
            self.id = xxh3_128_hexdigest("|".join(ns).encode())
        else:
            self.id = id
```

`_DEFAULT_INTERRUPT_ID` 作为哨兵值：当 `id` 未显式提供且存在废弃的 `ns` 参数时，用 xxhash 从命名空间生成确定性 ID。否则使用显式提供的 `id`。`@final` 阻止继承，因为 `Interrupt` 的语义是严格的值对象。

`interrupt()` 函数（第 705-828 行）是 LangGraph 人机交互的核心：

```python
def interrupt(value: Any) -> Any:
    conf = get_config()["configurable"]
    scratchpad = conf[CONFIG_KEY_SCRATCHPAD]
    idx = scratchpad.interrupt_counter()
    if scratchpad.resume:
        if idx < len(scratchpad.resume):
            conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
            return scratchpad.resume[idx]
    v = scratchpad.get_null_resume(True)
    if v is not None:
        assert len(scratchpad.resume) == idx, (scratchpad.resume, idx)
        scratchpad.resume.append(v)
        conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
        return v
    raise GraphInterrupt((Interrupt.from_ns(value=value, ns=conf[CONFIG_KEY_CHECKPOINT_NS]),))
```

核心逻辑是一个状态机，通过 scratchpad（草稿板）追踪中断计数器：

1. 递增中断索引
2. 检查是否有历史恢复值 → 有则返回对应索引的值
3. 检查当前恢复值 → 有则追加到恢复列表并返回
4. 无恢复值 → 抛出 `GraphInterrupt` 异常，暂停执行

**Overwrite 类（第 831-872 行）**

```python
@dataclass(slots=True)
class Overwrite:
    value: Any
```

注意这里没有 `frozen=True`——因为 `Overwrite` 需要在 `BinaryOperatorAggregate` 中被修改或检查。它是一个标记类型，告诉通道"绕过 reducer，直接写入值"。

### constants.py

```python
TAG_NOSTREAM = sys.intern("nostream")
TAG_HIDDEN = sys.intern("langsmith:hidden")
END = sys.intern("__end__")
START = sys.intern("__start__")
```

四个公共常量全部通过 `sys.intern()` 驻留。

**模块级 `__getattr__` 废弃技巧（第 34-64 行）**

```python
def __getattr__(name: str) -> Any:
    if name in ["Send", "Interrupt"]:
        warn(f"Importing {name} from langgraph.constants is deprecated. ...")
        from importlib import import_module
        module = import_module("langgraph.types")
        return getattr(module, name)

    try:
        from importlib import import_module
        private_constants = import_module("langgraph._internal._constants")
        attr = getattr(private_constants, name)
        warn(f"Importing {name} from langgraph.constants is deprecated. ...")
        return attr
    except AttributeError:
        pass

    raise AttributeError(f"module has no attribute '{name}'")
```

Python 3.7+ 支持模块级 `__getattr__`，当 `getattr(module, name)` 找不到属性时被调用。这段代码实现了两层回退：先将 `Send`/`Interrupt` 重定向到 `langgraph.types`，再尝试从私有常量模块查找，最后抛出 `AttributeError`。这允许旧代码 `from langgraph.constants import Send` 继续工作，同时发出废弃警告。

### errors.py

**错误层次结构：**

```
Exception
├── GraphRecursionError (extends RecursionError)
├── InvalidUpdateError
├── GraphBubbleUp
│   ├── GraphInterrupt
│   │   └── NodeInterrupt (deprecated)
│   └── ParentCommand
├── EmptyInputError
└── TaskNotFound
```

`GraphBubbleUp` 是"气泡异常"基类——在子图中产生、需要冒泡到父图处理的异常。`GraphInterrupt` 和 `ParentCommand` 都继承自它。

```python
class GraphInterrupt(GraphBubbleUp):
    def __init__(self, interrupts: Sequence[Interrupt] = ()) -> None:
        super().__init__(interrupts)
```

`GraphInterrupt` 的参数是 `Interrupt` 序列，通过 `super().__init__` 存入 `args` 元组。Pregel 引擎捕获此异常后提取中断信息。

```python
class ParentCommand(GraphBubbleUp):
    args: tuple[Command]

    def __init__(self, command: Command) -> None:
        super().__init__(command)
```

`ParentCommand` 是子图向父图发送 `Command` 的机制。它也继承 `GraphBubbleUp`，让 Pregel 的异常处理逻辑统一拦截"需要冒泡"的信号。

---

## 运行原理

### Send 创建 PUSH 任务的流程

```
条件边函数返回 [Send("node_a", {"x": 1}), Send("node_b", {"x": 2})]
        │
        ▼
Pregel 引擎检测到返回值类型为 Send 列表
        │
        ▼
为每个 Send 对象创建 PUSH 任务：
  - task_type = PUSH (而非 PULL)
  - task.name = Send.node
  - task.input = Send.arg
        │
        ▼
新任务写入对应通道，下一 super-step 执行
```

与 PULL 任务（由边触发）不同，PUSH 任务的输入来自 `Send.arg` 而非通道读取。这使得同一个节点可以并行执行多次（map-reduce 模式）。

### interrupt/resume 的数据流

```
首次执行:
  node() 调用 interrupt("请确认")
  → scratchpad.interrupt_counter() 递增
  → 无恢复值 → 抛出 GraphInterrupt
  → Pregel 保存 checkpoint，返回中断信息给客户端

恢复执行:
  客户端发送 Command(resume="确认通过")
  → Pregel 从 checkpoint 恢复
  → scratchpad.resume = ["确认通过"]
  → node() 重新执行
  → interrupt("请确认") 被调用
  → scratchpad.interrupt_counter() 返回 0
  → idx(0) < len(resume)(1) → 返回 resume[0] = "确认通过"
  → node() 继续执行
```

关键点：`interrupt()` 通过 `interrupt_counter()` 追踪当前节点中中断调用的顺序索引。每个 `interrupt()` 调用依次对应 `resume` 列表中的一个值。

### Command 的三合一操作

```
Command(update={"key": "val"}, resume="input", goto="next_node")
        │
        ├── update → 写入状态通道
        ├── resume → 填充中断恢复值
        └── goto   → 覆盖下一条边的路由
```

`Command` 将三种操作合并为一个原子指令，Pregel 引擎按 update → resume → goto 的顺序处理。

---

## 实现细节

### xxhash 中断 ID 生成

```python
self.id = xxh3_128_hexdigest("|".join(ns).encode())
```

`xxh3_128_hexdigest` 是 XXH3 算法的 128 位十六进制摘要。选择它的原因：

1. **确定性**：同一命名空间永远生成同一 ID，保证中断恢复时的精确匹配
2. **极高性能**：XXH3 是目前最快的非加密哈希算法之一，约 10 GB/s 吞吐
3. **128 位足够防碰撞**：在合理规模的中断集合中，碰撞概率可忽略

### _DEFAULT_INTERRUPT_ID 占位符模式

```python
_DEFAULT_INTERRUPT_ID = "placeholder-id"
```

这个字符串永远不会被用作真正的 ID——它是一个"非法哨兵"。`__init__` 中检查 `id == _DEFAULT_INTERRUPT_ID` 来判断用户是否显式提供了 ID。相比使用 `None` 作为默认值，这种模式避免了 `None` 与"用户明确传入 None"的歧义。

### constants.py 的 __getattr__ 废弃技巧

模块级 `__getattr__` 只在常规属性查找失败时被调用，不影响正常导入路径的性能。`import_module` 是延迟导入，仅在用户实际使用废弃路径时才触发。这是一种零开销的向后兼容策略。

### GraphOutput __getitem__ 的向后兼容

三层回退逻辑确保了最大的兼容性：

1. `result["__interrupt__"]` → 返回 `self.interrupts`
2. `result["some_dict_key"]`（当 value 是 dict）→ 字典键访问
3. `result["some_attr"]`（当 value 是 dataclass/pydantic）→ 属性访问

同时实现了 `__contains__`，让 `key in result` 语法也能工作。两处都发出废弃警告，引导用户迁移到 `result.value` 和 `result.interrupts`。

### PregelExecutableTask 的版本兼容 dataclass 参数

```python
if sys.version_info > (3, 11):
    _T_DC_KWARGS = {"weakref_slot": True, "slots": True, "frozen": True}
else:
    _T_DC_KWARGS = {"frozen": True}
```

`weakref_slot` 参数在 Python 3.12+ 中引入，允许 `__slots__` 类支持弱引用。在 3.11 及以下，`__slots__` 与 `weakref` 冲突，因此回退到无 `__slots__` 模式。`PregelExecutableTask` 需要 `slots` 和 `frozen` 以支持高效缓存和不可变语义，但弱引用在某些运行时场景中也必须存在。

### RetryPolicy 的 NamedTuple 选择

```python
class RetryPolicy(NamedTuple):
    initial_interval: float = 0.5
    backoff_factor: float = 2.0
    max_interval: float = 128.0
    max_attempts: int = 3
    jitter: bool = True
    retry_on: ... = default_retry_on
```

`NamedTuple` 天然不可变、可哈希、支持解包。作为配置对象，`RetryPolicy` 一旦创建就不应修改，且经常被存入字典或集合。`retry_on` 字段的复杂联合类型 `type[Exception] | Sequence[type[Exception]] | Callable[[Exception], bool]` 体现了 Python 类型系统的表达能力。

---

## 动手实验

### 实验 1：TypedDict vs dataclass vs NamedTuple 行为对比

```python
from typing import NamedTuple
from dataclasses import dataclass
from typing_extensions import TypedDict

# TypedDict: 仅类型层面存在，运行时就是 dict
class ConfigTD(TypedDict):
    name: str
    value: int

td = ConfigTD(name="test", value=42)
print(type(td))       # <class 'dict'>
print(td["name"])     # "test" — 字典键访问
# td.name              # AttributeError — 不是属性

# NamedTuple: 不可变元组 + 属性访问
class ConfigNT(NamedTuple):
    name: str
    value: int = 0

nt = ConfigNT(name="test")
print(type(nt))       # <class '__main__.ConfigNT'>
print(nt.name)        # "test" — 属性访问
print(nt[0])          # "test" — 索引访问
# nt.name = "new"     # AttributeError — 不可变

# dataclass: 可配置的灵活性
@dataclass(frozen=True, slots=True, kw_only=True)
class ConfigDC:
    name: str
    value: int = 0

dc = ConfigDC(name="test")
print(type(dc))       # <class '__main__.ConfigDC'>
print(dc.name)        # "test" — 属性访问
# dc.name = "new"     # FrozenInstanceError — 不可变
# ConfigDC("test")    # TypeError — kw_only 强制关键字参数
```

### 实验 2：sys.intern 的性能效果

```python
import sys
import timeit

# 非驻留字符串比较
s1 = "__start__" * 100
s2 = "__start__" * 100
print(s1 is s2)  # True (CPython 小字符串驻留优化)

# 大字符串：intern 才有效
s1 = "a" * 4096
s2 = "a" * 4096
print(s1 is s2)  # False — CPython 不自动驻留大字符串

i1 = sys.intern(s1)
i2 = sys.intern(s2)
print(i1 is i2)  # True — 手动驻留后共享对象

# 性能对比
t_eq = timeit.timeit(lambda: s1 == s2, number=10_000_000)
t_is = timeit.timeit(lambda: i1 is i2, number=10_000_000)
print(f"== 耗时: {t_eq:.3f}s, is 耗时: {t_is:.3f}s")
# is 比较快 2-3 倍
```

### 实验 3：Literal 与类型检查

```python
from typing import Literal

StreamMode = Literal["values", "updates", "messages"]

# 运行时 StreamMode 就是字符串的联合类型提示
# 配合 pyright/mypy 使用：

def process(mode: StreamMode) -> None:
    # 类型检查器会拒绝非法值：
    # process("invalid")  # pyright 报错
    process("values")     # 合法
    # 但运行时不做检查——这是 Literal 的设计意图
    pass

# 结合 TypeAliasType 创建泛型别名
from typing import TypeVar, Generic
from typing_extensions import TypeAliasType, TypedDict

T = TypeVar("T")

class Part(TypedDict, Generic[T]):
    type: Literal["data"]
    data: T

MyPart = TypeAliasType("MyPart", Part[T], type_params=(T,))
# MyPart[int] 在类型检查时等价于 Part[int]
```

### 实验 4：模块级 __getattr__ 模拟

```python
# mymodule.py
from warnings import warn

_PUBLIC = {"START": "__start__", "END": "__end__"}

def __getattr__(name):
    if name in _PUBLIC:
        warn(f"Importing {name} is deprecated", stacklevel=2)
        return _PUBLIC[name]
    raise AttributeError(f"module has no attribute '{name}'")

# 测试
import mymodule
print(mymodule.START)    # 发出警告，返回 "__start__"
print(mymodule.MISSING)  # AttributeError
```

### 实验 5：MISSING 哨兵模式

```python
# 对比 None 和 MISSING 作为默认值

MISSING = object()

def search(data: dict, key: str, default=MISSING):
    """区分'未提供 default'和'默认值为 None'"""
    if key not in data:
        if default is MISSING:
            raise KeyError(key)
        return default
    return data[key]

# 用户可以显式传入 None 作为默认值
search({"a": 1}, "b", None)   # 返回 None
search({"a": 1}, "b")          # 抛出 KeyError
# 如果用 default=None，则无法区分两种情况
```