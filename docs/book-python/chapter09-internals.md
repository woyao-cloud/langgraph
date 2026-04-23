# 第九章：内部基础设施 — _internal/ 模块

LangGraph 的 `_internal/` 目录是整个框架的基石。它定义了常量、类型工具、配置合并策略、字段反射、可运行适配器和序列化白名单——这些基础设施横跨 Pregel 引擎、图编译器和运行时的每一个角落。本章将深入这些模块的每一行代码，揭示其中蕴含的 Python 进阶技巧与设计权衡。

---

## 1. Python 进阶

### 1.1 `sys.intern()` — 字符串驻留与内存优化

Python 内部使用字符串驻留（interning）技术：对同一字符串只保留一份引用，通过 `sys.intern()` 手动驻留的字符串在比较时直接使用 `id()` 而非逐字符对比，从而将 O(n) 的字符串比较降为 O(1) 的指针比较。LangGraph 在高频路径上大量使用驻留字符串作为字典键——例如 `configurable["__pregel_send"]` 这类查找每步都会发生，驻留后哈希表查找更快。

### 1.2 `_` 前缀约定 — 私有模块

`_internal/` 目录本身以 `_` 开头，向用户传达"这不是公共 API"的信号。Python 没有真正的访问控制，`_` 前缀是社区约定的私有标记，IDE 和 linter 会据此隐藏这些符号。

### 1.3 `{**a, **b}` — 字典合并表达式

Python 3.5+ 的字典解包语法 `{**a, **b}` 创建新字典，右侧覆盖左侧同名键。这比 `a.update(b)`（原地修改）更安全，符合不可变编程范式。LangGraph 在配置合并中大量使用此模式。

### 1.4 `object()` — 哨兵单例模式

`MISSING = object()` 创建一个唯一标识符，用于区分"未设置"与 `None`。因为每次 `object()` 调用返回的新实例的 `id` 唯一，`x is MISSING` 永远不会与任何合法值冲突。

### 1.5 `WeakKeyDictionary` + `@lru_cache` — 带弱引用的缓存

`weakref.WeakKeyDictionary` 以弱引用持有键，当键对象被垃圾回收时条目自动删除，避免内存泄漏。LangGraph 用它缓存类型注解的反射结果，因为类型对象的生命周期由模块系统管理，缓存不应阻止其释放。

### 1.6 `Annotated[T, metadata]` — 类型级元数据

`typing.Annotated` 允许在类型标注上附加任意元数据而不影响类型本身。LangGraph 用它声明 `IsLastStep = Annotated[bool, IsLastStepManager]`，将"如何计算此值"的逻辑编码在类型中，实现声明式的托管值系统。

### 1.7 `functools.partial` — 同步转异步

`functools.partial(run_in_executor, None, func)` 将同步函数包装为可调用的异步适配器，使 `RunnableCallable` 的 `afunc` 可以在异步上下文中安全地在线程池执行同步代码。

### 1.8 `inspect.signature` — 运行时函数内省

`inspect.signature(func).parameters` 在运行时提取函数的参数名、类型标注和种类（位置参数/关键字参数等），使 LangGraph 能够根据节点函数的签名自动注入 `config`、`store`、`writer` 等参数。

### 1.9 `__or__` 运算符重载 — 管道组合

`RunnableSeq.__or__` 重载了 `|` 运算符，使 `step1 | step2` 语法能够创建顺序执行的可运行管道，这是 LangChain 生态系统的标志性 API 设计。

---

## 2. 代码走读

### 2.1 `_constants.py` — 保留键与命名空间常量

```python
import sys
from typing import Literal, cast

INPUT = sys.intern("__input__")
INTERRUPT = sys.intern("__interrupt__")
RESUME = sys.intern("__resume__")
ERROR = sys.intern("__error__")
NO_WRITES = sys.intern("__no_writes__")
TASKS = sys.intern("__pregel_tasks")
RETURN = sys.intern("__return__")
PREVIOUS = sys.intern("__previous__")
```

所有保留写键（write key）均以 `sys.intern()` 驻留。`INPUT` 是图的输入通道，`INTERRUPT` 标记动态中断，`RESUME` 用于中断后的恢复值，`ERROR` 记录节点抛出的异常，`NO_WRITES` 是一个空写标记（表示节点没有产出任何写），`TASKS` 对应 `Send` 对象返回的推送任务，`RETURN` 直接记录返回值，`PREVIOUS` 处理控制流值。

```python
CONFIG_KEY_SEND = sys.intern("__pregel_send")
CONFIG_KEY_READ = sys.intern("__pregel_read")
CONFIG_KEY_CALL = sys.intern("__pregel_call")
CONFIG_KEY_CHECKPOINTER = sys.intern("__pregel_checkpointer")
CONFIG_KEY_STREAM = sys.intern("__pregel_stream")
CONFIG_KEY_CACHE = sys.intern("__pregel_cache")
CONFIG_KEY_RESUMING = sys.intern("__pregel_resuming")
CONFIG_KEY_REPLAY_STATE = sys.intern("__pregel_replay_state")
CONFIG_KEY_TASK_ID = sys.intern("__pregel_task_id")
CONFIG_KEY_THREAD_ID = sys.intern("thread_id")
CONFIG_KEY_CHECKPOINT_MAP = sys.intern("checkpoint_map")
CONFIG_KEY_CHECKPOINT_ID = sys.intern("checkpoint_id")
CONFIG_KEY_CHECKPOINT_NS = sys.intern("checkpoint_ns")
CONFIG_KEY_NODE_FINISHED = sys.intern("__pregel_node_finished")
CONFIG_KEY_SCRATCHPAD = sys.intern("__pregel_scratchpad")
CONFIG_KEY_RUNNER_SUBMIT = sys.intern("__pregel_runner_submit")
CONFIG_KEY_DURABILITY = sys.intern("__pregel_durability")
CONFIG_KEY_RUNTIME = sys.intern("__pregel_runtime")
CONFIG_KEY_RESUME_MAP = sys.intern("__pregel_resume_map")
```

这些 `CONFIG_KEY_*` 常量是注入到 `RunnableConfig["configurable"]` 字典中的键名。注意双下划线前缀（如 `__pregel_send`）标记它们是框架内部使用的，而 `thread_id`、`checkpoint_id` 等则是用户可见的配置键。

```python
PUSH = sys.intern("__pregel_push")
PULL = sys.intern("__pregel_pull")
NS_SEP = sys.intern("|")
NS_END = sys.intern(":")
CONF = cast(Literal["configurable"], sys.intern("configurable"))
NULL_TASK_ID = sys.intern("00000000-0000-0000-0000-000000000000")
OVERWRITE = sys.intern("__overwrite__")
```

`PUSH` 和 `PULL` 区分两种任务类型：推送式（由 `Send` 对象创建）和拉取式（由边触发）。`NS_SEP`（`|`）和 `NS_END`（`:`）构成检查点命名空间的编码格式，例如 `"graph|subgraph:task_id"`。`CONF` 通过 `cast` 将字符串 `"configurable"` 约束为 `Literal["configurable"]` 类型，使类型检查器确信这是 `RunnableConfig` 的合法键名。

`RESERVED` 集合汇总了所有保留标识符，用于运行时校验用户定义的键名不会与内部键冲突。

### 2.2 `_typing.py` — 哨兵值与类型别名

```python
from typing import Any, ClassVar, Protocol, TypeAlias
from pydantic import BaseModel
from typing_extensions import TypedDict

class TypedDictLikeV1(Protocol):
    __required_keys__: ClassVar[frozenset[str]]
    __optional_keys__: ClassVar[frozenset[str]]

class TypedDictLikeV2(Protocol):
    __required_keys__: frozenset[str]
    __optional_keys__: frozenset[str]
```

`TypedDictLikeV1` 和 `TypedDictLikeV2` 是两个 Protocol，描述"像 TypedDict"的类型。V1 用 `ClassVar` 声明类属性（Python 3.9 及以前 `TypedDict` 的行为），V2 不用 `ClassVar`（Python 3.10+ 的行为）。之所以需要两个版本，是因为不同 Python 版本下 `TypedDict` 的 `__required_keys__` 属性的语义不同。

```python
StateLike: TypeAlias = TypedDictLikeV1 | TypedDictLikeV2 | DataclassLike | BaseModel
```

`StateLike` 是 LangGraph 中"状态模式"的类型别名——它可以是 TypedDict、dataclass 或 Pydantic BaseModel。由于 `TypedDict` 和 `dataclass` 本身不是合法的类型标注目标，这里用 Protocol 和 TypeAlias 绕开了限制。

```python
MISSING = object()
EMPTY_SEQ: tuple[str, ...] = tuple()
```

`MISSING` 是经典哨兵模式——`object()` 每次调用返回唯一实例，`x is MISSING` 检测"值从未设置"，与 `None`（合法的空值）严格区分。`EMPTY_SEQ` 是一个不可变空元组单例，用于默认参数，避免 `[]` 默认参数陷阱。

```python
class DeprecatedKwargs(TypedDict):
    pass
```

`DeprecatedKwargs` 是一个空 TypedDict，用于标记函数的 `**kwargs` 参数已弃用。TypeScript 等语言有专门的 `@deprecated` 标注，而 Python 的 TypedDict 提供了一种类型级别的弃用信号——如果用户的代码中为这些关键字参数传值，类型检查器会因字段不在 TypedDict 定义中而发出警告。

### 2.3 `_fields.py` — 模式反射与 Annotated 解包

```python
ANNOTATED_KEYS_CACHE: weakref.WeakKeyDictionary[type[Any], tuple[str, ...]] = (
    weakref.WeakKeyDictionary()
)

def get_cached_annotated_keys(obj: type[Any]) -> tuple[str, ...]:
    if obj in ANNOTATED_KEYS_CACHE:
        return ANNOTATED_KEYS_CACHE[obj]
    if isinstance(obj, type):
        keys: list[str] = []
        for base in reversed(obj.__mro__):
            ann = base.__dict__.get("__annotations__")
            if ann is None:
                ann = getattr(base, "__annotations__", None)
            if ann is None or isinstance(ann, types.GetSetDescriptorType):
                continue
            keys.extend(ann.keys())
        return ANNOTATED_KEYS_CACHE.setdefault(obj, tuple(keys))
    else:
        raise TypeError(f"Expected a type, got {type(obj)}.")
```

`get_cached_annotated_keys` 从类型的 MRO（方法解析顺序）中收集所有注解键。它遍历 `reversed(obj.__mro__)` 以确保基类字段在前，子类字段在后。`WeakKeyDictionary` 保证当类型对象被卸载时缓存条目自动清理——如果用普通字典，模块重新加载会导致类型对象泄漏。

```python
def get_update_as_tuples(input: Any, keys: Sequence[str]) -> list[tuple[str, Any]]:
    if isinstance(input, BaseModel):
        keep = input.model_fields_set
        defaults = {k: v.default for k, v in type(input).model_fields.items()}
    else:
        keep = None
        defaults = {}
    return [
        (k, value)
        for k in keys
        if (value := getattr(input, k, MISSING)) is not MISSING
        and (value is not None
             or defaults.get(k, MISSING) is not None
             or (keep is not None and k in keep))
    ]
```

`get_update_as_tuples` 处理 Pydantic 模型和其他类型的差异。对于 Pydantic 模型，它用 `model_fields_set` 区分"用户显式设置的字段"和"使用默认值的字段"，只更新用户真正设置了或非默认的字段。这避免了 Pydantic 默认值意外覆盖图状态。

### 2.4 `_config.py` — 配置合并与注入

```python
def patch_configurable(
    config: RunnableConfig | None, patch: dict[str, Any]
) -> RunnableConfig:
    if config is None:
        return {CONF: patch}
    elif CONF not in config:
        return {**config, CONF: patch}
    else:
        return {**config, CONF: {**config[CONF], **patch}}
```

`patch_configurable` 用三层条件处理配置合并：(1) 无配置则创建新的；(2) 有配置但无 `configurable` 键则添加；(3) 已有 `configurable` 则用 `{**old, **new}` 合并，新值覆盖旧值。始终返回新字典，不修改原始对象。

```python
def merge_configs(*configs: RunnableConfig | None) -> RunnableConfig:
    base: RunnableConfig = {}
    for config in configs:
        if config is None:
            continue
        for key, value in config.items():
            if not value:
                continue
            if key == "metadata":
                if base_value := base.get(key):
                    base[key] = {**base_value, **value}
                else:
                    base[key] = value
            elif key == "tags":
                if base_value := base.get(key):
                    base[key] = [*base_value, *value]
                else:
                    base[key] = value
            elif key == CONF:
                if base_value := base.get(key):
                    base[key] = {**base_value, **value}
                else:
                    base[key] = value
            elif key == "callbacks":
                # ... 6 cases of callback merging ...
```

`merge_configs` 是配置系统的核心合并函数，它为不同类型的键采用不同策略：
- **metadata**：字典合并（`{**base, **override}`），新元数据覆盖旧的同名键
- **tags**：列表拼接（`[*base, *override]`），保留所有标签
- **configurable**：字典合并，子图可以覆盖父图的配置
- **callbacks**：最复杂的合并——有 6 种情况（None/列表/管理器的笛卡尔积），确保回调处理器不丢失
- **recursion_limit**：仅在非默认值时覆盖

```python
_OMIT = ("key", "token", "secret", "password", "auth")

def _exclude_as_metadata(key: str, value: Any, metadata: Mapping[str, Any]) -> bool:
    key_lower = key.casefold()
    return (
        key.startswith("__")
        or not isinstance(value, (str, int, float, bool))
        or key in metadata
        or any(substr in key_lower for substr in _OMIT)
    )
```

`_exclude_as_metadata` 决定哪些 `configurable` 条目应该提升到 `metadata`。它排除了：(1) 以 `__` 开头的内部键；(2) 非标量值；(3) 已存在于 metadata 的键；(4) 包含敏感词（key/token/secret/password/auth）的键。这是一个安全措施，防止将 API 密钥等敏感信息泄漏到追踪元数据中。

### 2.5 `_runnable.py` — 可运行适配器与管道组合

```python
KWARGS_CONFIG_KEYS: tuple[tuple[str, tuple[Any, ...], str, Any], ...] = (
    ("config", (RunnableConfig, "RunnableConfig", ...), "N/A", inspect.Parameter.empty),
    ("writer", (StreamWriter, "StreamWriter", ...), "stream_writer", lambda _: None),
    ("store", (BaseStore, "BaseStore", ...), "store", inspect.Parameter.empty),
    ("store", (Optional[BaseStore], "Optional[BaseStore]"), "store", None),
    ("previous", (ANY_TYPE,), "previous", inspect.Parameter.empty),
    ("runtime", (ANY_TYPE,), "N/A", inspect.Parameter.empty),
)
```

`KWARGS_CONFIG_KEYS` 是一个注册表，定义了哪些关键字参数可以在运行时从 `Runtime` 中注入。每个元组包含：(参数名, 可接受的类型标注, Runtime 属性名, 默认值)。`ANY_TYPE = object()` 作为通配符，表示接受任何类型标注。

```python
class RunnableCallable(Runnable):
    def __init__(self, func, afunc=None, *, name=None, trace=True, recurse=True, ...):
        self.func_accepts: dict[str, tuple[str, Any]] = {}
        params = inspect.signature(cast(Callable, func or afunc)).parameters
        for kw, typ, runtime_key, default in KWARGS_CONFIG_KEYS:
            p = params.get(kw)
            if p is None or p.kind not in VALID_KINDS:
                continue
            if typ != (ANY_TYPE,) and p.annotation not in typ:
                continue
            self.func_accepts[kw] = (runtime_key, default)
```

`RunnableCallable.__init__` 使用 `inspect.signature` 检查函数参数。对于每个在 `KWARGS_CONFIG_KEYS` 中注册的关键字参数，它检查目标函数是否声明了该参数且类型标注匹配。匹配的参数被记录在 `func_accepts` 字典中，运行时从 `Runtime` 提取对应值注入。

```python
def invoke(self, input, config=None, **kwargs):
    runtime = config.get(CONF, {}).get(CONFIG_KEY_RUNTIME)
    for kw, (runtime_key, default) in self.func_accepts.items():
        if kw in kwargs:
            continue
        kw_value: Any = MISSING
        if kw == "config":
            kw_value = config
        elif runtime:
            if kw == "runtime":
                kw_value = runtime
            else:
                try:
                    kw_value = getattr(runtime, runtime_key)
                except AttributeError:
                    pass
        if kw_value is MISSING:
            if default is inspect.Parameter.empty:
                raise ValueError(...)
            kw_value = default
        kwargs[kw] = kw_value
```

`invoke` 的核心注入逻辑：从 `config["configurable"]["__pregel_runtime"]` 获取 `Runtime` 对象，然后按 `func_accepts` 的映射将属性提取为关键字参数。`config` 是特例——直接传整个配置对象。如果参数缺失且无默认值，抛出 `ValueError`。

```python
def coerce_to_runnable(thing: RunnableLike, *, name, trace) -> Runnable:
    if isinstance(thing, Runnable):
        return thing
    elif is_async_generator(thing) or inspect.isgeneratorfunction(thing):
        return RunnableLambda(thing, name=name)
    elif callable(thing):
        if is_async_callable(thing):
            return RunnableCallable(None, thing, name=name, trace=trace)
        else:
            return RunnableCallable(
                thing,
                wraps(thing)(partial(run_in_executor, None, thing)),
                name=name, trace=trace,
            )
    elif isinstance(thing, dict):
        return RunnableParallel(thing)
    else:
        raise TypeError(...)
```

`coerce_to_runnable` 是一个多态适配器：已经是 `Runnable` 的直接返回；生成器函数包装为 `RunnableLambda`；异步可调用对象只设 `afunc`；同步可调用对象同时设 `func` 和用 `functools.partial(run_in_executor, None, func)` 包装的 `afunc`——后者将同步函数提交到线程池执行，避免阻塞事件循环。

```python
class RunnableSeq(Runnable):
    def __or__(self, other):
        if isinstance(other, RunnableSequence):
            return RunnableSeq(*self.steps, other.first, *other.middle, other.last, ...)
        elif isinstance(other, RunnableSeq):
            return RunnableSeq(*self.steps, *other.steps, ...)
        else:
            return RunnableSeq(*self.steps, coerce_to_runnable(other, ...), ...)
```

`RunnableSeq.__or__` 实现了 `|` 运算符，让 `step1 | step2 | step3` 自然地组合执行管道。它智能处理不同类型的右侧操作数：`RunnableSequence` 拆解为 first/middle/last 三部分拼接，`RunnableSeq` 直接展开步序列，其他类型先通过 `coerce_to_runnable` 适配。

### 2.6 `_serde.py` — 序列化白名单与循环检测

```python
def curated_core_allowlist() -> set[tuple[str, ...]]:
    allowlist: set[tuple[str, ...]] = set()
    for name in ("BaseMessage", "HumanMessage", "AIMessage", ...):
        cls = getattr(lc_messages, name, None)
        if cls is None:
            continue
        allowlist.add((cls.__module__, cls.__name__))
    return allowlist
```

`curated_core_allowlist` 收集 LangChain 消息类型的 `(module, name)` 元组。用 `getattr` + `None` 检查实现优雅降级——如果某个消息类不存在（例如旧版本），直接跳过而不报错。

```python
def collect_allowlist_from_schemas(*, schemas, channels):
    allowlist: set[tuple[str, ...]] = set()
    seen: set[Any] = set()
    seen_ids: set[int] = set()
    if schemas:
        for schema in schemas:
            _collect_from_type(schema, allowlist, seen, seen_ids)
    if channels:
        for channel in channels.values():
            value_type = getattr(channel, "ValueType", None)
            if value_type is not None:
                _collect_from_type(value_type, allowlist, seen, seen_ids)
            update_type = getattr(channel, "UpdateType", None)
            if update_type is not None:
                _collect_from_type(update_type, allowlist, seen, seen_ids)
    return allowlist
```

`collect_allowlist_from_schemas` 是序列化安全的基石。它从状态模式和通道定义中递归收集所有需要序列化的类型，确保只有已知类型能被 msgpack 序列化。

```python
def _collect_from_type(typ, allowlist, seen, seen_ids):
    if _already_seen(typ, seen, seen_ids):
        return
    if typ is Any or typ is None or typ is Literal:
        return
    if isinstance(typ, types.UnionType):
        for arg in typ.__args__:
            _collect_from_type(arg, allowlist, seen, seen_ids)
        return
    origin = get_origin(typ)
    if origin is Union:
        for arg in get_args(typ):
            _collect_from_type(arg, allowlist, seen, seen_ids)
        return
    if origin is Annotated or origin in (Required, NotRequired):
        args = get_args(typ)
        if args:
            _collect_from_type(args[0], allowlist, seen, seen_ids)
        return
    if origin in (list, set, tuple, dict, deque, frozenset):
        for arg in get_args(typ):
            _collect_from_type(arg, allowlist, seen, seen_ids)
        return
    if hasattr(typ, "__supertype__"):
        _collect_from_type(typ.__supertype__, allowlist, seen, seen_ids)
        return
    if is_typeddict(typ):
        for field_type in _safe_get_type_hints(typ).values():
            _collect_from_type(field_type, allowlist, seen, seen_ids)
        return
    if _is_pydantic_model(typ):
        allowlist.add((typ.__module__, typ.__name__))
        for field_type in _safe_get_type_hints(typ).values():
            _collect_from_type(field_type, allowlist, seen, seen_ids)
        return
    if dataclasses.is_dataclass(typ):
        if typ_name := getattr(typ, "__name__", None):
            allowlist.add((typ.__module__, typ_name))
        for field_type in _safe_get_type_hints(typ).values():
            _collect_from_type(field_type, allowlist, seen, seen_ids)
        return
    if isinstance(typ, type) and issubclass(typ, Enum):
        allowlist.add((typ.__module__, typ.__name__))
        return
```

`_collect_from_type` 是一个递归类型遍历器，处理 `Union`、`Annotated`、容器类型、TypedDict、Pydantic 模型、dataclass 和枚举。每种类型做不同处理：容器类型只递归进入参数类型（不将 `list` 本身加入白名单），具体类型则将自身加入白名单后递归其字段类型。

```python
def _already_seen(typ, seen, seen_ids):
    try:
        if typ in seen:
            return True
        seen.add(typ)
        return False
    except TypeError:
        typ_id = id(typ)
        if typ_id in seen_ids:
            return True
        seen_ids.add(typ_id)
        return False
```

`_already_seen` 实现了循环检测。某些类型（如带 `__eq__` 的自定义类）可能无法哈希，因此先尝试将类型加入 `seen` 集合，如果失败（`TypeError`）则回退到使用 `id()` 的 `seen_ids` 集合。双重集合策略确保了任意类型图的安全性。

---

## 3. 运行原理

### 3.1 配置注入流水线

```
用户调用 graph.invoke(input, config)
          │
          ▼
    Pregel 初始化
    将 send/read/call/runtime 等注入 config["configurable"]
          │
          ▼
    节点执行时，RunnableCallable.invoke() 被调用
          │
          ▼
    检查 func_accepts，从 config["configurable"] 中提取对应值
          │
          ▼
    将提取的值作为 kwargs 传给用户定义的节点函数
```

配置注入的关键路径：Pregel 在初始化时通过 `patch_config` 将 `send`、`read`、`call`、`Runtime` 等函数和对象注入到 `config["configurable"]` 中。然后 `RunnableCallable.invoke()` 通过 `inspect.signature` 反射知道函数需要哪些参数，从 `Runtime` 对象中提取并注入。

### 3.2 Annotated 模式反射

```
StateSchema (TypedDict/dataclass/BaseModel)
          │
          ▼
    get_cached_annotated_keys() 遍历 MRO 收集字段名
          │
          ▼
    get_type_hints() 获取每个字段的类型标注
          │
          ▼
    get_origin() 检查是否为 Annotated
          │
          ▼
    get_args() 解包得到 (基础类型, reducer/validator)
          │
          ▼
    用 Annotated 的元数据作为 reducer 函数
```

当用户定义 `Annotated[int, operator.add]` 时，LangGraph 通过 `get_origin()` 检测到 `Annotated`，再通过 `get_args()` 解包为 `(int, operator.add)`，将 `operator.add` 作为该字段的 reducer（更新函数）。

### 3.3 序列化白名单保证

```
StateSchema + Channel 定义
          │
          ▼
    collect_allowlist_from_schemas() 递归遍历
          │
          ▼
    _collect_from_type() 对每个类型递归展开
          │
          ▼
    生成 (module, name) 元组集合
          │
          ▼
    传给 checkpointer.with_allowlist()
          │
          ▼
    序列化时只允许白名单中的类型
```

白名单机制确保了反序列化安全——只有从代码中实际定义的类型才能被反序列化，防止任意代码执行攻击。

---

## 4. 实现细节

### 4.1 WeakKeyDictionary 缓存策略

`ANNOTATED_KEYS_CACHE` 使用 `WeakKeyDictionary` 而非普通字典。原因在于：如果模块热重载或类型被动态创建后销毁，普通字典会持有强引用导致内存泄漏。`WeakKeyDictionary` 在键对象被垃圾回收时自动删除条目，但由于类型对象通常由模块系统持有引用，缓存实际上很稳定——只在模块卸载时才清理。

### 4.2 merge_configs 的回调合并六宫格

`callbacks` 字段有三种可能类型：`None`、`list[handler]`、`BaseCallbackManager`。两个值的笛卡尔积产生 9 种情况，但代码用 `isinstance` 分支处理了 6 种（将 `None` 合并到任何类型简化为直接取另一值）。核心原则是：合并后必须包含两个来源的所有回调处理器，且以管理器形式存在以便传播。

### 4.3 _exclude_as_metadata 的安全过滤

将 `configurable` 中的条目提升到 `metadata`（用于追踪）时，必须排除敏感信息。`_OMIT` 元组定义了关键字黑名单（key、token、secret、password、auth），用 `casefold()` 做大小写不敏感匹配。双下划线前缀的内部键也被排除——它们是框架实现细节，不应暴露给用户。

### 4.4 coerce_to_runnable 的同步转异步策略

对于同步函数，`coerce_to_runnable` 同时创建 `func`（原始同步函数）和 `afunc`（`partial(run_in_executor, None, func)` 包装后的版本）。`run_in_executor` 的第一个参数为 `None` 时使用默认线程池。这样 `ainvoke` 路径不会阻塞事件循环。

### 4.5 collect_allowlist_from_schemas 的循环检测

递归类型如 `class Node: children: list[Node]` 会导致无限递归。`_already_seen` 使用双重集合策略：对可哈希类型用 `seen` 集合（O(1) 查找），对不可哈希类型用 `seen_ids` 集合（基于 `id()` 的 O(1) 查找）。这确保了任意类型图的安全性。

### 4.6 RunnableSeq 的上下文传播

`RunnableSeq.invoke` 只对第一步（实际节点）设置上下文（`set_config_context`），后续步骤（写入器等）不需要上下文。这是因为上下文传播的成本较高——它需要复制整个 `ContextVar` 状态。在 Python 3.11+ 中，`asyncio.create_task(coro, context=context)` 原生支持上下文传播；而在更早版本中，上下文无法在 `async` 函数间自动传播。

---

## 5. 动手实验

### 实验 1：TypedDict 类型内省

```python
from typing import get_type_hints, get_origin, get_args
from typing_extensions import TypedDict, Annotated
import operator

class State(TypedDict, total=False):
    messages: Annotated[list[str], operator.add]
    count: int

hints = get_type_hints(State, include_extras=True)
for name, typ in hints.items():
    print(f"Field: {name}")
    print(f"  Full type: {typ}")
    if get_origin(typ) is Annotated:
        base, *metadata = get_args(typ)
        print(f"  Base type: {base}")
        print(f"  Metadata: {metadata}")

# 输出:
# Field: messages
#   Full type: Annotated[list[str], operator.add]
#   Base type: list[str]
#   Metadata: [<built-in function add>]
# Field: count
#   Full type: <class 'int'>
```

### 实验 2：Annotated 元数据提取

```python
from typing import Annotated, get_origin, get_args

IsLastStep = Annotated[bool, "is_last_step_manager"]

origin = get_origin(IsLastStep)
print(f"Origin: {origin}")  # Annotated

args = get_args(IsLastStep)
print(f"Args: {args}")  # (bool, 'is_last_step_manager')
print(f"Base type: {args[0]}")  # bool
print(f"Metadata: {args[1:]}")  # ('is_last_step_manager',)
```

### 实验 3：WeakKeyDictionary 缓存行为

```python
import weakref

cache = weakref.WeakKeyDictionary()

class MyState:
    name: str

s1 = MyState()
cache[s1] = ("name",)
print(f"Cached keys for s1: {cache[s1]}")  # ('name',)

# 删除唯一引用，对象被垃圾回收
del s1
import gc; gc.collect()
print(f"Cache empty: {len(cache) == 0}")  # True

# 对比：普通字典会阻止垃圾回收
normal_cache = {}
s2 = MyState()
normal_cache[s2] = ("name",)
del s2  # s2 不会被回收，因为 normal_cache 持有强引用
gc.collect()
print(f"Normal dict still has entry: {len(normal_cache) > 0}")  # True
```

### 实验 4：RunnableSeq 管道组合

```python
from langgraph._internal._runnable import RunnableSeq, coerce_to_runnable

def double(x):
    return x * 2

def add_one(x):
    return x + 1

r_double = coerce_to_runnable(double, name="double", trace=False)
r_add = coerce_to_runnable(add_one, name="add_one", trace=False)

# 使用 | 运算符创建管道
pipeline = r_double | r_add  # 先加倍再加一

# 注意：RunnableSeq 需要在上下文中运行
# 这里仅演示组合机制
print(f"Steps: {len(pipeline.steps)}")  # 2
print(f"Step 0: {pipeline.steps[0].name}")  # double
print(f"Step 1: {pipeline.steps[1].name}")  # add_one
```

### 实验 5：sys.intern 字符串性能对比

```python
import sys
import timeit

# 驻留字符串的比较速度
s1 = sys.intern("__pregel_send")
s2 = sys.intern("__pregel_send")

# 普通字符串的比较速度
s3 = "__pregel_send"
s4 = "__pregel_send"

interned_time = timeit.timeit(lambda: s1 == s2, number=10_000_000)
normal_time = timeit.timeit(lambda: s3 == s4, number=10_000_000)

print(f"Interned string comparison: {interned_time:.4f}s")
print(f"Normal string comparison: {normal_time:.4f}s")
print(f"Speedup: {normal_time / interned_time:.2f}x")

# 在字典查找中效果更明显
d = {sys.intern(f"__key_{i}"): i for i in range(1000)}
key = sys.intern("__key_500")
interned_lookup = timeit.timeit(lambda: d[key], number=10_000_000)
print(f"Interned dict lookup: {interned_lookup:.4f}s")
```