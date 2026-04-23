# 第九章 内部基础设施 — _internal/ 模块

## 9.1 功能概览

`_internal/` 目录是 LangGraph 的"引擎室"——用户永远不会直接导入它，但框架的每一行代码都依赖它。它包含六个核心模块：

| 模块 | 职责 |
|------|------|
| `_constants.py` | 保留键名、配置键、命名空间分隔符等全局常量 |
| `_typing.py` | 哨兵值（MISSING）、协议类型（StateLike）、类型别名 |
| `_fields.py` | Annotated 键提取、默认值推断、Pydantic 更新拆包 |
| `_config.py` | 配置合并（merge_configs）、配置补丁（patch_config）、默认配置生成 |
| `_runnable.py` | RunnableCallable、RunnableSeq、coerce_to_runnable——管道适配层 |
| `_serde.py` | 反序列化安全白名单——递归类型收集与 allowlist 构建 |

这些模块构成了框架的"暗物质"：你看不到它们，但它们决定了 State 的键名冲突检测、config 的层层传递、`a | b` 管道语法的编译、以及 checkpoint 反序列化的安全性。

## 9.2 应用场景

### 场景一：理解保留键名，避免 State 命名冲突

LangGraph 内部使用双下划线前缀（如 `__input__`、`__interrupt__`）作为保留写通道。如果你在 State 中定义了同名键，会与框架冲突：

```python
from typing_extensions import TypedDict

# 危险！这些键名被框架占用
class BadState(TypedDict):
    __input__: str       # 与 _constants.INPUT 冲突
    __resume__: dict     # 与 _constants.RESUME 冲突
    tasks: list          # 安全，但注意 "__pregel_tasks" 是保留的

# 安全做法：避免以 __ 开头的键名
class GoodState(TypedDict):
    user_input: str      # 安全
    resume_data: dict    # 安全
    task_list: list      # 安全
```

`RESERVED` 集合包含所有保留名称，编译时 Pregel 会检查你的 State 键是否与此集合冲突。

### 场景二：Config 注入使 get_config()/get_stream_writer() 成为可能

在节点函数内部调用 `get_config()` 时，实际上是从 `var_child_runnable_config` 这个 ContextVar 中读取当前上下文的配置。这个注入管道的关键环节就在 `_runnable.py` 中：

```python
from langgraph.config import get_config, get_stream_writer

def my_node(state: dict):
    # 获取 thread_id 等运行时信息
    config = get_config()
    thread_id = config["configurable"]["thread_id"]

    # 使用 stream_writer 输出自定义流数据
    writer = get_stream_writer()
    writer({"progress": "50%", "step": "analyzing"})

    return {"result": "done"}
```

管道流程：Pregel 在调度节点时，将 Runtime 对象写入 `config["configurable"]["__pregel_runtime"]`，然后 `RunnableCallable.invoke` 在调用节点前，从 config 中提取 writer、store 等并注入为关键字参数。

### 场景三：Annotated[type, reducer] 的编译时内省

当你在 State 中写 `Annotated[list, add]` 时，`_fields.py` 的 `get_cached_annotated_keys()` 在编译图时被调用，遍历 MRO 提取所有注解键，然后用 `WeakKeyDictionary` 缓存结果，避免重复反射：

```python
from typing import Annotated
from operator import add
from typing_extensions import TypedDict

class MyState(TypedDict):
    messages: Annotated[list, add]   # 编译时 get_cached_annotated_keys 提取 "messages"
    count: int
```

`WeakKeyDictionary` 的好处是：当你的 State 类被垃圾回收后，缓存条目自动消失，不会造成内存泄漏。

### 场景四：Runnable 管道组合（a | b）的底层机制

LangChain 生态最优雅的 API 之一是管道操作符 `|`。当你写 `prompt | llm | parser` 时，实际上是调用了 `RunnableSeq.__or__` 方法：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 这行代码的底层实现
chain = prompt | llm | parser

# 等价于
chain = RunnableSeq(prompt, llm, parser)
# 每个非 Runnable 对象通过 coerce_to_runnable 转换
```

`coerce_to_runnable` 的转换逻辑：若已是 Runnable 则直接返回；若是生成器函数则包装为 `RunnableLambda`；若是普通 callable 则包装为 `RunnableCallable`（同时自动创建 async 版本用 `run_in_executor`）。

### 场景五：同步转异步包装与性能影响

`coerce_to_runnable` 对同步函数的处理值得关注——它会自动用 `functools.partial(run_in_executor, None, func)` 创建异步版本。这意味着在 `ainvoke` 时，同步节点会被放到线程池执行：

```python
# coerce_to_runnable 中对同步 callable 的处理
return RunnableCallable(
    thing,  # 同步版本
    wraps(thing)(partial(run_in_executor, None, thing)),  # 异步版本
    name=name,
    trace=trace,
)
```

**性能影响**：`run_in_executor` 会将同步函数放到默认线程池（最多 `min(32, os.cpu_count() + 4)` 个线程）。如果你的节点是 I/O 密集型，建议直接写 async 版本以避免线程池竞争。

### 场景六：Serde 白名单防御反序列化攻击

当 LangGraph 将 checkpoint 写入 Postgres/SQLite 时，需要序列化 State 中的对象。反序列化时如果不对类型做限制，恶意构造的 pickle 数据可以执行任意代码。`_serde.py` 的白名单机制只允许反序列化在 State schema 和 channels 中声明过的 Pydantic 模型和 dataclass：

```python
# build_serde_allowlist 会递归遍历你的 State schema
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int

class MyState(BaseModel):
    profile: UserProfile   # UserProfile 会被自动加入白名单
    score: int
```

白名单以 `(module_name, class_name)` 元组形式存储，如 `{('my_app.models', 'UserProfile')}`。反序列化时，任何不在白名单中的类型都会被拒绝。

### 场景七：理解 checkpoint 序列化失败的根源

某些 Python 类型天生无法被 msgpack/json 安全序列化：lambda 函数、打开的文件句柄、数据库连接、自定义 C 扩展对象。当你在 State 中放入这些值并使用 checkpointer 时，序列化会失败。理解 `_serde.py` 的类型收集逻辑可以帮助你排查：它只识别 TypedDict、Pydantic 模型、dataclass、Enum、以及标准容器（list/dict/tuple/set/deque/frozenset），如果你的类型不在这些类别中，就不会被加入白名单。

## 9.3 Python 进阶

### sys.intern() — 字符串驻留

`_constants.py` 中所有常量都使用 `sys.intern()` 驻留。驻留的字符串在全局只有一个副本，比较时可以直接用 `is` 而非 `==`，速度更快。在框架中，这些常量会被频繁比较（每次节点执行时读取 config），驻留能显著减少内存和比较开销。

### _prefix 约定

以 `_` 开头的模块名表示内部 API，不对外暴露。这是 Python 社区的惯例——`from langgraph._internal import ...` 是不被支持的用法。

### {**a, **b} 字典合并

`_config.py` 大量使用 `{**base, **override}` 语法创建合并后的新字典，而非原地修改。这是不可变模式的实现，确保原始配置不被污染。

### functools.partial

`coerce_to_runnable` 中用 `partial(run_in_executor, None, func)` 将"在线程池中执行 func"这个行为打包为可调用对象，作为 `afunc` 传入 `RunnableCallable`。

### object() 哨兵值

`MISSING = object()` 是经典的 Python 哨兵模式。用 `object()` 创建的实例保证全局唯一，用 `is` 判断时不会与任何合法值冲突（包括 `None`）。

### TypedDict 用于 kwargs

`DeprecatedKwargs(TypedDict)` 展示了一个有趣用法：用 TypedDict 为关键字参数提供类型提示，使 IDE 能对已弃用的参数发出警告。

### typing.get_type_hints() 与 get_origin/get_args

`_fields.py` 和 `_serde.py` 大量使用 `get_type_hints()` 解析类型的完整注解（包括 forward reference 解析），用 `get_origin`/`get_args` 拆解泛型（如 `Annotated[list, add]` → origin=Annotated, args=(list, add)）。

### WeakKeyDictionary 与缓存

`ANNOTATED_KEYS_CACHE = weakref.WeakKeyDictionary()` 是高效缓存方案：键是类型对象，当类型被垃圾回收时缓存条目自动清除。这避免了长期持有类型引用导致的内存泄漏。

### Annotated[T, metadata]

PEP 593 引入的 `Annotated` 类型让开发者可以在类型注解中附加元数据。LangGraph 利用它来声明 reducer：`Annotated[list, add]` 中 `add` 就是 reducer 函数，在运行时被提取使用。

### __or__ 运算符重载

`RunnableSeq.__or__` 实现了 `a | b` 语法。当左操作数是 `RunnableSeq` 时，展开步骤合并；当右操作数是普通对象时，先通过 `coerce_to_runnable` 转换再追加。

### inspect.signature

`RunnableCallable.__init__` 使用 `inspect.signature(cast(Callable, func or afunc)).parameters` 来检查函数签名，判断节点是否接受 `config`、`writer`、`store` 等特殊参数，实现自动注入。

### functools.partial(run_in_executor)

同步函数自动包装为异步版本的核心机制：`partial(run_in_executor, None, func)` 创建一个接受 event_loop 和 input 的可调用对象，内部将 func 提交到线程池执行。

## 9.4 代码走读

### _constants.py — 全局常量定义

```python
INPUT = sys.intern("__input__")        # 输入值写通道
INTERRUPT = sys.intern("__interrupt__") # 动态中断写通道
RESUME = sys.intern("__resume__")       # 恢复值写通道
ERROR = sys.intern("__error__")         # 错误写通道
NO_WRITES = sys.intern("__no_writes__") # 标记节点无写入
TASKS = sys.intern("__pregel_tasks")    # Send 对象写通道
RETURN = sys.intern("__return__")       # 记录任务返回值
PREVIOUS = sys.intern("__previous__")   # Control 值隐式分支
```

配置键以 `CONFIG_KEY_` 前缀，存储在 `config["configurable"]` 字典中：

```python
CONFIG_KEY_SEND = sys.intern("__pregel_send")       # 写函数
CONFIG_KEY_READ = sys.intern("__pregel_read")       # 读函数
CONFIG_KEY_CHECKPOINTER = sys.intern("__pregel_checkpointer")
CONFIG_KEY_RUNTIME = sys.intern("__pregel_runtime") # Runtime 实例
CONFIG_KEY_TASK_ID = sys.intern("__pregel_task_id")
CONFIG_KEY_THREAD_ID = sys.intern("thread_id")
```

命名空间常量定义了 checkpoint 命名空间的编码格式：

```python
NS_SEP = sys.intern("|")  # 层级分隔符: graph|subgraph|subsubgraph
NS_END = sys.intern(":")  # 命名空间与 task_id 分隔: graph:task_id
```

`RESERVED` 集合包含所有保留键名，编译时用于检测命名冲突。

### _typing.py — 类型工具

```python
MISSING = object()  # 全局唯一哨兵值，用于"未设置"语义
EMPTY_SEQ: tuple[str, ...] = tuple()  # 空字符串序列，用作默认值

class DeprecatedKwargs(TypedDict):
    """用 TypedDict 为弃用的关键字参数提供类型检查"""
    pass

StateLike: TypeAlias = TypedDictLikeV1 | TypedDictLikeV2 | DataclassLike | BaseModel
```

`StateLike` 是一个联合类型别名，表示"可以作为 State schema 的类型"——TypedDict、dataclass 或 Pydantic BaseModel。由于 Python 类型系统的限制，无法直接用 `TypedDict` 作为类型约束，因此定义了 `TypedDictLikeV1/V2` 两个 Protocol 来匹配 TypedDict 的结构。

### _config.py — 配置操作

`merge_configs(*configs)` 是核心函数，按优先级合并多个 RunnableConfig：

- `metadata`：字典合并（后者覆盖前者）
- `tags`：列表拼接
- `configurable`：字典合并
- `callbacks`：六种组合情况（None/list/manager 两两配对）
- `recursion_limit`：非默认值才覆盖

`patch_configurable(config, patch)` 用 `{**config[CONF], **patch}` 实现不可变式补丁，确保原始 config 不被修改。

`ensure_config(*configs)` 生成一个包含所有默认值的完整配置，从 ContextVar 中继承父级配置，并将非标准键移入 `configurable`。

`_exclude_as_metadata` 实现了智能过滤：以 `__` 开头的键、包含 `key`/`token`/`secret` 等敏感词的键不会暴露到 metadata 中。

### _fields.py — 字段内省

`get_cached_annotated_keys(obj)` 遍历 MRO 提取所有注解键：

```python
def get_cached_annotated_keys(obj: type[Any]) -> tuple[str, ...]:
    if obj in ANNOTATED_KEYS_CACHE:
        return ANNOTATED_KEYS_CACHE[obj]  # 缓存命中
    keys = []
    for base in reversed(obj.__mro__):  # 从基类到派生类
        ann = base.__dict__.get("__annotations__")
        if ann is None or isinstance(ann, types.GetSetDescriptorType):
            continue
        keys.extend(ann.keys())
    return ANNOTATED_KEYS_CACHE.setdefault(obj, tuple(keys))
```

`get_field_default(name, type_, schema)` 根据 Required/NotRequired/Optional/total=False 等多种标注推断字段默认值，处理 TypedDict 和 dataclass 两种语义。

`get_update_as_tuples(input, keys)` 专门处理 Pydantic 模型的更新逻辑：只更新 `model_fields_set` 中的字段或与默认值不同的字段，避免意外覆盖。

### _runnable.py — Runnable 适配层

`RunnableCallable` 是简化版 `RunnableLambda`，关键特性：

1. 同时持有 `func`（同步）和 `afunc`（异步）两个版本
2. 构造时用 `inspect.signature` 检查参数签名，将可注入的参数（config/writer/store/previous/runtime）记录到 `func_accepts` 字典
3. `invoke`/`ainvoke` 时从 config 的 Runtime 中提取值并注入为关键字参数
4. 使用 `copy_context()` 确保上下文传播正确

`RunnableSeq` 是简化版 `RunnableSequence`，通过 `__or__` 和 `__ror__` 支持 `|` 操作符。关键优化：第一步在 context 中执行（确保 ContextVar 传播），后续步骤直接执行（writer 不需要 context）。

`coerce_to_runnable(thing)` 将任意可调用对象转换为 Runnable，其中对同步 callable 的自动异步包装是通过 `partial(run_in_executor, None, thing)` 实现的。

### _serde.py — 序列化安全

`build_serde_allowlist(schemas, channels)` 是安全模型的核心入口：

1. 先调用 `curated_core_allowlist()` 加入 LangChain 消息类型的白名单
2. 再调用 `collect_allowlist_from_schemas()` 递归遍历用户的 State schema

`_collect_from_type` 是递归类型收集器，按类型分派处理：

- `Union`/`UnionType`：递归处理每个分支
- `Annotated`/`Required`/`NotRequired`：解包后递归处理第一个参数
- 标准容器（list/dict/tuple/set/deque/frozenset）：递归处理类型参数
- TypedDict：递归处理所有字段类型
- Pydantic/dataclass：加入白名单并递归处理字段
- Enum：加入白名单

`_already_seen` 使用 `seen` 集合检测循环引用。某些类型不可哈希（如带 forward reference 的泛型），此时回退到 `id(typ)` 判重。

## 9.5 实现原理

### Config 注入管道

完整的数据流如下：

```
用户调用 graph.invoke(input, config)
    → Pregel.stream() 创建 ensure_config() 并写入 Runtime
    → Pregel._prepare_tasks() 将 config 传递给每个 task
    → RunnableCallable.invoke() 从 config 中提取 Runtime
    → 根据 func_accepts 字典，将 config/writer/store 等注入为 kwargs
    → 调用 node(state, **kwargs)
```

`var_child_runnable_config` 是一个 `ContextVar`，在 `set_config_context` 上下文管理器中设置。`get_config()` 读取这个 ContextVar 来获取当前配置。Python 3.11+ 的 `asyncio.create_task(coro, context=ctx)` 确保 ContextVar 在异步任务间正确传播。

### Schema 内省的编译时缓存

当 `StateGraph(my_state).compile()` 被调用时，`get_cached_annotated_keys` 被触发。它遍历 MRO 而非简单的 `__annotations__`，因为子类可能继承父类的注解。`WeakKeyDictionary` 缓存确保同一类型只内省一次，且类型被回收后缓存自动清除。在 Python 3.14+ 中，Pydantic 模型的 `__annotations__` 变成了描述符（descriptor），代码通过 `getattr(base, "__annotations__", None)` 回退处理。

### Serde 白名单安全模型

白名单的设计哲学是"最小权限"：只允许反序列化用户 State schema 中显式声明的类型。这防止了恶意构造的序列化数据在反序列化时实例化任意类。`curated_core_allowlist()` 硬编码了 LangChain 的 15 种消息类型，`collect_allowlist_from_schemas` 则根据用户的 State 类型定义动态扩展白名单。白名单最终通过 `checkpointer.with_allowlist(allowlist)` 注入到 checkpointer 的严格 msgpack 解码器中。

## 9.6 动手实验

### 实验一：用 get_type_hints() 解析 TypedDict

```python
from typing import Annotated, get_type_hints, get_origin, get_args
from typing_extensions import TypedDict
from operator import add

class MyState(TypedDict):
    messages: Annotated[list, add]
    count: int

hints = get_type_hints(MyState, include_extras=True)
for name, typ in hints.items():
    print(f"{name}: origin={get_origin(typ)}, args={get_args(typ)}")
# messages: origin=typing.Annotated, args=(list, <built-in function add>)
# count: origin=None, args=()
```

### 实验二：WeakKeyDictionary 缓存行为

```python
import weakref

cache = weakref.WeakKeyDictionary()

class A:
    x: int
    y: str

class B(A):
    z: float

# 存入缓存
from langgraph._internal._fields import get_cached_annotated_keys
print(get_cached_annotated_keys(B))  # ('x', 'y', 'z')

# 查看缓存
print(B in cache)  # True
print(cache[B])    # ('x', 'y', 'z')

# 删除对象后缓存自动清除
del B
import gc; gc.collect()
# cache 中 B 的条目已被自动移除
```

### 实验三：用 __or__ 创建 Runnable 管道

```python
from langgraph._internal._runnable import RunnableSeq, coerce_to_runnable

def step1(x):
    return x + 1

def step2(x):
    return x * 2

# coerce_to_runnable 会将普通函数包装为 RunnableCallable
r1 = coerce_to_runnable(step1, name="step1", trace=False)
r2 = coerce_to_runnable(step2, name="step2", trace=False)

# 用 | 操作符创建管道
pipeline = r1 | r2
print(type(pipeline))   # <class 'RunnableSeq'>
print(len(pipeline.steps))  # 2

# 执行管道
result = pipeline.invoke(5)  # (5 + 1) * 2 = 12
print(result)  # 12
```

### 实验四：观察 serde 白名单收集

```python
from typing import Optional
from pydantic import BaseModel
from langgraph._internal._serde import collect_allowlist_from_schemas

class Address(BaseModel):
    city: str
    zip_code: str

class User(BaseModel):
    name: str
    address: Address   # 嵌套模型

allowlist = collect_allowlist_from_schemas(schemas=[User])
for entry in allowlist:
    print(entry)
# 输出类似：
# ('__main__', 'Address')
# ('__main__', 'User')
```

### 实验五：merge_configs 的合并策略

```python
from langgraph._internal._config import merge_configs

config1 = {"tags": ["a", "b"], "metadata": {"env": "dev"}, "configurable": {"k1": 1}}
config2 = {"tags": ["c"], "metadata": {"user": "alice"}, "configurable": {"k2": 2}}

merged = merge_configs(config1, config2)
print(merged["tags"])        # ['a', 'b', 'c']  — 列表拼接
print(merged["metadata"])    # {'env': 'dev', 'user': 'alice'}  — 字典合并
print(merged["configurable"])  # {'k1': 1, 'k2': 2}  — 字典合并
```