# 第九章 内部基础设施 — _internal/ 模块

## Java 桥梁

Java 开发者习惯将工具类和常量放在 `Utils.java`、`Constants.java` 这样的辅助类中，用 `public static final` 定义常量，用 `final` 类封死继承。Python 没有这些语言级约束，但 LangGraph 的 `_internal/` 目录承担了相同的职责：它是整个框架的"引擎室"，存放所有不对外暴露的基础设施代码——常量、类型工具、配置合并、模式反射、序列化、Runnable 适配器。

以 `_` 前缀命名的模块是 Python 的"私有"约定。Java 用 `private` 关键字强制访问控制，Python 则靠命名约定和文档声明"你不该用，但要用也能用"。LangGraph 的 `_internal/` 就是这种哲学的典型体现。

## Python 概念速查

| Python 概念 | Java 对应 | 说明 |
|---|---|---|
| `sys.intern()` | `String.intern()` | 字符串驻留，相同内容共享同一对象 |
| `_` 前缀 | `private` / `package-private` | 命名约定，非强制访问控制 |
| `{**a, **b}` 字典合并 | `Map.putAll()` / `new HashMap<>(m1) {{putAll(m2);}}` | 字典解包合并，后者覆盖前者 |
| `functools.partial` | 方法引用 `this::method` / Lambda 偏应用 | 固定部分参数，生成新函数 |
| `object()` 单例哨兵 | `Optional.empty()` / 特殊哨兵对象 | 区分"未设置"和 `None` |
| `TypedDict` | Lombok `@Builder` + `@Data` | 结构化字典的类型提示 |
| `typing.get_type_hints()` | Java 反射 `Field.getGenericType()` | 运行时获取类型注解 |
| `get_origin()` / `get_args()` | `ParameterizedType.getRawType()` / `getActualTypeArguments()` | 解析泛型的原始类型和参数 |
| `@lru_cache` | `ConcurrentHashMap` + 缓存计算 | 函数级自动缓存 |
| `Annotated[type, metadata]` | Java 注解 `@SomeAnnotation Type` | 在类型上附加元数据 |
| `__or__` 运算符重载 | 无直接对应（类似 Stream 的 `pipe`） | 实现 `a | b` 管道语法 |
| `Protocol`（结构化子类型） | Java `interface` | 鸭子类型的类型检查 |

## 代码走读

### 9.1 _constants.py — 全局常量的中央仓库

这个文件定义了 Pregel 执行引擎用到的所有保留字符串常量。Java 开发者可以把它理解为整个框架的"常量接口"。

**保留写入键**——这些是节点输出字典中的保留 key，用户状态不能与它们冲突：

```python
INPUT = sys.intern("__input__")       # 图的输入值
INTERRUPT = sys.intern("__interrupt__")  # 节点动态中断
RESUME = sys.intern("__resume__")     # 中断后恢复传入的值
ERROR = sys.intern("__error__")       # 节点抛出的错误
NO_WRITES = sys.intern("__no_writes__")  # 节点未写入任何值的标记
TASKS = sys.intern("__pregel_tasks")  # Send 对象对应的推送任务
RETURN = sys.intern("__return__")     # 记录任务返回值
PREVIOUS = sys.intern("__previous__") # 处理 Control 值的隐式分支
```

**任务类型标记**——区分任务的触发方式：

```python
PUSH = sys.intern("__pregel_push")  # 推送型任务，由 Send 对象创建
PULL = sys.intern("__pregel_pull")  # 拉取型任务，由边触发
```

**配置注入键**——这些是 `config["configurable"]` 字典中的保留 key，用于注入运行时函数：

```python
CONFIG_KEY_SEND = sys.intern("__pregel_send")        # 写入函数
CONFIG_KEY_READ = sys.intern("__pregel_read")        # 读取函数
CONFIG_KEY_CALL = sys.intern("__pregel_call")        # 调用函数
CONFIG_KEY_CHECKPOINTER = sys.intern("__pregel_checkpointer")  # 检查点保存器
CONFIG_KEY_CHECKPOINT_ID = sys.intern("checkpoint_id")
CONFIG_KEY_CHECKPOINT_NS = sys.intern("checkpoint_ns")
CONFIG_KEY_THREAD_ID = sys.intern("thread_id")
```

**命名空间分隔符**——子图检查点嵌套用：

```python
NS_SEP = sys.intern("|")  # 层级分隔，如 "graph|subgraph|subsubgraph"
NS_END = sys.intern(":")  # 命名空间与任务 ID 分隔
```

所有常量用 `sys.intern()` 驻留。Java 中 `String.intern()` 将字符串放入 JVM 字符串池，Python 的 `sys.intern()` 同理——相同内容的字符串只保留一份引用，后续比较可用 `is`（指针比较）代替 `==`（内容比较），在频繁比较的场景下显著提升性能。注意 Python 3.x 已自动驻留短字符串和标识符，但 `sys.intern()` 对动态构造的字符串同样有效。

文件末尾的 `RESERVED` 集合汇总了所有保留 key，供运行时校验使用——确保用户的 State 键名不会与内部保留键冲突。

### 9.2 _config.py — 配置的合并与修补

LangGraph 的 `RunnableConfig` 是一个 TypedDict，结构为 `{tags, metadata, callbacks, recursion_limit, configurable: {...}}`。`_config.py` 提供了操作这个配置的核心工具。

**merge_configs()** 合并多个配置字典：

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
                base[key] = {**base_value, **value}   # 字典合并
            elif key == "tags":
                base[key] = [*base_value, *value]     # 列表拼接
            elif key == CONF:                         # "configurable"
                base[key] = {**base_value, **value}   # 字典合并
            elif key == "callbacks":
                # 6 种合并情况：None/list/manager 两两组合
                ...
            elif key == "recursion_limit":
                if config["recursion_limit"] != DEFAULT_RECURSION_LIMIT:
                    base["recursion_limit"] = config["recursion_limit"]
            else:
                base[key] = config[key]               # 直接覆盖
```

Java 的 `Map.putAll()` 是就地修改，而 Python 的 `{**a, **b}` 创建新字典——这体现了 Python 社区偏好的不可变模式。`metadata` 和 `configurable` 用字典合并（后者覆盖前者），`tags` 用列表拼接，`callbacks` 则有 6 种分支处理（None/list/CallbackManager 两两组合），`recursion_limit` 只在显式指定时才覆盖。

**patch_config()** 用新值修补配置：

```python
def patch_config(config, *, callbacks=None, recursion_limit=None,
                 max_concurrency=None, run_name=None, configurable=None):
    config = config.copy() if config is not None else {}  # 浅拷贝，不修改原对象
    if configurable is not None:
        config[CONF] = {**config.get(CONF, {}), **configurable}
    return config
```

这里 `config.copy()` 是浅拷贝，类似 Java 的 `new HashMap<>(original)`。关键字参数全用 `*` 后置强制命名调用，防止参数顺序混淆——这在 Java 中不需要，因为 Java 方法签名已经明确了参数名。

**ensure_config()** 确保返回完整配置：从上下文变量 `var_child_runnable_config` 获取当前配置，合并传入配置，填充默认值。它还会将 `configurable` 中的安全键值对提升到 `metadata` 中供追踪系统使用，但排除了包含 `key`/`token`/`secret`/`password`/`auth` 的键——这是一种防御性编程，防止敏感信息泄露到追踪元数据。

### 9.3 _typing.py — 哨兵值与类型协议

**MISSING 哨兵**：

```python
MISSING = object()
```

Java 用 `null` 表示"无值"，但 `null` 无法区分"值为空"和"值未设置"。Python 的 `None` 有同样的歧义。`MISSING = object()` 创建一个全局唯一的哨兵对象，用 `is MISSING` 判断——这比 Java 的 `Optional.empty()` 更轻量，也更 Pythonic。在 `_fields.py` 中 `get_update_as_tuples()` 用它检测属性是否存在于输出对象上。

**EMPTY_SEQ**：

```python
EMPTY_SEQ: tuple[str, ...] = tuple()
```

空元组单例。Python 中 `tuple()` 每次调用都返回同一个不可变对象，所以 `EMPTY_SEQ` 本质上是一个命名单例。类似 Java 中 `Collections.emptyList()` 返回的不可变空列表单例。

**DeprecatedKwargs**：

```python
class DeprecatedKwargs(TypedDict):
    pass
```

一个空的 TypedDict，用于标记已废弃的关键字参数。Java 没有对应概念——通常用 `@Deprecated` 注解标注。Python 的 TypedDict 为字典提供了结构化的类型提示，让 IDE 和类型检查器能识别出键名和类型。

**StateLike 类型别名**：

```python
StateLike: TypeAlias = TypedDictLikeV1 | TypedDictLikeV2 | DataclassLike | BaseModel
```

LangGraph 的 State 可以是 TypedDict、dataclass 或 Pydantic BaseModel。由于 Python 的 `TypedDict` 和 `dataclass` 不能直接用于 `isinstance()` 检查，这里用 `Protocol` 定义了结构化子类型（鸭子类型的类型安全版本）。Java 开发者可以把 Protocol 理解为"不需要 `implements` 声明的 interface"——只要类拥有所需的属性和方法，就自动满足协议。

### 9.4 _serde.py — 安全序列化与白名单

`_serde.py` 不直接做序列化，而是为检查点持久化构建"类型白名单"——只允许已知安全类型通过 msgpack 序列化。这是 Python 社区从 `pickle` 安全事故中学到的教训：pickle 能执行任意代码，相当于 Java 的原生序列化漏洞（CVE 频发）。

**curated_core_allowlist()** 返回 LangChain 核心消息类型的白名单：

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

白名单用 `(module, classname)` 元组表示——类似 Java 的全限定类名 `com.example.Foo`。`getattr(lc_messages, name, None)` 相当于 Java 的 `Class.forName()`，但更安全，因为查找范围限定在已知模块内。

**collect_allowlist_from_schemas()** 递归收集 State schema 中的所有类型：

```python
def _collect_from_type(typ, allowlist, seen, seen_ids):
    if _already_seen(typ, seen, seen_ids):
        return  # 防止循环引用导致无限递归
    origin = get_origin(typ)
    if origin is Union:
        for arg in get_args(typ):
            _collect_from_type(arg, allowlist, seen, seen_ids)
    if origin in (list, set, tuple, dict, deque, frozenset):
        for arg in get_args(typ):
            _collect_from_type(arg, allowlist, seen, seen_ids)  # 递归收集元素类型
    if is_typeddict(typ):
        for field_type in _safe_get_type_hints(typ).values():
            _collect_from_type(field_type, allowlist, seen, seen_ids)
    if _is_pydantic_model(typ):
        allowlist.add((typ.__module__, typ.__name__))
        # 继续递归收集字段类型...
```

这段代码用 `get_origin()` 和 `get_args()` 解析泛型类型，等价于 Java 的 `ParameterizedType.getRawType()` 和 `getActualTypeArguments()`。`_already_seen()` 用两个集合（`seen` 处理可哈希类型，`seen_ids` 处理不可哈希类型）防止循环引用——类似 Java 深度拷贝中的循环检测。

**apply_checkpointer_allowlist()** 将白名单应用到检查点保存器：

```python
def apply_checkpointer_allowlist(checkpointer, allowlist):
    if not checkpointer or allowlist is None:
        return checkpointer
    if not _SUPPORTS_ALLOWLIST:
        logger.warning("Checkpointer does not support with_allowlist...")
        return checkpointer
    return checkpointer.with_allowlist(allowlist)
```

这是"能力检测"模式——先检查对象是否支持某个方法，不支持则降级。Java 通常用接口声明能力，Python 则用 `hasattr()` 运行时检测。这里的 `_SUPPORTS_ALLOWLIST` 缓存了检测结果，避免每次调用都做 `hasattr()` 检查。

### 9.5 _fields.py — 模式反射与缓存

**get_cached_annotated_keys()** 从 State schema 中提取所有带注解的键名：

```python
ANNOTATED_KEYS_CACHE: weakref.WeakKeyDictionary[type[Any], tuple[str, ...]] = (
    weakref.WeakKeyDictionary()
)

def get_cached_annotated_keys(obj: type[Any]) -> tuple[str, ...]:
    if obj in ANNOTATED_KEYS_CACHE:
        return ANNOTATED_KEYS_CACHE[obj]
    keys: list[str] = []
    for base in reversed(obj.__mro__):         # 按方法解析顺序遍历
        ann = base.__dict__.get("__annotations__")
        if ann is None:
            ann = getattr(base, "__annotations__", None)
        if ann is None or isinstance(ann, types.GetSetDescriptorType):
            continue
        keys.extend(ann.keys())
    return ANNOTATED_KEYS_CACHE.setdefault(obj, tuple(keys))
```

`__mro__` 是 Python 的方法解析顺序（Method Resolution Order），类似 Java 的类继承链，但支持多重继承。`reversed()` 保证父类的注解在前，子类可以覆盖。`WeakKeyDictionary` 是弱引用字典——当 key（类型对象）不再被引用时自动清除缓存条目，避免内存泄漏。Java 没有内置弱引用键的 Map，但 `WeakHashMap` 用弱引用键实现了类似功能。

**get_update_as_tuples()** 将节点输出转换为 `(key, value)` 元组列表：

```python
def get_update_as_tuples(input: Any, keys: Sequence[str]) -> list[tuple[str, Any]]:
    if isinstance(input, BaseModel):
        keep = input.model_fields_set      # Pydantic 记录哪些字段被显式设置
        defaults = {k: v.default for k, v in type(input).model_fields.items()}
    else:
        keep = None
        defaults = {}
    return [
        (k, value)
        for k in keys
        if (value := getattr(input, k, MISSING)) is not MISSING
        and (value is not None or defaults.get(k, MISSING) is not None
             or (keep is not None and k in keep))
    ]
```

这里用到了海象运算符 `:=`（walrus operator）——在列表推导中同时赋值和判断，Java 没有对应语法。对 Pydantic 模型的特殊处理体现了"只更新与默认值不同的字段"的语义——类似 Java Builder 模式中 `@Builder.Default` 的行为。

### 9.6 _runnable.py — Runnable 适配器层

LangChain 的 `Runnable` 接口是整个框架的核心抽象，但 LangGraph 的节点函数五花八门——同步函数、异步函数、带 `writer` 参数的、带 `store` 参数的。`_runnable.py` 负责统一这些差异。

**RunnableCallable** 是 LangGraph 简化版的 `RunnableLambda`：

```python
class RunnableCallable(Runnable):
    def __init__(self, func, afunc=None, *, name=None, trace=True, recurse=True, ...):
        self.func = func
        self.afunc = afunc
        self.func_accepts: dict[str, tuple[str, Any]] = {}
        # 检查函数签名，识别可注入的参数
        params = inspect.signature(func or afunc).parameters
        for kw, typ, runtime_key, default in KWARGS_CONFIG_KEYS:
            p = params.get(kw)
            if p is None or p.kind not in VALID_KINDS:
                continue
            if typ != (ANY_TYPE,) and p.annotation not in typ:
                continue
            self.func_accepts[kw] = (runtime_key, default)
```

构造时通过 `inspect.signature()` 反射函数参数，与 `KWARGS_CONFIG_KEYS` 匹配，判断哪些运行时依赖需要注入——这类似 Java 的依赖注入框架（Spring 的 `@Autowired`），但完全基于参数名和类型注解的约定，不需要任何注解声明。

`invoke()` 方法中，从 `Runtime` 对象提取依赖值：

```python
runtime = config.get(CONF, {}).get(CONFIG_KEY_RUNTIME)
for kw, (runtime_key, default) in self.func_accepts.items():
    if kw in kwargs:
        continue  # 显式传入的参数优先
    kw_value = MISSING
    if kw == "config":
        kw_value = config
    elif runtime:
        kw_value = getattr(runtime, runtime_key)  # 从 Runtime 动态获取
    if kw_value is MISSING:
        kw_value = default  # 使用默认值
    kwargs[kw] = kw_value
```

**RunnableSeq** 实现管道组合——`a | b | c` 的语法糖：

```python
class RunnableSeq(Runnable):
    def __or__(self, other) -> Runnable:
        # a | b 返回新的 RunnableSeq
        return RunnableSeq(*self.steps, coerce_to_runnable(other, ...))
```

`__or__` 是 Python 的运算符重载，让 `|` 运算符变成管道操作符。Java 没有运算符重载（除了 `+` 对 String），等价写法是 `a.pipe(b).pipe(c)` 或 `Stream.of(a, b, c)`。RunnableSeq 在 invoke 时按序执行步骤，第一步在 context 中运行（确保上下文变量传播），后续步骤直接调用。

**coerce_to_runnable()** 将任意可调用对象转为 Runnable：

```python
def coerce_to_runnable(thing, *, name=None, trace=True) -> Runnable:
    if isinstance(thing, Runnable):
        return thing                    # 已经是 Runnable
    elif is_async_generator(thing) or inspect.isgeneratorfunction(thing):
        return RunnableLambda(thing)    # 生成器函数
    elif callable(thing):
        if is_async_callable(thing):
            return RunnableCallable(None, thing, ...)   # 异步函数
        else:
            return RunnableCallable(thing,
                wraps(thing)(partial(run_in_executor, None, thing)), ...)  # 同步函数
    elif isinstance(thing, dict):
        return RunnableParallel(thing)  # 字典 → 并行执行
```

`functools.partial(run_in_executor, None, thing)` 是关键技巧：将同步函数包装成异步版本，在线程池中执行——类似 Java 的 `CompletableFuture.supplyAsync()` + `Executor`。`wraps(thing)` 保留原函数的元数据（`__name__`、`__doc__`），方便调试和追踪。

## 运行原理

`_internal/` 模块的运行时关系可以概括为：

1. **Pregel 引擎启动时**：`ensure_config()` 构建初始配置，`merge_configs()` 合并来自调用者和上下文的配置，`patch_config()` 注入 send/read/call 等内部函数到 `configurable` 中。

2. **节点执行时**：`RunnableCallable.invoke()` 从配置中提取 Runtime 对象，根据 `func_accepts` 字典自动注入 `config`、`writer`、`store` 等参数到节点函数。

3. **状态更新时**：`get_cached_annotated_keys()` 确定 State 的键集合，`get_update_as_tuples()` 将节点返回值转换为 `(key, value)` 写入对，供 Channel 的 reducer 处理。

4. **检查点保存时**：`build_serde_allowlist()` 从 State schema 递归收集类型白名单，`apply_checkpointer_allowlist()` 将白名单传给检查点保存器，确保只序列化安全类型。

5. **管道执行时**：`RunnableSeq` 按 `|` 运算符组成的顺序依次执行步骤，第一步在 `contextvars` 上下文中运行（确保 `get_config()` 等上下文函数可用），后续步骤直接执行。

## 动手实验

1. **观察常量驻留效果**：在 Python REPL 中执行：
   ```python
   import sys
   a = "__input__"
   b = sys.intern("__input__")
   print(a is b)  # True（短字符串自动驻留）
   # 构造动态字符串
   c = "__" + "input" + "__"
   d = sys.intern("__" + "input" + "__")
   print(c is d)  # 不一定 True，但 intern 保证 True
   ```

2. **体验配置合并**：
   ```python
   from langgraph._internal._config import merge_configs
   c1 = {"tags": ["a"], "metadata": {"k1": 1}, "configurable": {"x": 10}}
   c2 = {"tags": ["b"], "metadata": {"k2": 2}, "configurable": {"x": 20}}
   merged = merge_configs(c1, c2)
   print(merged["tags"])          # ["a", "b"]
   print(merged["metadata"])      # {"k1": 1, "k2": 2}
   print(merged["configurable"])  # {"x": 20} — 后者覆盖
   ```

3. **理解 MISSING 哨兵**：
   ```python
   from langgraph._internal._typing import MISSING
   d = {"name": "Alice"}
   print(d.get("name", MISSING) is MISSING)  # False — 有值
   print(d.get("age", MISSING) is MISSING)   # True — 未设置
   # 与 None 的区别：None 可能是合法的空值
   d["age"] = None
   print(d.get("age", MISSING) is MISSING)   # False — 值为 None，但已设置
   ```

4. **探索模式缓存**：
   ```python
   from typing import TypedDict
   from langgraph._internal._fields import get_cached_annotated_keys
   class MyState(TypedDict):
       name: str
       count: int
   keys1 = get_cached_annotated_keys(MyState)
   keys2 = get_cached_annotated_keys(MyState)
   print(keys1 is keys2)  # True — 同一缓存对象
   ```

5. **管道运算符**：
   ```python
   from langgraph._internal._runnable import RunnableCallable, RunnableSeq
   def add_one(x): return x + 1
   def double(x): return x * 2
   r1 = RunnableCallable(add_one, name="add_one", trace=False)
   r2 = RunnableCallable(double, name="double", trace=False)
   pipeline = r1 | r2  # RunnableSeq
   print(pipeline.invoke(3))  # (3+1)*2 = 8
   ```