# 第七章 检查点与中断恢复

## Python 进阶

本章涉及多个高级 Python 特性,理解它们是阅读源码的前提:

### copy.deepcopy vs 手动浅拷贝

Python 的 `copy.deepcopy()` 会递归复制所有嵌套对象,代价高昂。LangGraph 的 `copy_checkpoint()` 选择手动拷贝每个字段,精确控制拷贝深度:

```python
# 手动浅拷贝 —— 只复制顶层容器,内部对象仍为引用
channel_versions=checkpoint["channel_versions"].copy()
# dict.copy() 返回新 dict,但值仍是引用
# 对于 int/str/float 这类不可变类型,浅拷贝已足够
```

这种「选择性深拷贝」策略在 `versions_seen` 上体现得更明显:

```python
versions_seen={k: v.copy() for k, v in checkpoint["versions_seen"].items()}
```

外层 dict 浅拷贝,内层每个 `ChannelVersions` dict 也浅拷贝 —— 恰好两层,因为值是不可变的版本号。这比 `deepcopy` 高效得多。

### dict comprehension 与浅拷贝的配合

```python
{k: v.copy() for k, v in checkpoint["versions_seen"].items()}
```

这里用 dict comprehension 实现了嵌套 dict 的逐层拷贝。Python 的 comprehension 不仅简洁,还能避免 `for` 循环中 `append` 返回 `None` 这类常见陷阱:

```python
# 错误示范: list.append 返回 None
result = [items.append(x) for x in data]  # [None, None, ...]

# 正确写法
result = [x for x in data]
# 或者确实需要 side-effect 时
for x in data:
    items.append(x)
```

### 元组作为 dict key —— 不可变性与可哈希性

`Checkpoint` 中的 `channel_versions` 使用 `dict[str, str | int | float]` 类型别名 `ChannelVersions`。版本号必须是不可变类型,因为它们被用作比较和排序的依据。如果用可变类型(如 `list`),就无法作为 dict key,也无法可靠比较大小。

### dataclass(slots=True, frozen=True) 与不可变配置

```python
_DC_KWARGS = {"kw_only": True, "slots": True, "frozen": True}

@dataclass(**_DC_KWARGS)
class PregelScratchpad:
    step: int
    stop: int
    ...
```

`slots=True` 用 `__slots__` 替代 `__dict__`,节省内存并阻止动态属性添加。`frozen=True` 使实例不可变,禁止赋值。`kw_only=True` 要求构造时使用关键字参数,提高可读性。

当需要「修改」一个 frozen dataclass 时,应使用 `dataclasses.replace()`:

```python
from dataclasses import replace
new_scratchpad = replace(old_scratchpad, step=old_scratchpad.step + 1)
```

这创建新实例而非修改原实例,符合不可变数据模式。

### 异常驱动的控制流

`GraphInterrupt` 继承自 `GraphBubbleUp`,后者继承自 `Exception`。但在 LangGraph 中,中断不是「错误」,而是流程控制手段:

```python
raise GraphInterrupt(
    (Interrupt.from_ns(value=value, ns=conf[CONFIG_KEY_CHECKPOINT_NS]),)
)
```

这种「用异常实现控制流」的模式在 Python 中虽不常见,但在框架代码中是有意为之 —— 中断需要穿透多层调用栈,直到被 PregelLoop 捕获。异常的栈展开(stack unwinding)机制天然适合这种跨层传播。

### contextvars.ContextVar 与 per-async-task 状态

`interrupt()` 函数通过 `get_config()` 获取当前运行时配置,其中包含 scratchpad 引用。在异步场景下,`contextvars.ContextVar` 确保每个 asyncio Task 拥有独立的配置副本,不会交叉污染:

```python
conf = get_config()["configurable"]
scratchpad = conf[CONFIG_KEY_SCRATCHPAD]
idx = scratchpad.interrupt_counter()  # 每个 task 独立计数
```

---

## 代码走读

### `_checkpoint.py` — 检查点的创建、恢复与复制

整个文件仅 89 行,却承载了检查点生命周期的核心逻辑。

#### `empty_checkpoint()` —— 初始版本分配

```python
LATEST_VERSION = 4

def empty_checkpoint() -> Checkpoint:
    return Checkpoint(
        v=LATEST_VERSION,
        id=str(uuid6(clock_seq=-2)),
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values={},
        channel_versions={},
        versions_seen={},
    )
```

逐行解析:

- `v=LATEST_VERSION`: 检查点格式版本号。当前为 4。此字段用于向前兼容 —— 加载旧版本检查点时可以根据版本号做迁移。
- `id=str(uuid6(clock_seq=-2))`: 使用 UUID6(时间排序的 UUID)作为检查点 ID。`clock_seq=-2` 确保即使在同一微秒内也能生成唯一 ID。UUID6 是 UUID1 的改进版,时间戳递增,适合做排序键。
- `ts=datetime.now(timezone.utc).isoformat()`: UTC 时间戳,ISO 8601 格式。注意这里用 `timezone.utc` 而非 `datetime.utcnow()`,因为后者在 Python 3.12 中已被弃用。
- `channel_values={}`: 空 dict,所有 channel 尚未有值。
- `channel_versions={}`: 空 dict,所有 channel 尚未有版本号。
- `versions_seen={}`: 空 dict,没有任何节点看过任何 channel 的版本。

**设计要点**: `empty_checkpoint()` 返回的不是一个 `None` 或哨兵值,而是一个结构完整的 TypedDict。这让下游代码无需判空,统一了处理逻辑。

#### `create_checkpoint()` —— 快照 channel 到检查点

```python
def create_checkpoint(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel] | None,
    step: int,
    *,
    id: str | None = None,
    updated_channels: set[str] | None = None,
) -> Checkpoint:
```

参数解读:
- `checkpoint`: 前一个检查点,提供 `channel_versions` 和 `versions_seen` 的基础。
- `channels`: 当前 channel 映射。为 `None` 时直接沿用旧值。
- `step`: 步骤编号,用作 UUID 的 `clock_seq`,确保单调递增。
- `id`: 可选的自定义检查点 ID。
- `updated_channels`: 本次更新的 channel 名称集合,会被排序后存储。

核心逻辑:

```python
if channels is None:
    values = checkpoint["channel_values"]
else:
    values = {}
    for k in channels:
        if k not in checkpoint["channel_versions"]:
            continue  # 跳过尚未注册的 channel
        v = channels[k].checkpoint()
        if v is not MISSING:
            values[k] = v
```

关键细节:
1. `if k not in checkpoint["channel_versions"]`: 只保存已经注册(即前序检查点中已有版本号)的 channel。新出现的 channel 不在此刻快照。
2. `channels[k].checkpoint()`: 每个 channel 类型自己决定如何序列化。`LastValue` 保存当前值,`Topic` 可能保存消息列表。
3. `if v is not MISSING`: `MISSING` 是一个哨兵值,表示 channel 值为空(不是 `None`,而是真的没有值)。跳过这些 channel 避免存储无意义数据。

返回值:
```python
return Checkpoint(
    v=LATEST_VERSION,
    ts=ts,
    id=id or str(uuid6(clock_seq=step)),
    channel_values=values,
    channel_versions=checkpoint["channel_versions"],  # 引用传递,非拷贝!
    versions_seen=checkpoint["versions_seen"],           # 同上
    updated_channels=None if updated_channels is None else sorted(updated_channels),
)
```

注意: `channel_versions` 和 `versions_seen` 是直接引用传递,不是拷贝。这是有意为之 —— 调用者会负责在必要时拷贝,避免无谓的复制开销。

`sorted(updated_channels)` 保证确定性顺序,这对序列化和哈希比较都很重要。

#### `channels_from_checkpoint()` —— 从检查点恢复 channel

```python
def channels_from_checkpoint(
    specs: Mapping[str, BaseChannel | ManagedValueSpec],
    checkpoint: Checkpoint,
) -> tuple[Mapping[str, BaseChannel], ManagedValueMapping]:
```

这个函数将 channel 规格(specs)分为两类:

```python
channel_specs: dict[str, BaseChannel] = {}
managed_specs: dict[str, ManagedValueSpec] = {}
for k, v in specs.items():
    if isinstance(v, BaseChannel):
        channel_specs[k] = v
    else:
        managed_specs[k] = v
```

然后用 dict comprehension 从检查点值恢复 channel:

```python
{
    k: v.from_checkpoint(checkpoint["channel_values"].get(k, MISSING))
    for k, v in channel_specs.items()
}
```

这里 `v.from_checkpoint()` 是类方法/静态方法,根据保存的值重新构建 channel 对象。`checkpoint["channel_values"].get(k, MISSING)` 提供默认值 —— 对尚无值的 channel 传入 `MISSING`。

**设计模式**: 这体现了「工厂方法」模式 —— 每个 `BaseChannel` 子类知道如何从序列化数据恢复自身。`LastValue.from_checkpoint(value)` 和 `Topic.from_checkpoint(value)` 的行为完全不同,但调用方不需要关心。

#### `copy_checkpoint()` —— 选择性深拷贝

```python
def copy_checkpoint(checkpoint: Checkpoint) -> Checkpoint:
    return Checkpoint(
        v=checkpoint["v"],
        ts=checkpoint["ts"],
        id=checkpoint["id"],
        channel_values=checkpoint["channel_values"].copy(),
        channel_versions=checkpoint["channel_versions"].copy(),
        versions_seen={k: v.copy() for k, v in checkpoint["versions_seen"].items()},
        updated_channels=checkpoint.get("updated_channels", None),
    )
```

逐字段分析:
- `v`, `ts`, `id`: 不可变类型(`int`, `str`),直接引用,无需拷贝。
- `channel_values`: `.copy()` 浅拷贝。值可能是任意可变对象(列表、字典),但这里只拷贝容器 —— 因为 `create_checkpoint` 已经创建了新 dict。
- `channel_versions`: `.copy()` 浅拷贝。值为不可变版本号,浅拷贝已足够。
- `versions_seen`: 两层 dict comprehension 实现逐层浅拷贝。外层 key 是 `str`(节点名),内层值是 `ChannelVersions`(版本号映射)。两层拷贝是因为两层都可能被修改。
- `updated_channels`: 使用 `.get()` 并提供默认值 `None`,因为旧版本检查点可能没有此字段。

**为什么不直接 `deepcopy`?** `channel_values` 中的值可能是大型对象(如整个对话历史),`deepcopy` 会递归复制一切。而实际只需要拷贝容器层级 —— 内部对象的引用语义通常是正确的。

### `_scratchpad.py` — PregelScratchpad 数据类

```python
@dataclasses.dataclass(**_DC_KWARGS)
class PregelScratchpad:
    step: int
    stop: int
    # call
    call_counter: Callable[[], int]
    # interrupt
    interrupt_counter: Callable[[], int]
    get_null_resume: Callable[[bool], Any]
    resume: list[Any]
    # subgraph
    subgraph_counter: Callable[[], int]
```

注意 `_DC_KWARGS = {"kw_only": True, "slots": True, "frozen": True}`,所以:

1. **`frozen=True`**: 所有字段不可变。但 `resume: list[Any]` 是可变对象 —— frozen dataclass 禁止 `scratchpad.resume = new_list`,但不禁止 `scratchpad.resume.append(value)`。这是 Python 的一个微妙之处:frozen dataclass 防止属性重新绑定,但不阻止可变容器的内容变更。

2. **`slots=True`**: 使用 `__slots__`,内存效率更高。每个实例不会有 `__dict__`。

3. **`kw_only=True`**: 构造时必须使用关键字参数 `PregelScratchpad(step=1, stop=10, ...)`,避免位置参数混淆。

4. **`Callable[[], int]` 类型**: `call_counter`、`interrupt_counter`、`subgraph_counter` 的类型是零参数可调用对象,返回 `int`。这实际上是 `LazyAtomicCounter` 实例 —— 它既是可调用对象(`__call__`),又是线程安全的原子计数器。

### `LazyAtomicCounter` —— 延迟初始化的线程安全计数器

```python
LAZY_ATOMIC_COUNTER_LOCK = threading.Lock()

class LazyAtomicCounter:
    __slots__ = ("_counter",)
    _counter: Callable[[], int] | None

    def __init__(self) -> None:
        self._counter = None

    def __call__(self) -> int:
        if self._counter is None:
            with LAZY_ATOMIC_COUNTER_LOCK:
                if self._counter is None:
                    self._counter = itertools.count(0).__next__
        return self._counter()
```

这是一个精巧的并发设计:

1. **延迟初始化**: `_counter` 初始为 `None`,首次调用时才创建。避免在构造时分配不需要的资源。

2. **双重检查锁定**: `if self._counter is None` 先无锁检查,只有在确实需要初始化时才加锁。锁内再检查一次(`if self._counter is None`),防止多线程同时通过第一次检查。

3. **`itertools.count(0).__next__`**: 这不是 `count(0)` 对象本身,而是其 `__next__` 方法。`itertools.count()` 返回的迭代器在 CPython 中是 C 实现的,其 `__next__` 操作是线程安全的(GIL 保证原子性)。比 `threading.Lock` + 手动计数更高效。

4. **`__slots__`**: 只有一个属性 `_counter`,使用 `__slots__` 避免实例 `__dict__` 开销。

### `interrupt()` —— 中断函数全貌

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
    raise GraphInterrupt(
        (Interrupt.from_ns(value=value, ns=conf[CONFIG_KEY_CHECKPOINT_NS]),)
    )
```

逐行解析:

1. `idx = scratchpad.interrupt_counter()`: 每次调用 `interrupt()` 时计数器递增。第一个 `interrupt()` 得到 `idx=0`,第二个得到 `idx=1`,以此类推。

2. **第一次执行(中断模式)**:
   - `scratchpad.resume` 为空列表 `[]`
   - `scratchpad.get_null_resume(True)` 返回 `None`(无全局恢复值)
   - 进入 `raise GraphInterrupt(...)`,节点执行中止

3. **恢复执行时(有 resume 值)**:
   - `scratchpad.resume` 包含用户提供的恢复值列表
   - `idx < len(scratchpad.resume)` 为 `True`
   - `conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])`: 将恢复值写入 channel,使下游节点可以看到
   - `return scratchpad.resume[idx]`: 返回对应位置的恢复值

4. **null resume 的处理**: `Command(resume=None)` 时,`get_null_resume(True)` 返回 `None` 值本身。这里的 `consume=True` 参数表示「消费性读取」—— 一旦读取就从 pending_writes 中移除,防止重复消费。

5. **assert 的作用**: `assert len(scratchpad.resume) == idx` 确保在获取 null resume 之前,没有已经通过其他方式获取的 resume 值。这是运行时一致性检查。

### `_scratchpad()` 工厂函数

```python
def _scratchpad(
    parent_scratchpad: PregelScratchpad | None,
    pending_writes: list[PendingWrite],
    task_id: str,
    namespace_hash: str,
    resume_map: dict[str, Any] | None,
    step: int,
    stop: int,
) -> PregelScratchpad:
```

这个工厂函数创建闭包来捕获配置上下文:

```python
def get_null_resume(consume: bool = False) -> Any:
    if null_resume_write is None:
        if parent_scratchpad is not None:
            return parent_scratchpad.get_null_resume(consume)
        return None
    if consume:
        try:
            pending_writes.remove(null_resume_write)
            return null_resume_write[2]
        except ValueError:
            return None
    return null_resume_write[2]
```

关键设计:
- `get_null_resume` 是闭包,捕获了 `null_resume_write` 和 `parent_scratchpad`。
- 如果当前任务没有 null resume,向上查找父 scratchpad(用于子图场景)。
- `consume=True` 时,`pending_writes.remove()` 从列表中移除该写记录,确保 null resume 只被消费一次。
- `try/except ValueError` 处理竞态条件:如果另一个任务已经消费了这个 write,`remove` 会抛出 `ValueError`。

---

## 运行原理

### 中断/恢复的完整生命周期

```
                    第一次执行
                    ┌─────────────────────────────────────────────┐
                    │                                             │
  用户调用           │  Node 函数                                  │
  graph.stream()    │  ┌──────────────────────────────────────┐  │
  ──────────────►   │  │ answer = interrupt("what is X?")      │  │
                    │  │     ↓                                  │  │
                    │  │ idx = scratchpad.interrupt_counter() #0│  │
                    │  │ scratchpad.resume is []               │  │
                    │  │ get_null_resume(True) → None          │  │
                    │  │     ↓                                  │  │
                    │  │ raise GraphInterrupt(Interrupt(...))   │  │
                    │  └──────────────────────────────────────┘  │
                    │                     ↓                       │
                    │  PregelLoop 捕获异常                         │
                    │  调用 create_checkpoint() 保存状态           │
                    │  StateSnapshot 包含 interrupt 信息          │
                    └─────────────────────────────────────────────┘
                                      ↓
                    ┌─────────────────────────────────────────────┐
                    │  返回给客户端:                                 │
                    │  {'__interrupt__': [Interrupt(value='what  │
                    │   is X?', id='...')]}                       │
                    └─────────────────────────────────────────────┘
                                      ↓
                    第二次执行(恢复)
                    ┌─────────────────────────────────────────────┐
                    │  用户调用                                     │
                    │  graph.stream(Command(resume="42"))          │
                    │        ↓                                      │
                    │  PregelLoop 从检查点恢复                       │
                    │  _scratchpad() 构造 resume=["42"]             │
                    │        ↓                                      │
                    │  Node 函数重新执行                               │
                    │  ┌──────────────────────────────────────┐  │
                    │  │ answer = interrupt("what is X?")       │  │
                    │  │     ↓                                  │  │
                    │  │ idx = scratchpad.interrupt_counter()#0│  │
                    │  │ scratchpad.resume = ["42"]            │  │
                    │  │ idx(0) < len(resume)(1) → True        │  │
                    │  │ return resume[0] → "42"               │  │
                    │  │                                        │  │
                    │  │ answer = "42" ← 函数继续执行            │  │
                    │  └──────────────────────────────────────┘  │
                    └─────────────────────────────────────────────┘
```

### ChannelVersions 与 versions_seen 的协作

```
  检查点结构:
  {
    "channel_versions": {
      "messages": "v3",    ← messages channel 在步骤 3 更新
      "context":   "v2",   ← context channel 在步骤 2 更新
    },
    "versions_seen": {
      "chatbot": {         ← chatbot 节点
        "messages": "v2",  ← 上次看到 messages 时版本是 v2
        "context":   "v1",
      },
      "tool": {            ← tool 节点
        "messages": "v3",  ← tool 看到 messages 时版本是 v3
      }
    }
  }

  恢复逻辑:
  - chatbot 节点检查: messages 当前版本 v3 > 已看版本 v2 → 需要重新执行
  - tool 节点检查: messages 当前版本 v3 == 已看版本 v3 → 不需要重新执行
```

这种设计使得恢复时只重放有新数据的节点,避免不必要的重复计算。

---

## 实现细节

### xxhash 与 channel 版本管理

LangGraph 使用 `xxh3_128_hexdigest` 生成 channel 版本号:

```python
from xxhash import xxh3_128_hexdigest

def _xxhash_str(namespace: bytes, *parts: str | bytes) -> str:
    hex = xxh3_128_hexdigest(
        namespace + b"".join(p.encode() if isinstance(p, str) else p for p in parts)
    )
    return f"{hex[:8]}-{hex[8:12]}-{hex[12:16]}-{hex[16:20]}-{hex[20:32]}"
```

选择 xxhash 而非 SHA/MD5 的原因:
- **速度**: xxHash3 比 MD5 快约 5 倍,比 SHA-256 快约 12 倍
- **128-bit**: 与 UUID 长度一致,碰撞概率极低
- **非加密场景**: 版本号不需要密码学安全性

但在简单递增场景下,也保留了整数版本:

```python
def increment(current: int | None, channel: None) -> int:
    return current + 1 if current is not None else 1
```

检查点的 `null_version` 由 `checkpoint_null_version` 函数决定 —— 它检查所有 `channel_versions` 值的类型来决定是使用 `""` (字符串版本)还是 `0`(整数版本)。

### 检查点 null 版本的初始化

```python
def checkpoint_null_version(checkpoint: Checkpoint) -> V | None:
    for version in checkpoint["channel_versions"].values():
        return type(version)()
    return None
```

这个函数极为简洁但含义深刻:它从 `channel_versions` 中取第一个版本值,然后调用其类型的无参构造函数。如果版本类型是 `str`,则 `type("v1")()` → `str()` → `""`;如果是 `int`,则 `type(1)()` → `0`。如果没有任何版本,返回 `None`,表示检查点完全空白。

### 选择性 channel 拷贝

`create_checkpoint` 中只保存已注册的 channel:

```python
for k in channels:
    if k not in checkpoint["channel_versions"]:
        continue  # 新 channel,暂不保存
    v = channels[k].checkpoint()
    if v is not MISSING:
        values[k] = v
```

这种「选择性保存」有两个好处:
1. 避免保存未初始化的 channel(值为 `MISSING`)
2. 只保存前序检查点中已知存在的 channel,保证恢复时一致性

### PregelScratchpad 的 interrupt_counter 与多中断匹配

当节点中有多个 `interrupt()` 调用时:

```python
def my_node(state):
    x = interrupt("question 1")  # idx=0
    y = interrupt("question 2")  # idx=1
    return {"x": x, "y": y}
```

恢复时:

```python
# 第一次中断
scratchpad.resume = []  # 空
interrupt_counter() → 0, resume[0] 不存在 → raise GraphInterrupt

# 恢复后
scratchpad.resume = ["answer1"]
idx=0, resume[0] = "answer1" → return "answer1"
idx=1, resume 不够长 → raise GraphInterrupt

# 再次恢复
scratchpad.resume = ["answer1", "answer2"]
idx=0 → return "answer1"
idx=1 → return "answer2"
```

`interrupt_counter()` 用 `itertools.count(0).__next__` 保证每次调用索引递增,即使在同一次执行中多次调用也能正确匹配。

### null_resume 与 Command(resume=None)

`Command(resume=None)` 是一个特殊场景 —— 用户想恢复中断但不提供值。`get_null_resume` 闭包处理这种情况:

1. `pending_writes` 中查找 `(NULL_TASK_ID, RESUME, None)` 记录
2. 找到后,`consume=True` 时通过 `pending_writes.remove()` 消费
3. `try/except ValueError` 处理并发移除

---

## 动手实验

### 实验 1: 简单中断/恢复流

```python
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    value: str

def ask_human(state: State) -> dict:
    answer = interrupt("Please provide input:")
    return {"value": answer}

builder = StateGraph(State)
builder.add_node("ask_human", ask_human)
builder.add_edge(START, "ask_human")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 第一次执行 —— 触发中断
config = {"configurable": {"thread_id": "1"}}
result = graph.invoke({"value": ""}, config)
# 结果: 中断触发, interrupt value="Please provide input:"

# 恢复执行
result = graph.invoke(Command(resume="42"), config)
print(result)  # {'value': '42'}
```

### 实验 2: 手动检查检查点结构

```python
# 获取最新检查点
state = graph.get_state(config)
print(state.values)        # 当前 channel 值
print(state.next)          # 下一步要执行的节点
print(state.tasks)         # 待执行任务(含 interrupt 信息)

# 查看检查点内部结构
checkpoint = state.checkpoint
print(checkpoint["channel_versions"])  # 版本号映射
print(checkpoint["versions_seen"])     # 节点已看版本映射
print(checkpoint["id"])                 # 检查点 ID(UUID6 格式)
```

### 实验 3: 多中断顺序匹配

```python
def multi_interrupt(state: State) -> dict:
    x = interrupt("first question")   # idx=0
    y = interrupt("second question")   # idx=1
    return {"value": f"{x}+{y}"}

# 第一次调用: 在第一个 interrupt 处中断
# 恢复时: 第一个 interrupt 返回 resume[0]
# 第二个 interrupt 再次中断
# 再次恢复: 第一个返回 resume[0], 第二个返回 resume[1]
```

这验证了 `interrupt_counter()` 的顺序匹配机制 —— 每个中断点按调用顺序与恢复值一一对应。

### 实验 4: LazyAtomicCounter 线程安全性

```python
import itertools
import threading

class LazyAtomicCounter:
    __slots__ = ("_counter",)
    _counter: Callable[[], int] | None

    def __init__(self):
        self._counter = None

    def __call__(self) -> int:
        if self._counter is None:
            self._counter = itertools.count(0).__next__
        return self._counter()

# 测试: 多线程下计数器是否递增
counter = LazyAtomicCounter()
results = []

def worker():
    for _ in range(100):
        results.append(counter())

threads = [threading.Thread(target=worker) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# 验证: 所有计数唯一
assert len(set(results)) == len(results), "存在重复计数!"
print(f"生成 {len(results)} 个唯一计数,范围 [0, {max(results)}]")
```