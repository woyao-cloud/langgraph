# 第七章：检查点与恢复 — pregel/_checkpoint.py, _internal/_scratchpad.py

## 7.1 概述

LangGraph 的检查点（Checkpoint）机制是图执行持久化的核心。它使得图在任意步骤中断后可以从上一次的状态恢复继续执行，支撑了人机交互（human-in-the-loop）、容错重试等关键能力。本章走读 `pregel/_checkpoint.py` 中的检查点创建与恢复逻辑，以及 `_internal/_scratchpad.py` 中的 PregelScratchpad——每个任务级别的可变状态容器。两者共同构成了中断-恢复（interrupt/resume）流程的底层基础设施。

## 7.2 Java 桥梁

如果你来自 Java 世界，可以把 Checkpoint 理解为一次序列化的"游戏存档"——类似于 Java 中用 `Serializable` 将对象图写到 `ObjectOutputStream`，再从 `ObjectInputStream` 读回来。但 LangGraph 的 Checkpoint 并非将整个运行时对象图序列化，而是只保存 Channel 的值和版本元数据，恢复时通过 Channel 的 `from_checkpoint()` 方法重建 Channel 对象。

PregelScratchpad 则类似 Java 的 `ThreadLocal`：每个任务有自己的 scratchpad 实例，其中保存了中断计数器、恢复值等运行时状态。不同的是，Python 的 `ContextVar` 可以跨 `await` 边界传播，比 `ThreadLocal` 更适合异步场景。

## 7.3 Python 概念速查

| 概念 | Python 用法 | Java 对应 |
|------|-----------|----------|
| `copy.deepcopy` | `import copy; copy.deepcopy(obj)` 递归拷贝所有嵌套对象 | 对象实现 `Cloneable` + 手写 `clone()`，或用序列化反序列化 |
| 字典推导式 | `{k: v.copy() for k, v in d.items()}` | `Map<K,V> newMap = new HashMap<>(); d.forEach((k,v) -> newMap.put(k, new HashMap<>(v)));` |
| 元组作字典键 | `(task_id, channel_name)` 可直接做 dict 的 key | 需要自定义 `equals()` 和 `hashCode()` 的不可变类 |
| `ContextVar` | `contextvars.ContextVar("name")` 跨 `await` 传播的上下文变量 | `ThreadLocal<T>` 线程级变量 |
| `list.append` 返回 `None` | `result = my_list.append(x)` 得到 `None`，不是新列表 | `ArrayList.add()` 返回 `boolean` |
| 异常做控制流 | `raise GraphInterrupt()` 中断执行流 | Java 的受检异常需 `throws` 声明，非受检异常可模拟类似行为 |

**重点提醒：`list.append` 返回 `None`**

这是 Java 开发者常踩的坑。在 Java 中 `list.add(elem)` 返回 `true`，可以链式调用。但 Python 的 `list.append()` 返回 `None`，下面这种写法会导致难以察觉的 bug：

```python
# 错误：result 是 None，不是新列表
result = scratchpad.resume.append(v)  # result == None

# 正确：先 append，再使用列表
scratchpad.resume.append(v)
return scratchpad.resume  # 返回列表本身
```

## 7.4 代码走读

### 7.4.1 Checkpoint 数据结构

Checkpoint 是一个 `TypedDict`，定义在 `langgraph/checkpoint/base/__init__.py` 中：

```python
class Checkpoint(TypedDict):
    v: int                              # 格式版本号，当前为 4
    id: str                             # 检查点唯一 ID（单调递增，可排序）
    ts: str                             # ISO 8601 时间戳
    channel_values: dict[str, Any]      # 各 Channel 的当前值
    channel_versions: ChannelVersions    # 各 Channel 的版本号
    versions_seen: dict[str, ChannelVersions]  # 每个节点已看到的 Channel 版本
    updated_channels: list[str] | None   # 本次更新的 Channel 列表
```

其中 `ChannelVersions = dict[str, str | int | float]`，key 是 Channel 名称，value 是单调递增的版本号。

**`versions_seen` 的含义**：这是判断"哪些节点需要执行"的关键。每个节点记录自己上次执行时各 Channel 的版本。如果某个 Channel 的当前版本大于节点记录的版本，说明有新数据，该节点需要被触发。

Java 类比：可以想象一个事件驱动系统中，每个消费者记录自己消费到的 offset。`versions_seen` 就是消费者的 offset，`channel_versions` 就是分区的最新 offset。消费者只需处理 offset 之后的增量。

### 7.4.2 empty_checkpoint() — 创建初始检查点

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

`empty_checkpoint()` 创建一个"空白"检查点，所有字典均为空。`uuid6` 是一个时间排序的 UUID（UUID version 6），`clock_seq=-2` 确保初始 ID 足够小，保证后续 ID 单调递增。

Java 对比：Java 中创建初始状态通常用 `new State()` 调用无参构造器。Python 这里用工厂函数而非类构造器，返回一个 TypedDict——本质就是一个带类型提示的普通 `dict`。

### 7.4.3 create_checkpoint() — 快照当前 Channel

```python
def create_checkpoint(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel] | None,
    step: int,
    *,
    id: str | None = None,
    updated_channels: set[str] | None = None,
) -> Checkpoint:
    ts = datetime.now(timezone.utc).isoformat()
    if channels is None:
        values = checkpoint["channel_values"]
    else:
        values = {}
        for k in channels:
            if k not in checkpoint["channel_versions"]:
                continue
            v = channels[k].checkpoint()
            if v is not MISSING:
                values[k] = v
    return Checkpoint(
        v=LATEST_VERSION,
        ts=ts,
        id=id or str(uuid6(clock_seq=step)),
        channel_values=values,
        channel_versions=checkpoint["channel_versions"],
        versions_seen=checkpoint["versions_seen"],
        updated_channels=None if updated_channels is None else sorted(updated_channels),
    )
```

关键逻辑：

1. 如果传入 `channels`（运行中的 Channel 对象映射），则逐个调用 `channel.checkpoint()` 提取每个 Channel 的当前值。注意 `MISSING` 哨兵值的使用——某些 Channel 可能尚未初始化，跳过它们。
2. 如果 `channels` 为 `None`，则直接沿用已有 checkpoint 的 `channel_values`（用于恢复场景）。
3. 检查点 ID 由 `uuid6(clock_seq=step)` 生成，`step` 确保了 ID 的单调递增。

**Python 细节**：`*` 分隔符之后的参数强制使用关键字传参。Java 没有此特性，但可通过 Builder 模式模拟。`sorted(updated_channels)` 将集合转为排序后的列表，确保序列化结果的确定性——类似 Java 中用 `TreeSet` 保证顺序。

### 7.4.4 copy_checkpoint() — 深拷贝检查点

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

这里没有使用 `copy.deepcopy()`，而是手动逐层拷贝。原因有两点：

1. **性能**：`deepcopy` 会递归遍历所有嵌套对象，而这里已知数据结构只有两层嵌套（`versions_seen` 是 `dict[str, dict[str, ...]]`），手动拷贝更快。
2. **可控性**：`deepcopy` 可能遇到不可序列化的对象时报错，手动拷贝则只拷贝已知可拷贝的部分。

**字典推导式**：`{k: v.copy() for k, v in checkpoint["versions_seen"].items()}` 是 Python 的字典推导式。Java 开发者可以把它想象成 Stream API：

```java
// Java 等效伪代码
Map<String, Map<String, Object>> newVersionsSeen = versionsSeen.entrySet().stream()
    .collect(Collectors.toMap(
        Map.Entry::getKey,
        e -> new HashMap<>(e.getValue())
    ));
```

### 7.4.5 channels_from_checkpoint() — 从检查点恢复 Channel

```python
def channels_from_checkpoint(
    specs: Mapping[str, BaseChannel | ManagedValueSpec],
    checkpoint: Checkpoint,
) -> tuple[Mapping[str, BaseChannel], ManagedValueMapping]:
    channel_specs: dict[str, BaseChannel] = {}
    managed_specs: dict[str, ManagedValueSpec] = {}
    for k, v in specs.items():
        if isinstance(v, BaseChannel):
            channel_specs[k] = v
        else:
            managed_specs[k] = v
    return (
        {
            k: v.from_checkpoint(checkpoint["channel_values"].get(k, MISSING))
            for k, v in channel_specs.items()
        },
        managed_specs,
    )
```

恢复过程分两步：

1. 将规格说明（specs）分为普通 Channel 和托管值（ManagedValue）两类。
2. 对每个普通 Channel，调用其类方法 `from_checkpoint()` 重建 Channel 实例，传入检查点中保存的值。`MISSING` 哨兵值表示该 Channel 在检查点中无历史值，Channel 将以初始状态创建。

**元组作字典键**：`channel_values.get(k, MISSING)` 中的 `k` 是字符串。在 LangGraph 的其他部分，`PendingWrite` 使用 `(task_id, channel_name)` 元组作为键。Python 的元组天然可哈希、不可变，可直接用作 dict 的 key。Java 中若要实现相同效果，需要创建一个 `record PendingWriteKey(String taskId, String channelName)` 并依赖其自动生成的 `equals()` 和 `hashCode()`。

### 7.4.6 PregelScratchpad — 任务级可变状态

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

Scratchpad 是每个任务（task）独享的可变状态容器。它的字段分为三组：

- **call**：`call_counter` 追踪当前任务内部调用了多少个子任务（task），用于生成唯一的调用 ID。
- **interrupt**：`interrupt_counter` 追踪当前任务内发生了多少次 `interrupt()` 调用；`resume` 是恢复值的列表；`get_null_resume` 是一个闭包，用于获取"无明确 ID 的恢复值"。
- **subgraph**：`subgraph_counter` 追踪子图调用次数。

**`_DC_KWARGS = {"kw_only": True, "slots": True, "frozen": True}`**：`kw_only=True` 强制所有参数用关键字传入；`slots=True` 使用 `__slots__` 优化内存（类似 Java 的字段声明，避免 `__dict__` 开销）；`frozen=True` 使实例不可变——但注意，`resume: list[Any]` 是可变列表，frozen 只阻止对 `scratchpad.resume` 属性本身重新赋值，不阻止 `scratchpad.resume.append(v)`。

**Java 对比**：`frozen=True` 类似 Java 的 `final` 字段。但 Java 的 `final List<Any> resume` 同样只阻止引用变更，不阻止 `resume.add(v)`。两者的语义一致。

### 7.4.7 _scratchpad() 工厂函数

在 `pregel/_algo.py` 中，`_scratchpad()` 函数根据待处理的写入（pending_writes）和恢复映射（resume_map）构造 PregelScratchpad 实例：

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

核心逻辑：

1. 遍历 `pending_writes`，寻找 `NULL_TASK_ID` 对应的恢复写入（null_resume_write）和特定 task_id 的恢复写入。
2. `get_null_resume` 是一个闭包：如果当前 scratchpad 没有 null_resume_write，则向父 scratchpad 委托查找（子图场景）。
3. 计数器使用 `LazyAtomicCounter`——延迟初始化的线程安全计数器，内部用 `itertools.count(0).__next__` 实现。

**LazyAtomicCounter 的线程安全设计**：

```python
class LazyAtomicCounter:
    __slots__ = ("_counter",)

    def __call__(self) -> int:
        if self._counter is None:
            with LAZY_ATOMIC_COUNTER_LOCK:
                if self._counter is None:
                    self._counter = itertools.count(0).__next__
        return self._counter()
```

双重检查锁定（Double-Checked Locking）——Java 开发者对此模式应该非常熟悉。`itertools.count(0).__next__` 等效于 Java 的 `AtomicInteger.getAndIncrement()`。

## 7.5 运行原理：完整的中断-恢复流程

### 第一步：interrupt() 被调用

节点函数内调用 `interrupt(value)`（定义在 `types.py`）：

```python
def interrupt(value: Any) -> Any:
    conf = get_config()["configurable"]
    scratchpad = conf[CONFIG_KEY_SCRATCHPAD]
    idx = scratchpad.interrupt_counter()          # 获取当前中断索引
    if scratchpad.resume:                          # 有恢复值？
        if idx < len(scratchpad.resume):
            conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
            return scratchpad.resume[idx]          # 返回对应的恢复值
    v = scratchpad.get_null_resume(True)           # 尝试获取 null_resume
    if v is not None:
        scratchpad.resume.append(v)
        conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
        return v
    raise GraphInterrupt((                         # 无恢复值 → 抛异常
        Interrupt.from_ns(value=value, ns=conf[CONFIG_KEY_CHECKPOINT_NS]),
    ))
```

这段代码的精妙之处在于：**同一个函数，首次调用时抛出异常中断执行，恢复调用时返回恢复值**。这是通过 scratchpad 的 `resume` 列表和 `interrupt_counter` 协同实现的。

### 第二步：PregelLoop 捕获 GraphInterrupt

在 `_loop.py` 中，PregelLoop 的 `__exit__` 方法捕获 `GraphInterrupt`：

```python
suppress = isinstance(exc_value, GraphInterrupt) and not self.is_nested
if suppress:
    # 将中断前的写入应用到 Channel
    updated_channels = apply_writes(...)
    # 创建新的检查点
    self._put_checkpoint(self.checkpoint_metadata)
    self._put_pending_writes()
    # 发出中断事件
    self._emit("updates", lambda: iter([{INTERRUPT: exc_value.args[0]}]))
    return True  # 抑制异常，优雅退出
```

**Python 异常控制流 vs Java 受检异常**：在 Java 中，如果要用异常做控制流，需要声明 `throws GraphInterrupt` 或使用非受检异常。Python 没有受检异常的概念，任何异常都可以在任何地方抛出，框架代码负责捕获。LangGraph 利用这一特性将 `GraphInterrupt` 作为"正常的中断信号"而非"错误"。

### 第三步：客户端恢复执行

```python
# 客户端调用
graph.stream(Command(resume="用户输入"), config)
```

`Command(resume=...)` 携带恢复值，被写入 `pending_writes`。

### 第四步：PregelLoop 从检查点恢复

PregelLoop 启动时从检查点读取状态，调用 `channels_from_checkpoint()` 重建 Channel，调用 `_scratchpad()` 构造 scratchpad——此时 scratchpad 的 `resume` 列表中包含了恢复值。

### 第五步：节点重新执行

节点函数被重新执行（从头开始），再次调用 `interrupt(value)` 时：
- `interrupt_counter()` 返回 0（第一次 interrupt）
- `scratchpad.resume` 非空且长度大于 0
- `scratchpad.resume[0]` 就是用户提供的恢复值
- `interrupt()` 返回恢复值，不再抛异常

**重执行语义**：这是 Java 开发者最需要理解的点。中断后节点不是从断点继续，而是从节点开头重新执行。所有 `interrupt()` 调用都会按顺序返回之前存入 scratchpad 的恢复值。这意味着节点中 `interrupt()` 之前的代码会重新运行——但 LangGraph 通过检查点缓存避免了重复计算（如 `@task` 的结果会被缓存）。

## 7.6 动手实验

### 实验 1：观察检查点结构

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.pregel._checkpoint import empty_checkpoint, create_checkpoint
from langgraph.channels.last_value import LastValue

# 创建空检查点
cp = empty_checkpoint()
print("空检查点:", cp)
# 注意：channel_values, channel_versions, versions_seen 均为空字典

# 创建一个 Channel 并生成检查点
channels = {"messages": LastValue(str)}
channels["messages"].update("hello")
snapshot = create_checkpoint(cp, channels, step=1)
print("快照:", snapshot)
# channel_values = {"messages": "hello"}
# channel_versions 仍为空（因为 cp 中无版本记录）
```

### 实验 2：体验中断-恢复

```python
import uuid
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

class State(dict):
    pass

def human_review_node(state):
    # 第一次执行：抛出 GraphInterrupt
    # 恢复后：返回用户的审核结果
    review = interrupt("请审核此内容")
    return {"review": review}

builder = StateGraph(dict)
builder.add_node("review", human_review_node)
builder.add_edge(START, "review")
graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# 第一次调用——被中断
for chunk in graph.stream({"input": "test"}, config):
    print("中断事件:", chunk)

# 恢复执行
for chunk in graph.stream(Command(resume="通过"), config):
    print("恢复结果:", chunk)
```

### 实验 3：多次 interrupt

```python
def multi_interrupt_node(state):
    name = interrupt("请输入姓名")      # 第1次中断
    age = interrupt("请输入年龄")       # 第2次中断
    return {"name": name, "age": age}

# 需要分两次恢复，或者一次传入列表
# Command(resume={"id_1": "张三", "id_2": "25"})
# 或按顺序恢复
```

### 思考题

1. `copy_checkpoint()` 为什么不使用 `copy.deepcopy()`？在什么场景下使用 `deepcopy` 更安全？
2. 如果一个节点有 3 个 `interrupt()` 调用，但用户只提供了 2 个恢复值，第三次调用 `interrupt()` 时会发生什么？
3. `versions_seen` 如何保证在分布式检查点存储（如 Postgres）中正确地判断哪些节点需要重新执行？