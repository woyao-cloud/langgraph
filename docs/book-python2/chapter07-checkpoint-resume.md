# 第七章 检查点与中断恢复

## 1. 功能概览

LangGraph 的检查点（Checkpoint）与中断恢复（Interrupt/Resume）机制，是框架实现持久化执行和人机协作的核心基础设施。从用户视角看，它提供三大能力：

- **状态持久化**：图的每一步执行状态都被自动保存，即使进程崩溃也能从最近的检查点恢复
- **人机协作**：通过 `interrupt()` 函数暂停执行，等待人工输入后用 `Command(resume=...)` 恢复，实现审批、纠错等交互流程
- **状态审查与修改**：通过 `get_state()` 查看任意时刻的快照，通过 `update_state()` 手动覆盖状态

这些能力的底层依赖三个模块协作：`pregel/_checkpoint.py` 负责检查点的创建与恢复，`_internal/_scratchpad.py` 负责追踪中断位置与恢复值，`types.py` 中的 `interrupt()` 函数是用户侧的入口。

## 2. 应用场景

### 场景 A：人工审批工作流

最常见的场景——在关键决策点暂停，等待人类审批后继续：

```python
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    document: str
    approved: Optional[bool]
    feedback: Optional[str]

def review_node(state: State):
    # 中断执行，将文档发送给审批人
    result = interrupt({"question": "是否批准此文档？", "document": state["document"]})
    # 恢复后 result 就是人类提供的审批结果
    return {"approved": result["approved"], "feedback": result.get("feedback", "")}

def publish_node(state: State):
    if state["approved"]:
        return {"document": f"[已发布] {state['document']}"}
    return {"document": f"[已驳回] {state['document']}"}

builder = StateGraph(State)
builder.add_node("review", review_node)
builder.add_node("publish", publish_node)
builder.add_edge(START, "review")
builder.add_conditional_edges("review", "publish",
    lambda s: "publish" if s["approved"] else END)
builder.add_edge("publish", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["review"])

config = {"configurable": {"thread_id": "doc-001"}}
# 第一次运行——在审批节点前中断
graph.invoke({"document": "季度报告_v3.pdf"}, config)

# 人类审批后恢复
graph.invoke(Command(resume={"approved": True, "feedback": "通过"}), config)
```

### 场景 B：多轮对话的检查点保持

利用 `thread_id` 隔离不同对话，检查点自动保存对话历史：

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 用户 A 的对话
config_a = {"configurable": {"thread_id": "user-A"}}
graph.invoke({"messages": [("user", "你好")]}, config_a)
graph.invoke({"messages": [("user", "继续聊")]}, config_a)  # 自动延续上下文

# 用户 B 的对话——互不干扰
config_b = {"configurable": {"thread_id": "user-B"}}
graph.invoke({"messages": [("user", "新话题")]}, config_b)
```

### 场景 C：长时运行管道的故障恢复

对于可能运行数小时的 ETL 流程，检查点确保崩溃后不必从头开始：

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 使用 SQLite 持久化检查点到磁盘
with SqliteSaver.from_conn_string("etl_checkpoint.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "etl-job-001"}}
    result = graph.invoke(initial_data, config)
    # 即使中途崩溃，下次用相同 thread_id 调用即可从断点恢复
```

### 场景 D：使用 get_state 调试检查点

```python
# 查看当前检查点的完整状态
state_snapshot = graph.get_state(config)
print(state_snapshot.values)       # 当前所有 channel 的值
print(state_snapshot.next)          # 下一步要执行的节点
print(state_snapshot.config)        # 检查点配置（含 checkpoint_id）
print(state_snapshot.metadata)      # 元数据（source, step 等）

# 查看历史检查点
for state in graph.get_state_history(config):
    print(f"Step {state.metadata['step']}: {state.values}")
```

### 场景 E：使用 update_state 手动修改状态

```python
# 手动覆盖某个 channel 的值
graph.update_state(config, {"approved": True}, "人为批准")

# 也可以从某个历史检查点分叉
past_state = list(graph.get_state_history(config))[-1]
graph.update_state(past_state.config, {"document": "修正后的文档"}, "修正文档")
```

### 场景 F：单节点内的多次中断

```python
def multi_step_review(state: State):
    # 第一个中断——确认基本信息
    basic_info = interrupt({"stage": "basic_check", "data": state["raw_data"]})
    # 第二个中断——高级审核
    advanced_review = interrupt({"stage": "advanced_review", "data": basic_info})
    # 恢复值按顺序匹配
    return {"result": advanced_review}

# 恢复时依次提供每个中断的值
graph.invoke(Command(resume="基本确认通过"), config)  # 匹配第一个 interrupt
graph.invoke(Command(resume="高级审核通过"), config)  # 匹配第二个 interrupt
```

### 场景 G：检查点存储选择

| 存储实现 | 适用场景 | 持久性 | 性能 |
|---------|---------|--------|------|
| `InMemorySaver` | 开发测试、原型验证 | 进程退出即丢失 | 最快 |
| `SqliteSaver` | 单机部署、轻量生产 | 文件持久化 | 中等 |
| `PostgresSaver` | 分布式生产环境 | 数据库持久化 | 需连接池 |

### 场景 H：基于 thread_id 的状态隔离

每个 `thread_id` 对应独立的检查点链，不同线程的执行互不干扰：

```python
# 同一个 graph，不同 thread_id 各自维护独立状态
config_1 = {"configurable": {"thread_id": "order-001"}}
config_2 = {"configurable": {"thread_id": "order-002"}}

graph.invoke({"order": "A"}, config_1)
graph.invoke({"order": "B"}, config_2)  # 不会影响 order-001
```

## 3. Python 进阶

### copy.deepcopy 与手动拷贝

`copy_checkpoint` 函数没有使用 `copy.deepcopy`，而是逐字段手动拷贝：

```python
def copy_checkpoint(checkpoint: Checkpoint) -> Checkpoint:
    return Checkpoint(
        v=checkpoint["v"],                          # int，不可变
        ts=checkpoint["ts"],                          # str，不可变
        id=checkpoint["id"],                          # str，不可变
        channel_values=checkpoint["channel_values"].copy(),       # 浅拷贝
        channel_versions=checkpoint["channel_versions"].copy(),   # 浅拷贝
        versions_seen={k: v.copy() for k, v in checkpoint["versions_seen"].items()},
        updated_channels=checkpoint.get("updated_channels", None),
    )
```

选择手动拷贝而非 `deepcopy` 的原因：`deepcopy` 会递归遍历所有嵌套对象，对于已知结构的 TypedDict 来说开销大且可能触发意料之外的深拷贝（如拷贝了本应共享的 channel 对象）。手动拷贝更安全、更快、行为更可预测。

### dict 推导式与 tuple 作为字典键

`versions_seen` 的拷贝使用了字典推导式 `{k: v.copy() for k, v in ...}`，这是 Python 中拷贝嵌套字典的惯用手法。

`ChannelVersions` 的类型是 `dict[str, str | int | float]`，其键是字符串，但在更底层的 `versions_seen` 中，结构是 `dict[str, ChannelVersions]`——以节点 ID 为键、每个节点已看到的 channel 版本映射为值。这种嵌套字典结构使得判断"哪些节点需要重新执行"变得高效。

### ContextVar 与并发隔离

`interrupt()` 函数通过 `get_config()` 获取当前执行上下文，而非全局变量。LangGraph 底层使用 `contextvars.ContextVar` 将 scratchpad 注入到配置中：

```python
conf = get_config()["configurable"]
scratchpad = conf[CONFIG_KEY_SCRATCHPAD]
idx = scratchpad.interrupt_counter()
```

这保证了在并发场景下，不同协程各自拥有独立的 scratchpad 状态，互不干扰。

### list.append 返回 None

在 `interrupt()` 的恢复逻辑中：

```python
scratchpad.resume.append(v)
conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
return v
```

`list.append()` 返回 `None`，所以必须分两步——先 append，再用 `return` 返回值。初学者常犯的错误是写成 `return scratchpad.resume.append(v)`，这会返回 `None`。

### 基于异常的控制流

`GraphInterrupt` 继承自 `GraphBubbleUp`（而非 `Exception`），这是一种用异常实现控制流的设计：

```python
raise GraphInterrupt((
    Interrupt.from_ns(value=value, ns=conf[CONFIG_KEY_CHECKPOINT_NS]),
))
```

当节点调用 `interrupt()` 时，第一次执行抛出异常中断整个执行循环；恢复时节点重新执行，此时 scratchpad 中已存有恢复值，`interrupt()` 正常返回而不抛异常。这种"抛异常-重执行"的模式在同步和异步代码中都能工作。

### dataclass(slots=True, frozen=True)

`_DC_KWARGS = {"kw_only": True, "slots": True, "frozen": True}` 被用于多个数据类定义：

- `slots=True`：用 `__slots__` 替代 `__dict__`，节省内存并禁止动态属性
- `frozen=True`：实例不可变，创建后无法修改属性，保证数据安全
- `kw_only=True`：构造函数只接受关键字参数，防止位置参数混乱

`Interrupt` 类虽然用了 `slots=True`，但没有 `frozen`——因为中断值需要动态构建。

### dataclasses.replace()

检查点库中的 `copy_checkpoint` 函数在 `langgraph/checkpoint/base/__init__.py` 中使用了 TypedDict 构造而非 `dataclasses.replace()`，因为 `Checkpoint` 是 `TypedDict` 不是 dataclass。但在 `PregelScratchpad` 等 dataclass 场景中，`frozen=True` 意味着你不能用赋值修改字段，必须通过 `dataclasses.replace()` 创建新实例。

## 4. 代码走读

### _checkpoint.py：检查点创建与恢复

**empty_checkpoint()** 创建一个空白检查点作为图的初始状态：

```python
LATEST_VERSION = 4

def empty_checkpoint() -> Checkpoint:
    return Checkpoint(
        v=LATEST_VERSION,                   # 版本号，当前为 4
        id=str(uuid6(clock_seq=-2)),          # UUID v6，单调递增
        ts=datetime.now(timezone.utc).isoformat(),  # ISO 8601 时间戳
        channel_values={},                    # 空 channel 值
        channel_versions={},                  # 空 channel 版本
        versions_seen={},                     # 空"已见版本"映射
    )
```

`uuid6(clock_seq=-2)` 保证了 ID 的单调递增——这对排序检查点至关重要。`clock_seq=-2` 是一个特殊标记，确保初始检查点 ID 在排序时位于所有后续检查点之前。

**create_checkpoint()** 从当前 channel 状态创建新检查点：

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

关键逻辑：当 `channels` 不为 None 时，遍历每个 channel 调用 `v.checkpoint()` 获取序列化值，但跳过不在 `checkpoint["channel_versions"]` 中的 channel（即尚未写入的 channel）。`MISSING` 哨兵值用于区分"未设置"和"设置为 None"。

**channels_from_checkpoint()** 执行恢复——从检查点重建 channel 对象：

```python
def channels_from_checkpoint(
    specs: Mapping[str, BaseChannel | ManagedValueSpec],
    checkpoint: Checkpoint,
) -> tuple[Mapping[str, BaseChannel], ManagedValueMapping]:
```

它将 specs 分为普通 channel 和 managed value 两类，对每个 channel 调用 `from_checkpoint()` 从保存的值重建状态。这种分离设计让 managed value（如配置注入器）不参与检查点的序列化。

**copy_checkpoint()** 实现浅拷贝，已在前面分析。

### _scratchpad.py：中断位置追踪器

`PregelScratchpad` 是一个冻结的 dataclass，记录当前执行步骤的中断信息：

```python
@dataclasses.dataclass(**_DC_KWARGS)
class PregelScratchpad:
    step: int                          # 当前执行步骤号
    stop: int                          # 最大步骤数
    call_counter: Callable[[], int]    # call() 调用计数器
    interrupt_counter: Callable[[], int]  # interrupt() 调用计数器
    get_null_resume: Callable[[bool], Any]  # 获取全局恢复值
    resume: list[Any]                  # 任务级别的恢复值列表
    subgraph_counter: Callable[[], int]    # 子图计数器
```

每个字段的作用：
- `step/stop`：控制执行步数上限
- `call_counter`：为 `@task` 调用分配唯一 ID
- `interrupt_counter`：追踪 `interrupt()` 的调用序号，确保同一节点内多次中断的值能按顺序匹配
- `get_null_resume`：从全局恢复写入中提取值，`consume=True` 时消费（删除）该写入
- `resume`：当前任务的中断恢复值列表

**LazyAtomicCounter** 是一个精巧的线程安全计数器：

```python
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

它使用了双重检查锁定（Double-Checked Locking）模式：先无锁检查 `_counter` 是否为 None，仅在首次调用时加锁创建。底层使用 `itertools.count(0).__next__` 作为原子递增——因为 `count` 本身是线程安全的（GIL 保证 `__next__` 的原子性）。`__slots__` 进一步节省内存。

**_scratchpad() 工厂函数**（位于 `_algo.py`）负责组装 scratchpad：

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

它从三个来源收集恢复值：
1. **全局恢复**（`null_resume_write`）：`NULL_TASK_ID` 的 `RESUME` 写入
2. **任务级恢复**（`task_resume_write`）：特定 `task_id` 的 `RESUME` 写入
3. **命名空间恢复**（`resume_map`）：基于 `namespace_hash` 的恢复值映射

`get_null_resume` 闭包中的 `consume` 参数控制是否消费（删除）全局恢复值——防止同一个恢复值被多个节点重复使用。

### interrupt() 函数

`interrupt()` 的完整逻辑：

```python
def interrupt(value: Any) -> Any:
    conf = get_config()["configurable"]
    scratchpad = conf[CONFIG_KEY_SCRATCHPAD]
    idx = scratchpad.interrupt_counter()      # 获取当前中断索引
    if scratchpad.resume:
        if idx < len(scratchpad.resume):
            # 有对应的恢复值，直接返回
            conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
            return scratchpad.resume[idx]
    v = scratchpad.get_null_resume(True)       # 尝试获取全局恢复值
    if v is not None:
        assert len(scratchpad.resume) == idx
        scratchpad.resume.append(v)
        conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
        return v
    raise GraphInterrupt((                     # 无恢复值，抛出中断
        Interrupt.from_ns(value=value, ns=conf[CONFIG_KEY_CHECKPOINT_NS]),
    ))
```

三条路径：
1. 任务级恢复值命中——直接返回
2. 全局恢复值命中——追加到 resume 列表并返回
3. 无恢复值——抛出 `GraphInterrupt`

## 5. 实现原理

### 中断/恢复的完整生命周期

一次完整的 interrupt-resume 周期分为五步：

**步骤 1：首次执行，遇到 interrupt()**

节点函数执行到 `interrupt(value)` 时，scratchpad 的 `resume` 为空、`get_null_resume` 返回 None，于是抛出 `GraphInterrupt`。Pregel 循环捕获此异常，将 `Interrupt(value=..., ns=...)` 保存到检查点，并终止执行。

**步骤 2：检查点保存**

Pregel 在每个超步骤（superstep）结束后调用 `create_checkpoint()` 保存当前 channel 状态。此时 `versions_seen` 记录了每个节点已经看到的 channel 版本，`channel_versions` 记录了当前所有 channel 的版本号。

**步骤 3：客户端恢复执行**

客户端使用 `Command(resume=value)` 恢复执行。Pregel 将恢复值作为 `RESUME` 类型的 pending write 存入检查点。

**步骤 4：重新执行节点**

Pregel 从检查点恢复 channel 状态，创建 scratchpad 时从 pending writes 中提取恢复值填充 `resume` 列表。节点重新执行，再次到达 `interrupt()` 时，`scratchpad.interrupt_counter()` 返回 0，`scratchpad.resume[0]` 存在，于是直接返回恢复值而不抛异常。

**步骤 5：继续执行**

节点拿到恢复值后继续执行，完成后的输出写入 channel，Pregel 创建新的检查点。

### ChannelVersions 与 versions_seen 的协作

这是 Pregel 决定"哪些节点需要执行"的核心机制：

- `channel_versions`：记录每个 channel 的当前版本号（单调递增）
- `versions_seen`：记录每个节点上次执行时各 channel 的版本号

当某次超步骤结束后，Pregel 遍历所有节点，对于节点 N，如果存在 channel C 使得 `channel_versions[C] > versions_seen[N][C]`，则节点 N 需要在下一轮执行。这保证了只有输入真正发生变化的节点才会被触发。

### 检查点恢复流程

1. `channels_from_checkpoint()` 从检查点的 `channel_values` 重建每个 channel 对象
2. 每个 channel 调用 `from_checkpoint(saved_value)` 恢复内部状态
3. managed value 不参与序列化，在恢复时重新初始化
4. scratchpad 从 pending writes 中解析恢复值

### Scratchpad 追踪中断位置

`interrupt_counter()` 使用 `LazyAtomicCounter` 返回 0, 1, 2... 的递增序列。在单节点多中断场景下，每次调用 `interrupt()` 都会递增计数器。恢复时，scratchpad 按索引从 `resume` 列表中取值，确保第一个中断对应第一个恢复值，第二个对应第二个，以此类推。

## 6. 动手实验

### 实验 1：构建人机协作审批流

```python
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    request: str
    approved: Optional[bool]
    reason: Optional[str]

def approve_node(state: State):
    decision = interrupt({"request": state["request"], "question": "批准还是驳回？"})
    return {"approved": decision["ok"], "reason": decision.get("reason", "")}

def result_node(state: State):
    if state["approved"]:
        return {"request": f"[已批准] {state['request']}"}
    return {"request": f"[已驳回] {state['request']} - {state['reason']}"}

builder = StateGraph(State)
builder.add_node("approve", approve_node)
builder.add_node("result", result_node)
builder.add_edge(START, "approve")
builder.add_edge("approve", "result")
builder.add_edge("result", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "exp1"}}

# 第一次调用——触发中断
result = graph.invoke({"request": "请假3天"}, config)
print("中断后：", result)  # 输出为空，因为执行被中断

# 恢复执行
result = graph.invoke(Command(resume={"ok": True, "reason": "同意"}), config)
print("恢复后：", result)  # {'request': '[已批准] 请假3天'}
```

### 实验 2：检查查检查点结构

```python
# 查看检查点的内部结构
state = graph.get_state(config)
checkpoint = state.values
print("channel_values:", state.values)
print("下一步节点:", state.next)
print("检查点配置:", state.config)

# 浏览检查点历史
for h in graph.get_state_history(config):
    print(f"Step {h.metadata.get('step', '?')}: "
          f"next={h.next}, values={h.values}")
```

### 实验 3：测试多中断顺序匹配

```python
class State2(TypedDict):
    name: str
    age: str
    result: str

def form_node(state: State2):
    name = interrupt("请输入姓名")
    age = interrupt("请输入年龄")
    return {"name": name, "age": age, "result": f"{name}，{age}岁"}

builder2 = StateGraph(State2)
builder2.add_node("form", form_node)
builder2.add_edge(START, "form")
builder2.add_edge("form", END)

graph2 = builder2.compile(checkpointer=InMemorySaver())
config2 = {"configurable": {"thread_id": "exp3"}}

# 第一次中断——停在第1个 interrupt
graph2.invoke({"name": "", "age": "", "result": ""}, config2)

# 恢复第1个中断，然后停在第2个 interrupt
graph2.invoke(Command(resume="张三"), config2)

# 恢复第2个中断，执行完成
result = graph2.invoke(Command(resume="25"), config2)
print(result)  # {'name': '张三', 'age': '25', 'result': '张三，25岁'}
```

这个实验验证了 `interrupt_counter()` 的顺序匹配机制：第一个 `interrupt` 对应第一个 `resume` 值，第二个对应第二个。

### 实验 4：使用 update_state 修改状态

```python
# 人为修改检查点状态
graph.update_state(config, {"approved": False, "reason": "管理员强制驳回"}, as_node="approve")
state = graph.get_state(config)
print("修改后状态：", state.values)
print("下一步：", state.next)  # 会显示需要重新执行的节点
```