# 第 2 章：通道系统

> 源码文件：`channels/` 模块

## Java 桥梁

在 Java 并发编程中，你用过 `ConcurrentHashMap`、`BlockingQueue`、`CountDownLatch` 等并发原语。LangGraph 的通道系统本质上也是一类并发安全的状态容器——但在 BSP（Bulk Synchronous Parallel）模型中，"并发"的含义不同：不是多线程并发访问，而是多节点在同一超步中写入、在超步结束后统一应用。

你可以把通道类比为 Java 中的 **变量 + 并发控制策略**：
- `LastValue` ≈ `AtomicReference`（只允许一次写入）
- `BinaryOperatorAggregate` ≈ `Accumulator`（用 reduce 函数聚合值）
- `Topic` ≈ `BlockingQueue`（发布-订阅通道）
- `NamedBarrierValue` ≈ `CyclicBarrier`（等待所有参与者到达）
- `EphemeralValue` ≈ `ThreadLocal`（每步清除）

## Python 概念速查

### `ABC`（抽象基类）— Python 版的 Java interface

Java：

```java
public abstract class BaseChannel<V, U, C> {
    public abstract V get();
    public abstract boolean update(List<U> values);
}
```

Python：

```python
from abc import ABC, abstractmethod

class BaseChannel(Generic[Value, Update, Checkpoint], ABC):
    @abstractmethod
    def get(self) -> Value: ...

    @abstractmethod
    def update(self, values: Sequence[Update]) -> bool: ...
```

关键区别：
- Python 的 `ABC` 用 `@abstractmethod` 标记抽象方法，和 Java 的 `abstract` 方法等价
- `ABC` 不能用 `@abstractmethod` 标记属性，但可以用 `@property + @abstractmethod` 组合
- Python 没有接口（interface）和抽象类的区分——都用 `ABC`
- `ABC` 的子类如果不实现所有抽象方法，**实例化时会报 TypeError**（和 Java 一样）

### `@property` — Python 的属性访问器

Java：

```java
public class LastValue {
    private Class<?> valueType;
    public Class<?> getValueType() { return valueType; }
}
```

Python：

```python
class LastValue(BaseChannel[Value, Value, Value]):
    @property
    def ValueType(self) -> type[Value]:
        return self.typ
```

`@property` 把方法调用伪装成属性访问：`channel.ValueType` 而不是 `channel.get_ValueType()`。这类似 C# 的属性或 Kotlin 的属性，但 Java 没有直接等价物（Java 需要 getter 方法）。

### `Self` 类型 — 方法返回自身类型

Java 中返回自身类型通常用递归泛型：

```java
public abstract class BaseChannel<V, U, C, THIS extends BaseChannel<V, U, C, THIS>> {
    public abstract THIS fromCheckpoint(C checkpoint);
}
```

Python 用 `Self`：

```python
from typing_extensions import Self

class BaseChannel(Generic[Value, Update, Checkpoint], ABC):
    def copy(self) -> Self:
        return self.from_checkpoint(self.checkpoint())
```

`Self` 自动替换为当前类类型，不需要复杂的递归泛型。

### `MISSING` 哨兵值 — Python 的"未设置"标记

Java 中你用 `null` 表示"未设置"。Python 也可以用 `None`，但 `None` 本身可能是合法值。LangGraph 用 `MISSING` 哨兵值区分"未设置"和"值为 None"：

```python
from langgraph._internal._typing import MISSING

self.value = MISSING  # 未设置

if self.value is not MISSING:  # 用 is 而非 == 比较
    return self.value
```

这类似 Java 的 `Optional.empty()` 与 `Optional.of(null)` 的区别，但 Python 的实现更简单——一个单例对象。

## 代码走读

### 1. BaseChannel — 通道抽象基类

```python
class BaseChannel(Generic[Value, Update, Checkpoint], ABC):
    __slots__ = ("key", "typ")

    def __init__(self, typ: Any, key: str = "") -> None:
        self.typ = typ   # 通道存储的值类型
        self.key = key   # 通道的键名（对应 State 的字段名）

    # --- 抽象属性 ---
    @property
    @abstractmethod
    def ValueType(self) -> Any: ...   # 值的类型

    @property
    @abstractmethod
    def UpdateType(self) -> Any: ...  # 更新的类型

    # --- 序列化/反序列化 ---
    def copy(self) -> Self: ...           # 创建副本
    def checkpoint(self) -> Checkpoint | Any: ...  # 序列化为检查点
    @abstractmethod
    def from_checkpoint(self, checkpoint: Checkpoint | Any) -> Self: ...  # 从检查点恢复

    # --- 读取 ---
    @abstractmethod
    def get(self) -> Value: ...          # 获取当前值
    def is_available(self) -> bool: ...  # 是否有值

    # --- 写入 ---
    @abstractmethod
    def update(self, values: Sequence[Update]) -> bool: ...  # 应用更新
    def consume(self) -> bool: ...  # 通知：订阅任务已运行
    def finish(self) -> bool: ...   # 通知：Pregel 运行即将结束
```

**三个生命周期方法 `update → consume → finish`：**

| 方法 | 调用时机 | Java 类比 |
|------|---------|----------|
| `update()` | 每步结束时，批量应用写入 | `ConcurrentHashMap.put()` |
| `consume()` | 订阅此通道的任务运行后 | `CountDownLatch.countDown()` |
| `finish()` | Pregel 即将结束时 | `Runtime.addShutdownHook()` |

**泛型参数 `Generic[Value, Update, Checkpoint]`：**
- `Value` — 通道存储的值类型（对外读取时的类型）
- `Update` — 更新操作的输入类型（写入时的类型）
- `Checkpoint` — 序列化时的类型（保存到检查点的类型）

这三者可以不同！例如 `Topic` 通道：
- `Value = Sequence[T]`（读取时返回列表）
- `Update = T | list[T]`（写入时可以是单个值或列表）
- `Checkpoint = list[T]`（序列化为列表）

### 2. LastValue — 最简单的通道

```python
class LastValue(Generic[Value], BaseChannel[Value, Value, Value]):
    __slots__ = ("value",)

    def __init__(self, typ: Any, key: str = "") -> None:
        super().__init__(typ, key)
        self.value = MISSING

    def update(self, values: Sequence[Value]) -> bool:
        if len(values) == 0:
            return False
        if len(values) != 1:
            raise InvalidUpdateError(
                f"At key '{self.key}': Can receive only one value per step. "
                "Use an Annotated key to handle multiple values."
            )
        self.value = values[-1]  # 取最后一个值
        return True

    def get(self) -> Value:
        if self.value is MISSING:
            raise EmptyChannelError()
        return self.value
```

**核心规则：每步只能接收一个值。** 如果两个节点在同一超步中对同一个 `LastValue` 通道写入，就抛出 `InvalidUpdateError`。

**为什么？** `LastValue` 用于 State 中没有 reducer 的字段。例如 `State(TypedDict)` 的 `count: int` 字段，如果两个节点同时设置 `count = 1` 和 `count = 2`，这是矛盾的——应该用 `Annotated[int, operator.add]` 代替。

**Java 类比：** `LastValue` ≈ `AtomicReference<V>`，但限制更强——`compareAndSet` 只允许一次成功。

### 3. BinaryOperatorAggregate — 聚合器通道

```python
class BinaryOperatorAggregate(Generic[Value], BaseChannel[Value, Value, Value]):
    __slots__ = ("value", "operator")

    def __init__(self, typ: type[Value], operator: Callable[[Value, Value], Value]):
        super().__init__(typ)
        self.operator = operator
        # 用类型的零值初始化
        try:
            self.value = typ()  # 例如 list() → [], int() → 0
        except Exception:
            self.value = MISSING

    def update(self, values: Sequence[Value]) -> bool:
        if not values:
            return False
        if self.value is MISSING:
            self.value = values[0]
            values = values[1:]
        seen_overwrite: bool = False
        for value in values:
            is_overwrite, overwrite_value = _get_overwrite(value)
            if is_overwrite:
                if seen_overwrite:
                    raise InvalidUpdateError("Can receive only one Overwrite per super-step.")
                self.value = overwrite_value
                seen_overwrite = True
                continue
            if not seen_overwrite:
                self.value = self.operator(self.value, value)
        return True
```

**核心规则：用二元操作符聚合所有写入值。** 类似 Java 的 `Stream.reduce()` 或 `Collector.accumulator()`。

**典型使用场景：**

```python
# operator.add: 列表拼接
import operator
BinaryOperatorAggregate(list, operator.add)
# [1, 2] + [3] → [1, 2, 3]

# 自定义 reducer
def append_if_not_none(a: list, b: int | None) -> list:
    return a + [b] if b is not None else a
BinaryOperatorAggregate(list, append_if_not_none)
```

**`Overwrite` 机制：** `Overwrite` 允许绕过 reducer 直接覆盖值。类似 Java 中你有时想跳过 `accumulate()` 直接 `set()`：

```python
# 正常更新：使用 reducer (operator.add)
return {"messages": ["新消息"]}    # messages = 旧消息 + ["新消息"]

# Overwrite：绕过 reducer 直接替换
return {"messages": Overwrite(value=["全新列表"])}  # messages = ["全新列表"]
```

**`typ()` 零值初始化的技巧：** Python 中 `list()` 返回 `[]`，`int()` 返回 `0`，`dict()` 返回 `{}`。这和 Java 不同——Java 的 `int` 默认就是 `0`，但 `List` 需要显式初始化。Python 的 `typ()` 利用类型的无参构造函数创建零值。

### 4. Topic — 发布订阅通道

```python
class Topic(Generic[Value], BaseChannel[Sequence[Value], Value | list[Value], list[Value]]):
    __slots__ = ("values", "accumulate")

    def __init__(self, typ: type[Value], accumulate: bool = False) -> None:
        self.accumulate = accumulate  # 是否跨步累积
        self.values = list[Value]()

    def update(self, values: Sequence[Value | list[Value]]) -> bool:
        updated = False
        if not self.accumulate:
            updated = bool(self.values)
            self.values = list[Value]()  # 不累积：每步清空
        if flat_values := tuple(_flatten(values)):
            updated = True
            self.values.extend(flat_values)
        return updated
```

**`accumulate` 参数决定行为：**
- `accumulate=False`：每步开始时清空，类似 Java 的 `LinkedBlockingQueue.drainTo()` + 清空
- `accumulate=True`：跨步累积，类似 Java 的 `ConcurrentLinkedQueue`（永远追加）

**`Topic` 的核心用途是 `TASKS` 通道**——存储 `Send` 对象。当条件边返回多个 `Send` 时，它们被写入 `TASKS` Topic，在下一步中每个 `Send` 创建一个 PUSH 任务。

**`_flatten` 辅助函数：** 把混合了单值和列表的输入展平。类似 Java `Stream.flatMap()`：

```python
def _flatten(values: Sequence[Value | list[Value]]) -> Iterator[Value]:
    for value in values:
        if isinstance(value, list):
            yield from value    # 展开列表
        else:
            yield value         # 单值直接产出
```

### 5. EphemeralValue — 瞬态通道

```python
class EphemeralValue(Generic[Value], BaseChannel[Value, Value, Value]):
    __slots__ = ("value", "guard")

    def update(self, values: Sequence[Value]) -> bool:
        if len(values) == 0:
            if self.value is not MISSING:
                self.value = MISSING  # 没有新值就清除！
                return True
            return False
        if len(values) != 1 and self.guard:
            raise InvalidUpdateError(...)
        self.value = values[-1]
        return True
```

**核心行为：如果本步没有新值写入，值自动清除。** 这类似 Java 的 `ThreadLocal`——请求结束后清除，但更严格：不等到请求结束，每步结束就清除。

**用途：** `__start__`（START）通道用 `EphemeralValue`。图的输入只在第一步有效，后续步骤如果没有新输入，`__start__` 通道自动清空，防止 START 节点被重复触发。

### 6. NamedBarrierValue — 屏障通道

```python
class NamedBarrierValue(Generic[Value], BaseChannel[Value, Value, set[Value]]):
    __slots__ = ("names", "seen")

    def __init__(self, typ: type[Value], names: set[Value]) -> None:
        self.names = names  # 期望收到的所有名字
        self.seen: set[str] = set()  # 已收到的名字

    def update(self, values: Sequence[Value]) -> bool:
        for value in values:
            if value in self.names:
                if value not in self.seen:
                    self.seen.add(value)
            else:
                raise InvalidUpdateError(f"Value {value} not in {self.names}")
        return updated

    def get(self) -> Value:
        if self.seen != self.names:
            raise EmptyChannelError()  # 所有名字都收到后才可用
        return None

    def consume(self) -> bool:
        if self.seen == self.names:
            self.seen = set()  # 消费后清空
            return True
        return False
```

**核心行为：等待所有命名源写入后才变为可用。** 类似 Java 的 `CyclicBarrier` 或 `CountDownLatch`：

```java
// Java 等价
CountDownLatch latch = new CountDownLatch(3);  // 等待 3 个节点
latch.countDown();  // 节点 A 完成
latch.countDown();  // 节点 B 完成
latch.await();      // 等待节点 C 完成
```

**用途：** 当你写 `graph.add_edge(["A", "B", "C"], "D")` 时，LangGraph 创建一个 `NamedBarrierValue(names={"A", "B", "C"})`。节点 D 只有在 A、B、C 都执行完毕后才会被触发。

**`consume()` 的作用：** 任务运行后清空 `seen`，防止下次被错误触发。类似 `CyclicBarrier.reset()`。

### 7. UntrackedValue — 不持久化的通道

```python
class UntrackedValue(Generic[Value], BaseChannel[Value, Value, Value]):
    """写入值不会被持久化到检查点的通道。"""
```

**用途：** 存储不需要跨重启保留的瞬态数据。类似 Java 中你存在 `HashMap` 而非数据库的数据——程序重启后就没了。

### 8. AnyValue — 宽松的 LastValue

```python
class AnyValue(Generic[Value], BaseChannel[Value, Value, Value]):
    """和 LastValue 类似，但静默接受多个值（取最后一个）。"""
```

**用途：** 当多个节点可能写入同一个通道，但你知道它们的值相同时使用。类似 Java 的 `AtomicReference` 但用 `lazySet` 语义——多个写入不报错，取最后一个。

## 运行原理

### 通道类型选择流程

当你定义 State 并编译 StateGraph 时，通道类型自动选择：

```
State(TypedDict) 的每个字段
    │
    ├── Annotated[type, reducer]  →  BinaryOperatorAggregate(type, reducer)
    │
    ├── type（无 Annotated）      →  LastValue(type)
    │
    └── ManagedValue 子类         →  managed value（非通道，见第 10 章）
```

当编译 StateGraph 时，边的类型决定额外通道：

```
graph.add_edge("A", "B")
    → 创建 EphemeralValue 通道 "branch:to:B"

graph.add_edge(["A", "B", "C"], "D")
    → 创建 NamedBarrierValue(names={"A", "B", "C"}) 通道

条件边
    → 创建 EphemeralValue(guard=False) 通道
```

### 通道版本追踪

每个通道在检查点中有一个版本号（`ChannelVersions`）。每当 `update()` 返回 `True`（值确实改变了），版本号递增。版本号用于：
1. **触发计算**：节点的 `triggers` 列表中的通道版本如果比上次看到的新，就触发该节点
2. **中断判断**：检查上次中断以来哪些通道被更新了

```
步骤 N:
  通道 A: version=1  →  节点 X 写入  →  version=2
  通道 B: version=1  →  无写入  →  version=1

步骤 N+1:
  节点 Y 的 triggers=[A, B]
  A 版本从 1→2（新），触发 Y
  B 版本没变，不触发 Y
```

## 动手实验

```python
# 实验 1：通道的基本行为
from langgraph.channels.last_value import LastValue
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.topic import Topic
from langgraph._internal._typing import MISSING
import operator

# LastValue — 只接受一个值
lv = LastValue(int, "count")
lv.update([42])
print(lv.get())  # 42

try:
    lv.update([1, 2])  # 两个值 → 报错
except Exception as e:
    print(f"Error: {type(e).__name__}")  # InvalidUpdateError

# BinaryOperatorAggregate — 聚合多个值
agg = BinaryOperatorAggregate(list, operator.add)
agg.update([[1, 2]])      # 写入 [1, 2]
agg.update([[3, 4]])      # 写入 [3, 4]
print(agg.get())  # [1, 2, 3, 4]  — 用 operator.add 聚合

# Topic — 发布订阅
topic = Topic(str, accumulate=False)
topic.update(["a", "b"])
print(list(topic.get()))  # ['a', 'b']
topic.update([])          # 不累积，清空
try:
    topic.get()  # 空了 → EmptyChannelError
except Exception as e:
    print(f"Error: {type(e).__name__}")

# 实验 2：Overwrite 绕过 reducer
from langgraph.types import Overwrite

agg2 = BinaryOperatorAggregate(list, operator.add)
agg2.update([[1, 2]])
agg2.update([[3]])
print(agg2.get())  # [1, 2, 3]  — 正常聚合

agg2.update([Overwrite(value=[99])])  # 绕过 reducer
print(agg2.get())  # [99]  — 直接替换

# 实验 3：NamedBarrierValue 屏障
from langgraph.channels.named_barrier_value import NamedBarrierValue

barrier = NamedBarrierValue(str, names={"A", "B", "C"})
barrier.update(["A"])  # A 完成
try:
    barrier.get()  # 还没全到齐 → EmptyChannelError
except Exception:
    print("Barrier not satisfied yet")

barrier.update(["B"])  # B 完成
barrier.update(["C"])  # C 完成
print(barrier.get())   # None — 所有节点都到了，通道可用
print(barrier.is_available())  # True
```