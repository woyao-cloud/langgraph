# LangGraph 源码深度解读

> 面向 Python 程序员的框架内幕解析——从类型定义到执行引擎，逐模块走读核心代码

## 阅读指南

本书假设你已熟悉 Python 基础语法和标准库，但对以下高级特性可能需要温习：
- `Generic` / `TypeVar` / `Protocol` — 泛型与结构化子类型
- `Annotated[T, metadata]` — 类型注解的元数据扩展
- `dataclass(frozen=True, slots=True)` — 不可变数据类
- `NamedTuple` — 带命名字段的元组
- `Literal` / `TypedDict` — 字面量类型与结构化字典
- `async/await` / `contextmanager` — 异步与上下文管理
- `contextvars.ContextVar` — 协程安全的上下文传递
- `@overload` / `ParamSpec` — 类型系统高级特性

建议按章节顺序阅读。每章包含：
- **Python 进阶**：本章涉及的 Python 高级特性详解
- **代码走读**：逐段阅读关键源码，附逐行注释
- **运行原理**：数据流与控制流图解
- **实现细节**：关键函数的算法与边界处理
- **动手实验**：可运行的代码示例

## 目录

- [第 1 章：类型系统与常量定义](./chapter01-types-and-constants.md) — `types.py`, `constants.py`, `errors.py`
- [第 2 章：通道——状态存储原语](./chapter02-channels.md) — `channels/` 模块
- [第 3 章：图构建——从声明到编译](./chapter03-graph-construction.md) — `graph/` 模块
- [第 4 章：Pregel 引擎——入口与协议](./chapter04-pregel-entry.md) — `pregel/main.py`, `pregel/protocol.py`
- [第 5 章：Pregel 引擎——核心算法与循环](./chapter05-pregel-loop.md) — `pregel/_loop.py`, `pregel/_algo.py`, `pregel/_runner.py`
- [第 6 章：读写管线——数据的流入与流出](./chapter06-read-write-branch.md) — `pregel/_read.py`, `pregel/_write.py`, `pregel/_call.py`
- [第 7 章：检查点与中断恢复](./chapter07-checkpoint-resume.md) — `pregel/_checkpoint.py`, `_internal/_scratchpad.py`
- [第 8 章：函数式 API](./chapter08-functional-api.md) — `func/` 模块
- [第 9 章：内部基础设施](./chapter09-internals.md) — `_internal/` 模块
- [第 10 章：配置注入与运行时](./chapter10-config-and-runtime.md) — `config.py`, `runtime.py`, `managed/`

## 源码版本

基于 LangGraph v1.x，文件路径均为 `langgraph/` 下的相对路径。