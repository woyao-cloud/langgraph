# LangGraph 溆码解读：原理、应用与 Python 实践

> 面向 Python 程序员的框架深度指南——每个模块讲清楚三件事：它做什么、什么时候用、怎么实现的

## 本书特色

与普通源码走读不同，本书对每个模块都回答三个核心问题：

1. **功能性说明**：这段代码从用户视角看在做什么？解决什么问题？
2. **应用场景**：在什么情况下你会用到这个功能？有哪些典型用法？
3. **实现原理**：代码内部是怎么工作的？用了哪些 Python 高级特性？

每个模块还附带 **Python 进阶** 环节，讲解代码中使用的 Python 高级特性，让你在理解框架的同时提升 Python 功底。

## 目标读者

- 有 Python 基础，想深入理解 LangGraph 内部工作原理的开发者
- 想知道"什么时候该用什么功能"的实践者
- 对 Python 高级特性（泛型、Protocol、异步、描述符等）感兴趣的学习者

## 章节结构（每章统一格式）

| 小节 | 内容 |
|------|------|
| 功能概览 | 模块对外提供的功能和 API |
| 应用场景 | 典型使用场景、代码示例 |
| Python 进阶 | 本章涉及的 Python 高级特性详解 |
| 代码走读 | 关键源码逐段解读 |
| 实现原理 | 内部工作机制、数据流、控制流 |
| 动手实验 | 可运行的代码示例 |

## 目录

- [第 1 章：核心类型与公共 API](./chapter01-types-and-api.md) — `types.py`, `constants.py`, `errors.py`
- [第 2 章：通道——状态存储原语](./chapter02-channels.md) — `channels/` 模块
- [第 3 章：图构建——声明与编译](./chapter03-graph-construction.md) — `graph/` 模块
- [第 4 章：Pregel 引擎——入口与执行协议](./chapter04-pregel-entry.md) — `pregel/main.py`, `pregel/protocol.py`
- [第 5 章：Pregel 引擎——算法与循环](./chapter05-pregel-loop.md) — `pregel/_loop.py`, `pregel/_algo.py`, `pregel/_runner.py`
- [第 6 章：读写管线](./chapter06-read-write-pipeline.md) — `pregel/_read.py`, `pregel/_write.py`, `pregel/_call.py`
- [第 7 章：检查点与中断恢复](./chapter07-checkpoint-resume.md) — `pregel/_checkpoint.py`, `_internal/_scratchpad.py`
- [第 8 章：函数式 API](./chapter08-functional-api.md) — `func/` 模块
- [第 9 章：内部基础设施](./chapter09-internals.md) — `_internal/` 模块
- [第 10 章：配置注入与运行时](./chapter10-config-and-runtime.md) — `config.py`, `runtime.py`, `managed/`

## 源码版本

基于 LangGraph v1.x，路径为 `langgraph/` 下的相对路径。