# LangGraph 源码解读：从 Java 到 Python

> 一本面向 Java 程序员的 LangGraph 框架源码走读指南

## 目录

- [第 1 章：入口与类型系统](./chapter01-types-and-constants.md) — `types.py`, `constants.py`, `errors.py`
- [第 2 章：通道系统](./chapter02-channels.md) — `channels/` 模块
- [第 3 章：图构建](./chapter03-graph-construction.md) — `graph/` 模块
- [第 4 章：Pregel 执行引擎（上）](./chapter04-pregel-entry.md) — `pregel/main.py`, `pregel/protocol.py`
- [第 5 章：Pregel 执行引擎（下）](./chapter05-pregel-loop.md) — `pregel/_loop.py`, `pregel/_algo.py`
- [第 6 章：读写与分支](./chapter06-read-write-branch.md) — `pregel/_read.py`, `pregel/_write.py`, `pregel/_call.py`
- [第 7 章：检查点与恢复](./chapter07-checkpoint-resume.md) — `pregel/_checkpoint.py`, `_internal/_scratchpad.py`
- [第 8 章：函数式 API](./chapter08-functional-api.md) — `func/` 模块
- [第 9 章：内部基础设施](./chapter09-internals.md) — `_internal/` 模块
- [第 10 章：配置与运行时](./chapter10-config-and-runtime.md) — `config.py`, `runtime.py`, `managed/`

## 阅读建议

- 每章结构：Java 桥梁 → Python 概念速查 → 代码走读 → 运行原理 → 动手实验
- 建议按章节顺序阅读，后续章节依赖前面章节的概念
- 源码基于 LangGraph v1.x，文件路径均为 `langgraph/` 下的相对路径