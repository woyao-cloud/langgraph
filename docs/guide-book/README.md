# LangGraph 指南：问题、理论与实践

> 一本关于 LangGraph 的完整指南——它解决什么问题、基于什么理论、有哪些场景、怎么用 API

## 本书定位

这不是源码走读，而是一本面向实践者的指南。你将了解：
- **为什么**需要 LangGraph——它解决的问题和理论根基
- **什么时候**用 LangGraph——真实场景与架构模式
- **怎么用** LangGraph——API 详解与代码示例

## 目录

- [第 1 章：LangGraph 解决什么问题](./chapter01-why-langgraph.md)
- [第 2 章：理论基础](./chapter02-theory.md) — BSP 模型、Actor 模型、有限状态机
- [第 3 章：核心概念](./chapter03-core-concepts.md) — State、Node、Edge、Channel、Checkpoint
- [第 4 章：StateGraph API 详解](./chapter04-stategraph-api.md) — 图构建的完整 API
- [第 5 章：执行与流式输出](./chapter05-execution-and-streaming.md) — invoke、stream、StreamMode
- [第 6 章：状态持久化与检查点](./chapter06-checkpointing.md) — Checkpointer、thread_id、状态恢复
- [第 7 章：人机交互](./chapter07-human-in-the-loop.md) — interrupt、Command、审核流程
- [第 8 章：高级路由](./chapter08-advanced-routing.md) — Send、Command、map-reduce、子图
- [第 9 章：函数式 API](./chapter09-functional-api.md) — @entrypoint、@task
- [第 10 章：架构模式与最佳实践](./chapter10-patterns-and-practices.md) — 常见架构、反模式、调试技巧

## 阅读建议

- 第 1-3 章：理解概念，无需写代码
- 第 4-9 章：每章都有完整代码示例，建议动手运行
- 第 10 章：实战总结，适合有经验的读者