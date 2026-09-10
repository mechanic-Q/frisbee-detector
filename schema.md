# Wiki Schema

## Page Types

| Type | Directory | Purpose |
|------|-----------|---------|
| entity | wiki/entities/ | 具体模型/仓库/数据集（如 D-FINE、frisbee_merged_v2） |
| concept | wiki/concepts/ | 技术概念与方法（如 P2 头、SAHI、硬负样本） |
| source | wiki/sources/ | 外部来源（论文、博客、视频教程） |
| query | wiki/queries/ | 未定案的开放问题 |
| comparison | wiki/comparisons/ | 架构/方案对比分析 |
| synthesis | wiki/synthesis/ | 跨页汇聚的结论与决策 |

## Naming Conventions

- 文件：`kebab-case.md`
- Entities：官方名（如 `dfine-s.md`、`frisbee-merged-v2.md`）
- Concepts：描述性名词（如 `hard-negative-mining.md`）
- Sources：`author-year-slug.md`
- Queries：问题作 slug（如 `can-monocular-speed-be-accurate.md`）

## Frontmatter

所有页面必须包含：

```yaml
---
type: entity | concept | source | query | comparison | synthesis | overview
title: 人类可读标题
tags: []
related: []
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

Source 页额外包含：

```yaml
authors: []
year: YYYY
url: ""
```

## Cross-link Rules

- 正文用 `[[page-slug]]` 语法互链（graph builder 只识别裸 stem wikilink）
- frontmatter 的 `related:` 用 YAML 数组语法（不产生图边）

## Index Format

`wiki/index.md` 按类型分组列出全部页面：

```
- [[page-slug]] — 一行描述
```

## Log Format

`wiki/log.md` 逆时间序记录研究活动：

```
## YYYY-MM-DD

- 动作 / 发现
```
