# 指令匹配识别模块设计

## 后端模式

- 服务支持两种匹配后端，通过环境变量 `COMMAND_MATCH_BACKEND` 控制：
  - `bm25_fuzz`（默认）：使用 BM25 召回 + RapidFuzz 模糊比对。该模式不依赖任何语义向量模型，适合 RK3588/Orange Pi 5 Plus 等板子。
  - `semantic`：沿用 `bge-small-zh` 语义向量，使用余弦相似度。
- 运行时只会初始化当前后端所需资源：当 `bm25_fuzz` 启用时不会加载 `SentenceTransformer` 模型。
- 阈值语义保持不变（0–1 之间的浮点数）。在 `bm25_fuzz` 模式下 RapidFuzz 的 0–100 分值会归一化为 0–1 再与阈值比较。

## 数据库结构

### `commands`
| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | INTEGER PK | 自增主键 |
| `user_id` | INTEGER | 指令所有者（使用开关的帐号）。若管理员为团队配置指令，则填管理员用户 ID；若支持员工独立指令集，则填员工 ID |
| `text` | TEXT UNIQUE | 指令原文 |
| `embedding` | BLOB | 语义向量（二进制存储）。当后端为 `bm25_fuzz` 时写入空字节以占位 |
| `created_at` / `updated_at` | DATETIME | 记录时间戳 |

> 当前版本以“拥有该指令集并控制开关的用户”为 `user_id`，即通常是管理员本人的 ID；后台可根据业务将员工映射到对应的管理员 ID。

### `command_settings`
| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | INTEGER PK |
| `user_id` | INTEGER UNIQUE | 对应用户 |
| `enable_matching` | BOOLEAN | 是否开启“指令识别” |
| `match_threshold` | FLOAT | 相似度阈值，默认 0.75 |
| `updated_at` | DATETIME | 更新时间 |

## 服务类与流程
1. **CommandService**
   - 初始化时根据 `COMMAND_MATCH_BACKEND` 选择后端。
   - 在 `semantic` 模式下实例化 `CommandEmbeddingService` 并在上传/更新指令时生成 embedding。
   - 在 `bm25_fuzz` 模式下跳过向量生成，数据库 `embedding` 字段写入 `b""`。
2. **CommandMatcher**
   - 针对 `semantic` 缓存 `(texts, vectors)`，并在缺失 embedding 时自动补齐。
   - 针对 `bm25_fuzz` 缓存 `(texts, BM25Okapi)`，查询阶段使用 RapidFuzz 对召回候选重排。
3. **匹配流程**
   - 读取用户设置（开关 + 阈值），若关闭匹配直接返回 `matched=false`。
   - 根据所选后端执行余弦匹配或 BM25+RapidFuzz 匹配，返回 `CommandMatchResult`。
   - RapidFuzz 分数会先 `/100` 再与阈值比较，response 中 `score` 字段同样返回归一化后的结果。

## REST 接口

### `POST /api/commands/upload`
- Body：`{ "commands": ["指令一", "指令二", ...] }`
- 行为：校验文本 → 按选择的后端生成 embedding 或写入空字节 → `INSERT OR REPLACE` 到 `commands` 表（按当前用户 ID）。
- 响应：`{ "inserted": n }`

### `POST /api/commands/toggle`
- Body：`{ "enabled": true | false, "match_threshold": 0.75 }`（阈值可选，不传则保持现值）
- 行为：更新 `command_settings.enable_matching` 以及可选阈值。
- 响应：返回最新 `GET /api/commands` 同结构

### `GET /api/commands`
- Query：`page`（默认 1）、`page_size`（默认 20，最大 200）
- 响应示例：
  ```json
  {
    "enabled": true,
    "match_threshold": 0.75,
    "items": [
      {"id": 1, "text": "各号注意...", "created_at": "..."},
      {"id": 2, "text": "站综合信息检查...", "created_at": "..."}
    ],
    "total": 42,
    "page": 1,
    "page_size": 20,
    "updated_at": "2024-01-01T12:00:00Z"
  }
  ```
- 用于前端分页展示指令列表。

### `GET /api/commands/search`
- Query：`q`（关键词，必填）、`page`（默认 1）、`page_size`（默认 20，最大 200）
- 响应：`{ "items": [...], "total": 5, "page": 1, "page_size": 20 }`，可用于模糊查询的分页展示或搜索结果列表。

### `PUT /api/commands/{command_id}`
- Body：`{ "text": "新的指令内容" }`
- 行为：更新指令文本，并在 `semantic` 模式下重新写入 embedding；若文本重复或记录不存在返回 400/404。

### `DELETE /api/commands/{command_id}`
- 删除指定指令；成功返回 `{ "deleted": true }`。

## WebSocket 交互
- ASR 产生每个 segment 时读取用户的 `enable_matching`：
  - `false`：返回 payload 中 `command_match: { "matched": false }`，不调用 matcher。
  - `true`：执行匹配，结果写入：
    ```json
    {
      "type": "final",
      "text": "...",
      "command_match": {
        "matched": true,
        "command": "站综合信息检查一分钟准备",
        "score": 0.85
      }
    }
    ```
  - 未匹配：`"command_match": { "matched": false }`
- 上传指令后需刷新 matcher 缓存（服务端在写库后已自动失效缓存）。
- WebSocket `audio.start` 握手数据需携带 `token`（与 REST 登录返回的 access token 相同），服务端据此定位当前用户并读取其指令及开关状态；若无 token 则默认不启用指令识别。
