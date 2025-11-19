# 指令匹配识别模块设计

## 匹配模式

- 指令匹配统一采用 BM25 召回 + RapidFuzz 模糊比对：
  - 只依赖 `rank-bm25` 与 `rapidfuzz`，不会加载 `SentenceTransformer`、`scikit-learn` 等重量级依赖，适合 RK3588/Orange Pi 5 Plus 等板子。
  - 阈值语义保持在 0–1 之间的浮点数，RapidFuzz 的 0–100 分值会先归一化为 0–1 再与阈值比较。
  - 依赖 `jieba` 分词，若板子上未装将自动退化到按字符切分。

## 数据库结构

### `commands`
| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | INTEGER PK | 自增主键 |
| `user_id` | INTEGER | 指令所有者（使用开关的帐号）。若管理员为团队配置指令，则填管理员用户 ID；若支持员工独立指令集，则填员工 ID |
| `text` | TEXT UNIQUE | 指令原文 |
| `status` | TEXT | `enabled`/`disabled`，禁用的指令不会参与匹配 |
| `embedding` | BLOB | 兼容旧结构的占位字段，现阶段写入 `b""` |
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
   - 管理上传、分页、状态更新及用户阈值配置；写库时始终写入 `embedding=b""`。
2. **CommandMatcher**
   - 针对每个用户缓存 `(texts, normalized_texts, BM25Okapi)`；只加载 `status=enabled` 的指令。
   - 查询阶段使用 BM25 召回 top-k，再用 RapidFuzz 的 `token_set_ratio` 对候选重排。
3. **匹配流程**
   - 读取用户设置（开关 + 阈值），若关闭匹配直接返回 `matched=false`。
   - 执行 BM25+RapidFuzz 匹配，将 `fuzz` 返回的 0–100 分数除以 100 后与阈值比较，`score` 字段同样返回归一化后的结果。

## REST 接口

### `POST /api/commands/upload`
- Body：`{ "commands": ["指令一", "指令二", ...] }`
- 行为：校验文本 → `INSERT OR REPLACE` 到 `commands` 表（按当前用户 ID），`embedding` 写入空字节。
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
- 行为：更新指令文本（若文本重复或记录不存在返回 400/404）。

### `PATCH /api/commands/{command_id}/status`
- Body：`{ "status": "enabled" | "disabled" }`
- 行为：切换单条指令状态；禁用的指令不会再被 CommandMatcher 缓存及匹配逻辑使用。成功后返回最新的指令信息。

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
