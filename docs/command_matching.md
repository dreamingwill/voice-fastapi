# 指令匹配识别模块设计

## 数据库结构

### `commands`
| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | INTEGER PK | 自增主键 |
| `user_id` | INTEGER | 指令所有者（使用开关的帐号）。若管理员为团队配置指令，则填管理员用户 ID；若支持员工独立指令集，则填员工 ID |
| `text` | TEXT UNIQUE | 指令原文 |
| `embedding` | BLOB | 512 维 `float32` 向量（二进制存储） |
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
1. **CommandEmbeddingService**：封装 `SentenceTransformer("models/bge-small-zh")`，提供 `encode(texts)` 返回归一化向量。
2. **CommandRepository**：负责 `commands` 和 `command_settings` 的 CRUD。
3. **CommandMatcher**：缓存指定用户的全部指令向量，执行余弦相似度匹配，返回 `{"matched": bool, "command": str | None, "score": float}`。

## REST 接口

### `POST /api/commands/upload`
- Body：`{ "commands": ["指令一", "指令二", ...] }`
- 行为：校验文本 → 生成 embedding → `INSERT OR REPLACE` 到 `commands` 表（按当前用户 ID）。
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
- Query：`q`（关键词，必填）、`limit`（默认 20，最大 200）
- 返回模糊匹配的指令数组 `items`，用于前端局部搜索或自动补全。

### `PUT /api/commands/{command_id}`
- Body：`{ "text": "新的指令内容" }`
- 行为：更新指令文本并重新写入 embedding；若文本重复或记录不存在返回 400/404。

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

## 其它注意事项
- matcher 阈值默认 0.75，可放配置或 settings 表。
- `commands` 表可以按 `user_id` + `text` 建唯一索引，支持多用户。
- 若环境未预装模型，需确保 `models/bge-small-zh` 可用。*** End Patch
