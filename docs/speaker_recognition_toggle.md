# 说话人识别开关使用说明

## 背景

- 后端新增全局配置项 `enable_speaker_recognition`，用于控制实时会话中是否执行说话人识别。
- 该开关只在 **新建的 WebSocket 会话** 上生效；已经建立的会话仍按创建时的状态继续运行。
- 关闭识别后，WS 推送的 `partial`/`final` 消息里说话人字段会固定为 `"unknown"`，不会再发送 `type: speaker` 的候选更新。

## 接口

所有接口都要求管理员身份（需要携带后端颁发的 `Authorization: Bearer <token>`）。

### 1. 查询当前配置

```
GET /api/settings/system
```

响应示例：

```json
{
  "enable_speaker_recognition": true
}
```

前端可在设置页面加载时调用该接口以同步开关默认状态。

### 2. 更新配置

```
PATCH /api/settings/system
Content-Type: application/json

{
  "enable_speaker_recognition": false
}
```

- `true` 表示开启说话人识别（默认值），`false` 表示关闭。
- 调用成功后会立即更新数据库与服务器缓存，用于后续会话。
- 建议前端在提交后提示“新建会话生效”，避免误解。

响应示例：

```json
{
  "enable_speaker_recognition": false
}
```

## WebSocket 元数据

当客户端发送 `audio.start` 后，后端会在 `type: "meta"` 消息中返回：

```json
{
  "speakerRecognitionEnabled": true,
  ...
}
```

前端可据此在实时页面展示当前会话的说话人识别状态。如果后台关闭了该功能，该字段将为 `false`，同时不会收到 `type: speaker` 推送。

## 其他说明

- 新增全局设置会记录到后台事件日志（`event_type=settings`，`category=update`），方便审计。
- 声纹库管理、声纹聚合等接口仍可正常使用；关闭说话人识别不会影响声纹样本的上传与管理。
- 若需要恢复默认值，只需再次调用 PATCH，body 中设置为 `true` 即可。
