## 音频增强接口

### 1. 获取可选项

- **URL**：`GET /api/audio/enhancement/options`
- **返回示例**

```json
{
  "noiseModes": [
    {"id": "none", "label": "不降噪", "description": "直接送入识别模型。", "recommended": false},
    {"id": "classic", "label": "经典谱减", "description": "经典谱减法，轻量快速。", "recommended": false},
    {"id": "improved", "label": "改进谱减（推荐）", "description": "自适应谱减，降低音乐噪声。", "recommended": true}
  ],
  "noiseStrength": {"min": 0.5, "max": 5.0, "default": 1.0},
  "dereverb": {
    "label": "启用 Dereverb（WPE）",
    "description": "选择后将对音频执行 WPE 混响消除。",
    "defaultEnabled": false,
    "parameters": {"delay": 3, "taps": 10, "iterations": 3}
  }
}
```

前端可使用该接口渲染：

```
降噪模式（3选1）：
( ) 不降噪
( ) 经典谱减
( ) 改进谱减（推荐）

混响消除（可选）：
[ ] 启用 Dereverb（WPE）
```

### 2. WebSocket 携带配置

客户端在发送 `audio.start` 握手时，附带 `enhancement` 字段，示例如下：

```json
{
  "type": "audio.start",
  "data": {
    "sessionId": "sess-123",
    "sampleRate": 16000,
    "format": "PCM16",
    "enhancement": {
      "noiseMode": "improved",
      "noiseStrength": 1.2,
      "enableDereverb": true
    }
  }
}
```

- `noiseMode`：`none` / `classic` / `improved`（默认 `none`）。
- `noiseStrength`：浮点数，范围 `0.5~5.0`，控制谱减强度。
- `enableDereverb`：布尔值，表示是否启用 WPE 混响消除。
- 进阶参数（可选）：`dereverbDelay`、`dereverbTaps`、`dereverbIterations`。

当握手成功，后端会在 `meta` 消息中返回 `enhancement` 字段，便于前端确认当前模式：

```json
{
  "type": "meta",
  "data": {
    "...": "...",
    "enhancement": {
      "noiseMode": "improved",
      "noiseStrength": 1.2,
      "enableDereverb": true,
      "dereverb": {"delay": 3, "taps": 10, "iterations": 3}
    }
  }
}
```

### 3. 流式处理效果

- 当前会话内的音频帧会先经过所选降噪模块（经典谱减/改进谱减），再进入可选的 WPE 混响消除，最后送至 ASR。
- 无需额外的任务接口，所有逻辑都在 WebSocket 会话生命周期内完成。
