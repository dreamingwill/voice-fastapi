# Config 说明

这个目录存放 `voice-fastapi` 的启动配置样例，主要供：

- `python main.py --config config/app_config.json`
- `launch.sh`
- `start.sh`

使用。

## 配置优先级

当前仓库里，配置来源分成两层：

1. `app_config*.json`
2. 环境变量覆盖部分字段

对 `app_config*.json` 中的大多数字段，启动时会读入并作为命令行参数默认值。

对转发相关字段，目前优先级是：

- `COMMAND_FORWARD_URL` 环境变量优先
- `COMMAND_FORWARD_TIMEOUT` 环境变量优先
- 如果环境变量没有设置，则使用 `app_config*.json` 中的 `command_forward_url` / `command_forward_timeout`

不过要注意一件事：

当前 Linux 线上部署如果是由 `systemd` 调用 `start.sh` 启动，那么普通终端里的手动 `export COMMAND_FORWARD_URL=...` 并不会传给服务进程。

因此按当前部署约定，想修改 Linux 服务实际使用的转发地址，最直接的方式仍然是修改 [start.sh](/Users/zrh/Projects/speak/voice-fastapi/start.sh:4) 里的默认值，然后重启服务。

## 目录中的文件

- `app_config.json`
  默认主配置，当前更偏 RK3588 + `offline_sense_voice` 场景。
- `app_config_mac.json`
  macOS 上的 `streaming_transducer` 样例。
- `app_config_mac_offline_sense_voice.json`
  macOS 上的 `offline_sense_voice` 样例。
- `app_config_rk3588_zipformer.json`
  RK3588 上的 `streaming_transducer` 样例。
- `app_config_test_rknn.json`
  一份偏测试用途的 RKNN streaming 配置。
- `command_text_aliases.json`
  指令文本纠错/别名表，不是主启动配置。

## 通用字段

下面这些字段在多数配置文件里都可能出现。

### 服务监听

- `host`
  FastAPI/uvicorn 监听地址。`0.0.0.0` 表示允许局域网或容器外访问。
- `port`
  服务端口。

### 说话人识别

- `model_path`
  说话人识别模型路径，用于识别当前讲话人是谁。
- `sample_rate`
  音频采样率，通常是 `16000`。ASR、VAD、说话人识别最好保持一致。
- `threshold`
  说话人识别阈值。相似度高于该值才认为匹配到已知用户。
  值越高越严格，误识别更少，但“识别不出来”的情况会变多。
- `min_spk_seconds`
  进行说话人识别前，至少累计多少秒音频。
  值太小容易不稳定，值太大则需要更久才能识别出说话人。

### 推理后端

- `provider`
  推理后端，例如 `cpu`、`rknn`。
  它需要和模型文件格式匹配：
  `rknn` 一般对应 `.rknn` 模型，非 `rknn` 一般对应 `.onnx`。
- `num_threads`
  推理线程数。CPU 模式下通常更敏感；RKNN 模式下也会影响部分组件负载。

### 命令转发

- `command_forward_url`
  当语音识别命中某条指令后，后端将把 `projectCode`、`speaker`、`createTime` POST 到这个地址。
  如果为空，则自动转发功能视为未配置。
- `command_forward_timeout`
  转发 HTTP 请求超时时间，单位秒。
  下游接口慢、网络不稳定时可以适度调大。

## ASR 模式字段

`asr_mode` 决定 ASR 引擎使用哪一套模型和参数。

### `asr_mode = "streaming_transducer"`

适合流式边说边出字。

#### 必填字段

- `tokens`
  词表文件。
- `encoder`
  编码器模型路径。
- `decoder`
  解码器模型路径。
- `joiner`
  joiner 模型路径。

#### 常用字段

- `decoding_method`
  解码方法，当前样例使用 `greedy_search`。
- `max_active_paths`
  多路径解码时的活跃路径数。当前代码里支持，但现有样例一般没有显式配置。
- `blank_penalty`
  blank token 惩罚项。适合用来微调输出倾向，减轻某些吞字或迟迟不出字的问题。
- `hotwords_file`
  热词文件路径。适合把高频专有名词、口令、代号提前注入。
- `hotwords_score`
  热词加权分数。值越大，热词越容易被命中，但也可能带来误偏置。
- `hr_rule_fsts`
  热词规则 FST 文件。
- `hr_lexicon`
  热词词典文件。

#### 端点规则

这些字段只对 `streaming_transducer` 有意义，用于控制 sherpa-onnx 的 endpoint detection：

- `rule1_min_trailing_silence`
  规则 1 的最小尾部静音时间。
- `rule2_min_trailing_silence`
  规则 2 的最小尾部静音时间。
- `rule3_min_utterance_length`
  规则 3 的最短语句长度。

它们共同决定“什么时候把一段话视为结束”。
如果发现切句太快、太慢、总被截断，优先调这里。

### `asr_mode = "offline_sense_voice"`

适合按句处理，一般配合 VAD 先切段，再对整段做识别。

#### 必填字段

- `sense_voice_model`
  SenseVoice 模型路径。
- `tokens`
  词表文件。
- `vad_max_utterance_ms`
  单句最大时长，必须大于 0。当前应用启动时会校验这个条件。

#### 常用字段

- `sense_voice_use_itn`
  是否开启 ITN。
  开启后会更倾向输出规范化文本，例如数字、时间等格式更整齐。
- `sense_voice_language`
  识别语言。
  `auto` 表示自动判断；如果场景固定，也可以直接指定以减少歧义。
- `feature_dim`
  声学特征维度，通常跟模型配套。
- `decoding_method`
  解码方法，当前样例一般为 `greedy_search`。
- `hr_rule_fsts`
  热词规则 FST 文件。
- `hr_lexicon`
  热词词典文件。

## VAD 字段

这组参数控制语音活动检测，也就是“什么时候开始收一段话、什么时候结束一段话”。

- `vad_pre_roll_ms`
  开始判定为讲话前，向前保留多少毫秒音频，避免句首被裁掉。
- `vad_post_roll_ms`
  结束判定后，额外保留多少毫秒音频，避免句尾被裁掉。
- `vad_snr_open_db`
  触发开口的信噪比阈值，越高越严格。
- `vad_open_min_ms`
  至少持续多少毫秒才认定为真正开口。
- `vad_end_silence_ms`
  静音持续多久后认定一句话结束。
- `vad_max_utterance_ms`
  单句最大时长。
  对 `offline_sense_voice` 是关键字段；过长会导致句子拖太久，过短会导致长句被硬切。
- `vad_reopen_min_ms`
  一次结束后，重新打开新语句前要求的最小时长。
- `vad_noise_margin_db`
  噪声更新边际。
  用来平衡噪声地板更新速度和稳定性。
- `vad_noise_bootstrap_ms`
  噪声基线初始化阶段时长。

如果遇到以下问题，可以优先这样调整：

- 句首被吃掉：增大 `vad_pre_roll_ms`
- 句尾被截断：增大 `vad_post_roll_ms` 或 `vad_end_silence_ms`
- 噪音环境下误触发：增大 `vad_snr_open_db` 或 `vad_open_min_ms`
- 一句话总是拖得太久才结束：减小 `vad_end_silence_ms`

## 哪些字段是模式相关的

### streaming_transducer 主要关心

- `tokens`
- `encoder`
- `decoder`
- `joiner`
- `blank_penalty`
- `hotwords_file`
- `hotwords_score`
- `rule1_min_trailing_silence`
- `rule2_min_trailing_silence`
- `rule3_min_utterance_length`

### offline_sense_voice 主要关心

- `asr_mode`
- `sense_voice_model`
- `tokens`
- `sense_voice_use_itn`
- `sense_voice_language`
- `feature_dim`
- `vad_max_utterance_ms`

## `command_text_aliases.json` 是做什么的

这个文件不是 `main.py --config ...` 的主配置。

它用于“指令文本纠错/归一化”，典型用途包括：

- 把常见 ASR 错字统一成标准指令文本
- 处理口令别名
- 把容易识别错的词替换成系统里真正存储的文本

当前文件包含两类规则：

- `exact_aliases`
  完整句子级别的精确替换。
- `replacements`
  子串级别的局部替换。

注意：它默认不会自动生效，只有在环境变量 `COMMAND_TEXT_ALIASES_PATH` 指向该文件时才会被加载。

例如：

```bash
export COMMAND_TEXT_ALIASES_PATH="./config/command_text_aliases.json"
python main.py --config config/app_config.json
```

## 修改建议

- 先复制最接近你设备/模型的一份配置，再改，不要直接混用不同模式的字段。
- 改 `provider` 时，同时检查模型文件后缀是否匹配。
- 改 VAD 时一次只改 1 到 2 个参数，方便回溯效果。
- 如果部署环境有差异，把默认值放进 `app_config*.json`，再用环境变量做覆盖，比把地址硬编码在脚本里更容易维护。

## 最小示例

### RK3588 + offline_sense_voice

```json
{
  "host": "0.0.0.0",
  "port": 8000,
  "model_path": "./models/3dspeaker.onnx",
  "sample_rate": 16000,
  "threshold": 0.35,
  "asr_mode": "offline_sense_voice",
  "provider": "rknn",
  "sense_voice_model": "./models/sense_voice.rknn",
  "tokens": "./models/tokens.txt",
  "sense_voice_use_itn": true,
  "sense_voice_language": "auto",
  "command_forward_url": "http://127.0.0.1:8089/send/message",
  "command_forward_timeout": 5,
  "vad_max_utterance_ms": 20000
}
```

### macOS + streaming_transducer

```json
{
  "host": "0.0.0.0",
  "port": 8008,
  "model_path": "./models/3dspeaker.onnx",
  "sample_rate": 16000,
  "threshold": 0.35,
  "provider": "cpu",
  "tokens": "./models/tokens.txt",
  "encoder": "./models/encoder.onnx",
  "decoder": "./models/decoder.onnx",
  "joiner": "./models/joiner.onnx",
  "command_forward_url": "http://127.0.0.1:8089/send/message",
  "command_forward_timeout": 5
}
```
