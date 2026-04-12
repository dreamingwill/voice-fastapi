# voice-fastapi 语言识别系统 FastAPI 后端

## 一、获取代码

```bash
git clone <git-repo-url>
cd voice-fastapi
```

> 将 `<git-repo-url>` 替换为实际的 Git 仓库地址。

## 二、准备运行环境

推荐使用 conda 创建虚拟环境：

```bash
conda create -n voice-fastapi python=3.10 -y
conda activate voice-fastapi
pip install -r requirements.txt
```

## 三、准备模型与数据库

1. **模型文件**：从本地拖入/拷贝到 `./models/`，确保大小写与路径完全一致：

   - `./models/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx`
   - `./models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/`（目录）

   如果拿到 `models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2` 压缩包，可执行：
   ```bash
   tar -xjf models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2 -C models
   ```

2. **数据库目录**：项目需要 `./database` 目录，如果不存在请手动创建：
   ```bash
   mkdir -p database
   ```
   运行时会在其中生成/使用 `voiceprints.db`。

## 四、启动服务

### 本地手动启动

```bash
bash launch.sh
```

`launch.sh` 内部会调用 `python main.py` 并传入默认参数，若需自定义可查看脚本或直接运行：

```bash
python main.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model_path ./models/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx \
  --sample_rate 16000 \
  --threshold 0.6 \
  --tokens ./models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
  --encoder ./models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx \
  --decoder ./models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
  --joiner ./models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx
```

### Linux systemd 部署

当前线上 Linux 部署是通过 `systemd -> start.sh` 启动服务。

在这套部署方式下，指令转发地址默认以 [start.sh](/Users/zrh/Projects/speak/voice-fastapi/start.sh:4) 中的配置为准：

- `COMMAND_FORWARD_URL`
- `COMMAND_FORWARD_TIMEOUT`

如果只是你在普通终端里手动执行：

```bash
export COMMAND_FORWARD_URL=http://xxx
```

这不会影响 `systemctl` 管理的服务进程。因为 `systemd` 默认不会继承你当前 shell 的环境变量。

所以按当前仓库的部署约定：

- 修改 Linux 服务使用的转发地址，优先直接修改 `start.sh`
- 修改后执行 `sudo systemctl restart <service-name>` 让服务生效

如果后面希望把地址配置从脚本中移出去，再考虑改成 `systemd` 的 `Environment=` 或 `EnvironmentFile=`

## 五、查看 API

服务启动后访问 `http://<host>:<port>/docs` 查看 Swagger 文档（默认 `http://127.0.0.1:8000/docs`）。
