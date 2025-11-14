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

## 五、查看 API

服务启动后访问 `http://<host>:<port>/docs` 查看 Swagger 文档（默认 `http://127.0.0.1:8000/docs`）。
