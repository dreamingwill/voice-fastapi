# voice-fastapi 语言识别系统FastAPI后端

配置相关环境

```shell
pip install -r requirements.txt
```



数据库与模型路径：

1. ./models/{模型类型}/... 存放语音识别模型路径
2. ./models/ 存放声纹识别路径
3. ./database/voiceprints.db sqlite数据库



启动命令

```shell
bash launch.sh
```

具体命令可以查看launch.sh文件内的命令行参数

```
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



查看API文档访问http://host:port/docs可以查看具体的API文档

