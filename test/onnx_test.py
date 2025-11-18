import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

# Mean Pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # 输出: (batch, seq_len, hidden)
    input_mask_expanded = np.expand_dims(attention_mask, -1)
    
    # 平均池化（只平均真实 token 部分）
    return np.sum(token_embeddings * input_mask_expanded, axis=1) / np.clip(
        np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None
    )

tokenizer = AutoTokenizer.from_pretrained("Xenova/bge-small-zh-v1.5")

session = ort.InferenceSession("./models/bge-onnx/onnx/model.onnx")

text = "你好，我想测试中文文本嵌入"

inputs = tokenizer(text, return_tensors="np", padding=True)

# token_type_ids 缺失时手动补
if "token_type_ids" not in inputs:
    seq_len = inputs["input_ids"].shape[1]
    inputs["token_type_ids"] = np.zeros((1, seq_len), dtype=np.int64)

# ONNX 推理
outputs = session.run(
    None,
    {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "token_type_ids": inputs["token_type_ids"]
    }
)

# Step 1: Mean Pooling
sentence_emb = mean_pooling(outputs, inputs["attention_mask"])

# Step 2: L2 normalize
sentence_emb = sentence_emb / np.linalg.norm(sentence_emb, axis=1, keepdims=True)

print("Embedding shape:", sentence_emb.shape)
print(sentence_emb[0][:10])
