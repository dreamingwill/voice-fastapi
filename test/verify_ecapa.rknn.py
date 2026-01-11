import numpy as np
from rknnlite.api import RKNNLite

MODEL_PATH = "./models/ecapa_tdnn_fp16.rknn"

rknn = RKNNLite()
assert rknn.load_rknn(MODEL_PATH) == 0, "load_rknn failed"

ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
assert ret == 0, "init_runtime failed"

# 构造假输入：80-d fbank, 300 frames
x = np.random.rand(1, 80, 300).astype(np.float32)

out = rknn.inference(inputs=[x])[0]

print("Raw output shape:", out.shape)
emb = np.squeeze(out)
print("Embedding shape:", emb.shape)
print("Embedding sample (first 5):", emb[:5])
