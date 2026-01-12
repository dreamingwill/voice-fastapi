from rknn.api import RKNN
import numpy as np
import time

MODEL = "eres2net_large_T400.rknn"
FBANK = "fbank_000.npy"

rknn = RKNN()

print(">>> Load RKNN")
rknn.load_rknn(MODEL)

print(">>> Init runtime (NPU)")
rknn.init_runtime(core_mask=RKNN.NPU_CORE_0_1_2)

x = np.load(FBANK)  # shape: (1, 400, 80)

# 预热一次
rknn.inference(inputs=[x])

# 正式计时
t0 = time.time()
out = rknn.inference(inputs=[x])
t1 = time.time()

print("Output shape:", out[0].shape)
print("Latency:", (t1 - t0) * 1000, "ms")

rknn.release()
