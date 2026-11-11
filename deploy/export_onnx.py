import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

import torch
from train import SmallCNN

# 1. 加载模型（weights_only=True 消掉那个警告）
model = SmallCNN()
ckpt = torch.load(os.path.join(BASE_DIR, "best.pth"), weights_only=True)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"loaded best.pth, val_acc={ckpt.get('val_acc')}")

# 2. 导出 ONNX
dummy = torch.randn(1, 3, 32, 32)
onnx_path = os.path.join(BASE_DIR, "deploy", "model.onnx")
torch.onnx.export(
    model, dummy, onnx_path,
    input_names=["input"], output_names=["output"],
    opset_version=12,
)
print(f"exported -> {onnx_path}")

# 3. 用 onnxruntime 验证，和 PyTorch 输出对比
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(onnx_path)
onnx_out = sess.run(None, {"input": dummy.numpy()})[0]

with torch.no_grad():
    torch_out = model(dummy).numpy()

diff = float(np.abs(onnx_out - torch_out).max())
print(f"max diff torch vs onnx: {diff:.2e}")
assert diff < 1e-4, "ONNX 和 PyTorch 输出不一致！"
print("ONNX verification passed")