import os, sys
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))
from train import SmallCNN

from rknn.api import RKNN

ONNX_MODEL = os.path.join(BASE_DIR, 'deploy', 'model.onnx')
CKPT = os.path.join(BASE_DIR, 'best.pth')

# 1. PyTorch 模型
model = SmallCNN()
model.load_state_dict(torch.load(CKPT, weights_only=True)['model'])
model.eval()

# 2. 一张真实 CIFAR 测试图（已归一化）
norm = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
tf = T.Compose([T.ToTensor(), T.Normalize(*norm)])
test = torchvision.datasets.CIFAR10(root=os.path.join(BASE_DIR, 'data'),
                                    train=False, transform=tf)
x, y = test[0]
xin = x.unsqueeze(0).numpy().astype(np.float32)

with torch.no_grad():
    t_out = model(torch.from_numpy(xin)).numpy()

# 3. RKNN 模拟器：必须 load_onnx + build
rknn = RKNN(verbose=False)
rknn.config(target_platform='rk3588')
rknn.load_onnx(model=ONNX_MODEL)
rknn.build(do_quantization=False)
rknn.init_runtime()          # 不传 target = 模拟器
xin_nhwc = np.transpose(xin, (0, 2, 3, 1)).astype(np.float32)   # NCHW -> NHWC
r_out = rknn.inference(inputs=[xin_nhwc])[0]
rknn.release()

print('torch out :', t_out)
print('rknn  out :', r_out)
print('max diff  :', np.abs(t_out - r_out).max())
print('pred torch:', t_out.argmax(1), '| pred rknn:', r_out.argmax(1), '| label:', y)