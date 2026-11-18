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
CKPT       = os.path.join(BASE_DIR, 'best.pth')
CALIB      = os.path.join(BASE_DIR, 'data', 'calib', 'calib.txt')

N = 500
MEAN = [125.307, 122.961, 113.858]
STD  = [62.985, 62.093, 66.708]

ds = torchvision.datasets.CIFAR10(root=os.path.join(BASE_DIR, 'data'),
                                  train=False, transform=None)

# ---------- PyTorch FP32 ----------
norm = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
tf = T.Compose([T.ToTensor(), T.Normalize(*norm)])

model = SmallCNN()
model.load_state_dict(torch.load(CKPT, weights_only=True)['model'])
model.eval()

correct_t = 0
for i in range(N):
    img, label = ds[i]
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).argmax(1).item()
    correct_t += int(pred == label)
print(f'torch FP32 acc: {correct_t/N:.4f}')

# ---------- RKNN INT8 模拟器 ----------
rknn = RKNN(verbose=False)
rknn.config(target_platform='rk3588',
            mean_values=[MEAN], std_values=[STD])
rknn.load_onnx(model=ONNX_MODEL)
rknn.build(do_quantization=True, dataset=CALIB)
rknn.init_runtime()

correct_r = 0
for i in range(N):
    img, label = ds[i]
    arr = np.array(img).astype(np.float32)   # HWC, 0-255 原始图
    out = rknn.inference(inputs=[arr])[0]
    pred = int(np.argmax(out))
    correct_r += int(pred == label)
rknn.release()

print(f'rknn INT8 acc: {correct_r/N:.4f}')
print(f'drop         : {(correct_t - correct_r)/N*100:.2f}%')