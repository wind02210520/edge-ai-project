import os
from rknn.api import RKNN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ONNX_MODEL = os.path.join(BASE_DIR, 'deploy', 'model.onnx')
RKNN_MODEL = os.path.join(BASE_DIR, 'deploy', 'model.rknn')

rknn = RKNN(verbose=True)

CALIB = os.path.join(BASE_DIR, 'data', 'calib', 'calib.txt')
RKNN_MODEL = os.path.join(BASE_DIR, 'deploy', 'model_int8.rknn')   # 新名字，别覆盖

rknn.config(
    target_platform='rk3588',
    mean_values=[[125.307, 122.961, 113.858]],   # 0.4914/0.4822/0.4465 × 255
    std_values=[[62.985, 62.093, 66.708]],       # 0.2470/0.2435/0.2616 × 255
)
rknn.load_onnx(model=ONNX_MODEL)
rknn.build(do_quantization=True, dataset=CALIB)   # ← 开量化
rknn.export_rknn(RKNN_MODEL)

rknn.release()
print('转换完成:', RKNN_MODEL)