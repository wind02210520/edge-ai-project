import os
import torchvision

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(BASE_DIR, 'data', 'calib')
os.makedirs(out, exist_ok=True)

# transform=None 拿原始 PIL 图（0-255），不要归一化
ds = torchvision.datasets.CIFAR10(root=os.path.join(BASE_DIR, 'data'),
                                  train=False, transform=None)

lines = []
for i in range(50):
    img, _ = ds[i]
    p = os.path.join(out, f'{i:03d}.png')
    img.save(p)
    lines.append(p)

with open(os.path.join(out, 'calib.txt'), 'w') as f:
    f.write('\n'.join(lines))

print('calib done:', len(lines), '张 ->', out)