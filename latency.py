import torch
import time
from model import SAM2
from sam2.build_sam import build_sam2
import hydra
from tqdm import tqdm
model_cfg = "sam2_hiera_mini.yaml"
# hydra is initialized on import of sam2, which sets the search path which can't be modified
# so we need to clear the hydra instance
hydra.core.global_hydra.GlobalHydra.instance().clear()
# reinit hydra with a new search path for configs
hydra.initialize_config_module('sam2_configs', version_base='1.2')

pretrain = '/home/Qing_Xu/hd1/xq/IJCAI2025/medsam2/outputs/small_ultra_35.pth'

model = SAM2(build_sam2(model_cfg, pretrain, mode='eval'))

model.eval()
model = model.cuda()
input_tensor = torch.randn(1, 3, 1024, 1024).cuda()  # 模拟输入
mask_tensor = torch.randn(1, 1, 1024, 1024).cuda()  # 模拟输入

# Warm-up
for _ in tqdm(range(10)):
    with torch.no_grad():
        _ = model(input_tensor, gt=None)

# Measure latency
total_time = 0
n_runs = 100
for _ in tqdm(range(n_runs)):
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(input_tensor, gt=None)
    total_time += time.perf_counter() - start

avg_latency = total_time / n_runs * 1000  # 转为毫秒
print(f"Average latency: {avg_latency:.2f} ms")