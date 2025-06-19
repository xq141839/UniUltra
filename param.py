from thop import profile
import torch
import hydra
from model import SAM2
from sam2.build_sam import build_sam2


sam_pretrain='/home/Qing_Xu/pretrain/sam2_hiera_large.pt'

model_cfg = "sam2_hiera_mini.yaml"
# hydra is initialized on import of sam2, which sets the search path which can't be modified
# so we need to clear the hydra instance
hydra.core.global_hydra.GlobalHydra.instance().clear()
# reinit hydra with a new search path for configs
hydra.initialize_config_module('sam2_configs', version_base='1.2')

model = SAM2(build_sam2(model_cfg, sam_pretrain)).cuda()


randn_input = torch.randn(1, 3, 1024, 1024).cuda()
# randn_mask = torch.randn(1, 1, 1024, 1024)
with torch.autocast(device_type="cuda"): 
    flops, params = profile(model, inputs=(randn_input, ))
print('FLOPs = ' + str(flops/1000**3) + 'G')

total_num = sum(p.numel() for p in model.model.image_encoder.parameters())
print('Params = ' + str(round(total_num/1000**2,2)) + 'M')

