import os
import torch

# 매핑할 키: old_key -> new_key
key_map = {
    'num_latent_array': 'num_latents',
    'latent_channels': 'hidden_dim'
}


def update_checkpoint(path: str):
    """
    주어진 경로의 .pt 체크포인트를 로드하여
    best_model_cfg 딕셔너리 내부의 키 이름을 변경한 뒤 다시 저장합니다.
    """
    ckpt = torch.load(path, weights_only=False)
    
    if 'best_model_cfg' in ckpt:
        cfg = ckpt['best_model_cfg']
        for old_key, new_key in key_map.items():
            if old_key in cfg:
                cfg[new_key] = cfg.pop(old_key)
        ckpt['best_model_cfg'] = cfg
    
    torch.save(ckpt, path)

for root, dirs, files in os.walk('output'):
    if 'TabPerceiver.pt' in files:
        pt_path = os.path.join(root, 'TabPerceiver.pt')
        update_checkpoint(pt_path)
        print(f"Updated keys in: {pt_path}")