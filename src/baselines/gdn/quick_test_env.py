"""验证 GDN 环境依赖"""
import sys
sys.path.insert(0, './third_party/gdn')

import torch
import numpy as np

print("[1/3] PyTorch version:", torch.__version__)
print("      CUDA available:", torch.cuda.is_available())

try:
    from torch_geometric.nn.conv import MessagePassing
    print("[2/3] PyTorch Geometric: OK")
except Exception as e:
    print(f"[2/3] PyTorch Geometric: FAILED - {e}")
    sys.exit(1)

try:
    from models.GDN import GDN
    print("[3/3] GDN model import: OK")
except Exception as e:
    print(f"[3/3] GDN model import: FAILED - {e}")
    sys.exit(1)

# 快速前向测试
edge_index = torch.tensor([[0,1,2], [1,2,0]], dtype=torch.long)
model = GDN([edge_index], node_num=3, dim=16, input_dim=10, out_layer_num=1, out_layer_inter_dim=32, topk=2)
x = torch.randn(2, 3, 10).float()
out = model(x, edge_index)
print(f"      Input shape:  {x.shape}")
print(f"      Output shape: {out.shape}")
print("\n[OK] 环境验证通过，可以开始训练")
