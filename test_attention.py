import torch
import torch.nn.functional as F

from attention import causal_attention1
from attention2 import causal_attention2
from attention3 import causal_attention3

def test_attention():
    B = 4 # 4
    H = 12 # 12
    T = 1024 # 1024
    C = 768 # 768
    q = torch.randn(size=(B, H, T, C), device="cuda", dtype=torch.float16)
    k = torch.randn(size=(B, H, T, C), device="cuda", dtype=torch.float16)
    v = torch.randn(size=(B, H, T, C), device="cuda", dtype=torch.float16)
    mask = torch.tril(torch.ones(size=(T, T), device="cuda")).view(1, 1, T, T)

    output_triton, qk_triton = causal_attention3(q, k, v)
    qk_triton = qk_triton.masked_fill(mask[:,:,:T,:T] == 0, 0.0)
    output_torch = F.softmax((q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5).masked_fill(mask[:,:,:T,:T] == 0, float('-inf')), dim=-1) @ v
    qk_torch = (q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5).masked_fill(mask[:,:,:T,:T] == 0, 0.0)

    qk_diffs = torch.abs(qk_triton - qk_torch)
    print("avg diff:", torch.mean(qk_diffs))
    print("max diff:", torch.max(qk_diffs))
    print("min diff:", torch.min(qk_diffs))

    # exit()
    if torch.allclose(output_triton, output_torch, atol=1e-2, rtol=1e-2):
        print("All ok!")
    else:
        print("Something is wrong!")
        print(output_torch[0, 0, 0])
        print(output_triton[0, 0, 0])
        
    diffs = torch.abs(output_triton - output_torch)
    print("avg diff:", torch.mean(diffs))
    print("max diff:", torch.max(diffs))
    print("min diff:", torch.min(diffs))

test_attention()