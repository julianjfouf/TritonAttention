import torch
import torch.nn as nn
import torch.nn.functional as F

import triton

from attention import causal_attention1
from attention2 import causal_attention2
from attention3 import causal_attention3

configs = []

for measure in ["ms", ]:
    configs.append(
        triton.testing.Benchmark(
        x_names=["C"],
        x_vals=[2**i for i in range(8, 13)],
        x_log=True,
        line_arg="provider",
        line_names=["kernel1", "kernel2", "kernel3", "flashattn"],
        line_vals=["kernel1", "kernel2", "kernel3", "flashattn"],
        styles=[("blue", "-"), ("red", "-"), ("purple", "-"), ("yellow", "-")],
        ylabel=measure,
        plot_name="attention-" + measure,
        args={"measure": measure},
        )
    )

@triton.testing.perf_report(configs)
def benchmark(measure, C, provider):
    B = 4
    H = 12
    T = 1024
    q = torch.randn(size=(B, H, T, C), device="cuda", dtype=torch.float16)
    k = torch.randn(size=(B, H, T, C), device="cuda", dtype=torch.float16)
    v = torch.randn(size=(B, H, T, C), device="cuda", dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "kernel1":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: causal_attention1(q, k, v), quantiles=quantiles)

    if provider == "kernel2":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: causal_attention2(q, k, v), quantiles=quantiles)

    if provider == "kernel3":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: causal_attention3(q, k, v), quantiles=quantiles)
        
    if provider == "flashattn":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), quantiles=quantiles)
    
    return ms, min_ms, max_ms

benchmark.run(save_path="benchmark_results2/C")