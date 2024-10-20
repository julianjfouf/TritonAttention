Tried some different ways of computing attention. This is some of the results of my efforts. Not sure how accurate these benchmarks are w.r.t actual performance because it may be the case that I did the benchmarks wrong. 

In any case, the calculation results match and the benchmarks show these kernels doing better than F.scaled_dot_product_attention on varying batch sizes, context lengths, and hidden dimension sizes.
In the benchmark I called the PyTorch layer flashattn because I originally thought that this was flash attention, however, at this point I am not certain that I am actually comparing against it.
Work is a bit unorganized but I have published it anyways for documentation purposes.
