Simple PoC of a fragmentationless cuda allocator.

Uses [VMM](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html) and a dry run to set up virtual addresses and virtual to physical mappings that guarantee no physical memory fragmentation.

The dry run is slow, but replay overhead is neglible.

Useful for static workloads that can be captured once and replayed many times (e.g. cuda graphs).
