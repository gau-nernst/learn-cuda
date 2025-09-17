# Peer-to-Peer (P2P) Memory Access

Resources:
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/#peer-to-peer-memory-access
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html

Learnings
- When P2P is enabled, one GPU can access data of another GPU via a device pointer directly. This means that we can read/write data residing in a remote GPU directly in a kernel. And things like Triton will work too.
- Two key concepts: Memory Coherence and Memory Ordering.
- First check `cudaDeviceCanAccessPeer()`, then `cudaDeviceEnablePeerAccess()` so that remote GPUs can access data of local GPU. Each GPU must do this for all other GPUs.
- In multi-GPU setup, typically we launch 1 process for each GPU. Addresses are only valid within the same process. To share the memory pointers (and CUDA events) across process, we must use **IPC API**.
- `cudaIpcGetMemHandle()` to get IPC handle of the current device for an address, and pass it to other devices via other means (e.g. shared memory, MPI, or simply PyTorch distributed in a PyTorch environment). On other ranks, `cudaIpcOpenMemHandle()` to retrieve the remote address that is valid in the current process. When done, `cudaIpcCloseMemHandle()`.
- To avoid multiple allocations and repeated IPC handles sharing, we can do **Arena allocation**: allocate a buffer large enough for our problem size, then share IPC handles of this allocation. Once shared, we can slice from this buffer and don't need to reshare IPC handles. We probably need to be careful with the offsets from the base allocation - depending on our usage, but it's likely a good practice to do the same slicing / object allocation across GPUs, so the same object has the same offset from the base address. This is basically a **symmetric heap**.
