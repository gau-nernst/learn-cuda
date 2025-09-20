from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

WORLD_SIZE = 2


def load_module():
    CURRENT_DIR = Path(__file__).parent
    load(
        name="p2p_module",
        sources=[str(CURRENT_DIR / "p2p.cu")],
        is_python_module=False,
    )
    return torch.ops.p2p_module


def distributed_main(rank: int):
    import os

    # use gloo to exchange IPC handles
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "2025"
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)

    def _print(*args):
        print(f"{rank=}:", *args, flush=True)

    try:
        # let pytorch initialize CUDA runtime
        torch.cuda.set_device(rank)

        ops = load_module()  # this will load pre-compiled module
        ops.enable_p2p(rank, WORLD_SIZE)

        heap = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
        ipc_handle = ops.get_ipc_handle(heap)

        # all-gather IPC handles
        all_ipc_handles = ipc_handle.new_empty(WORLD_SIZE, *ipc_handle.shape)
        dist.all_gather_into_tensor(all_ipc_handles.view(-1), ipc_handle)

        # convert to memory addresses
        heap_bases = [
            heap.data_ptr() if i == rank else ops.open_ipc_handle(all_ipc_handles[i]) for i in range(WORLD_SIZE)
        ]
        heap_bases = torch.tensor(heap_bases, dtype=torch.int64, device="cuda")

        # check write
        sym_data = heap.view(torch.int32)[:WORLD_SIZE]
        sym_data.fill_(-1)  # fill with -1, so we know if something is not written to
        torch.cuda.synchronize()  # make sure writes are flushed to global memory

        dist.barrier()  # previous write finishes
        ops.check_p2p_write(sym_data, heap_bases, rank, WORLD_SIZE)
        torch.cuda.synchronize()  # flush

        dist.barrier()  # previous write finishes
        _print(sym_data.tolist())  # should be [0, 1] on all ranks

        # check read
        sym_data.fill_(rank)
        local_data = torch.full((WORLD_SIZE,), -2, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()

        dist.barrier()
        ops.check_p2p_read(local_data, sym_data, heap_bases, rank, WORLD_SIZE)
        torch.cuda.synchronize()

        dist.barrier()
        _print(local_data.tolist())  # should be [0, 1] on all ranks

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn")  # fork is not compatible with CUDA
    load_module()  # compile C++ extension

    procs = []
    for i in range(WORLD_SIZE):
        p = mp.Process(target=distributed_main, args=(i,))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
