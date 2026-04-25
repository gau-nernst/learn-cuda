import triton
import triton.language as tl


@triton.jit
def _rms_norm(x_ptr, norm_ptr, out_ptr, dim: tl.constexpr, BLOCK_SIZE: tl.constexpr = 1024):
    if dim == BLOCK_SIZE:
        # 1-pass
        offs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs).to(tl.float32)
        norm = tl.load(norm_ptr + offs).to(tl.float32)

        inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) * (1 / dim) + 1e-6)
        x = x * inv_rms * norm
        tl.store(out_ptr + offs, x)

    else:
        # 2-pass
        sum_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for k in range(tl.cdiv(dim, BLOCK_SIZE)):
            offs = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < dim
            x = tl.load(x_ptr + offs, mask, other=0.0).to(tl.float32)
            sum_sq += x * x

        inv_rms = tl.rsqrt(tl.sum(sum_sq, axis=0) * (1 / dim) + 1e-6)

        for k in range(tl.cdiv(dim, BLOCK_SIZE)):
            offs = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < dim
            x = tl.load(x_ptr + offs, mask, other=0.0).to(tl.float32)
            norm = tl.load(norm_ptr + offs, mask, other=0.0).to(tl.float32)

            x = x * inv_rms * norm
            tl.store(out_ptr + offs, x, mask)


@triton.jit
def _spin_wait(ptr, count):
    # triton will rewrite atomic add 0 to ld.acquire
    while tl.atomic_add(ptr, 0, sem="acquire", scope="gpu") != count:
        pass


@triton.jit
def _grid_sync(ptr, count):
    tl.atomic_add(ptr, 1, sem="release", scope="gpu")  # signal done
    _spin_wait(ptr, count)
