# FP6 primitives

This code was initially developed for https://github.com/pytorch/ao/pull/248. However, I found a cleaner and better way to do that with Triton, so the C++/CUDA extensions were removed from the PR.

## Introduction to Floating point representation

Each floating point (fp) type (e.g. FP32, FP16, BF16) consists of 3 parts: (1) 1 sign bit, (2) E exponent bits, and (3) M mantissa bits. Some popular ones are listed below

dtype | E | M  | bit pattern
--------|----|-----|---------------
FP32 | 8  | 23 | `SEEE EEEE EMMM MMMM MMMM MMMM MMMM MMMM`
FP16 | 5  | 10 | `                    SEEE EEMM MMMM MMMM`
BF16 | 8  | 7  | `                    SEEE EEEE EMMM MMMM`
FP6 (we are here) | 3 | 2 | `                                SE EEMM`

The (normal) value of a fp number is determined by the formula (sign is omitted for brevity)

```
2^(exponent bits - exponents bias) * 1.mantissa
```
 
Note that `1.mantissa` is in binary format. Thus, `1.1b = (1 + 0.5)`, `1.01b = (1 + 0.25)`, and so on. The `1.` in front is known as the `implicit 1`

Exponent bias is around half of the max exponent bits. Its bit pattern is all 1s except the left-most (most significant) bit. Some examples

dtype | E-bias bits | E-bias value
--------|----------------|-----------------
FP32 | `0111 1111` | 127
FP16 | `   0 1111` | 15
BF16 | `0111 1111` | 127
FP6  | `      011` | 3

This is known as excess-K (or offset binary) coding, a way to code negative and positive numbers. One benefit of this is that we can compare fp numbers fast with integer math.

The above covers almost all of what you typically need to know about floating point. However, there are some special values
- +/-inf: exponent bits are all 1s (max), mantissa bits are all 0s
- NaN: exponent bits are all 1s (max), mantissa bits are not all 0s
- **subnormal numbers**: exponent bits are all 0s (min).

**Subnormal numbers** are annoying, since they use a different formula to calculate their value.

```
2^(1 - exponent bias) * 0.mantissa
```

There are 2 main differences compared to normal numbers
1. Even though exponent bits are 0s, we use value 1 in the formula
2. There is no "implicit 1" in the mantissa.

## Floating point conversion

For most cases, if it's a normal number -> normal number conversion, the logic is simple: we just need to shift the bits around to their correct position, and correct the exponent bias.

For example, doing FP32->FP16 conversion
- Step 0: FP32 bits `SEEE EEEE EMMM MMMM MMMM MMMM MMMM MMMM`
- Step 1: extract the components (using right shift and bitwise AND): `S`, `EEEE EEEE`, `MMM....`
- Step 2: truncate mantissa bits (keep the 2 left-most): `MM` (rounding logic is omitted for brevity)
- Step 3: correct exponent bias: `E = E - 127 + 15`
- Step 4: assemble them to form FP16: `SEEE EEMM MMMM MMMM`

However, throwing in +/-inf, NaN, and subnormal numbers, we need to handle them with care. i.e. checks for special values, check for subnormal numbers, use a different subroutine to handle subnormal numbers.

One solution to avoid the complexity is to do *exponent bias correction in floating point* (instead of bits). Instead of doing `E = E - 127 + 15`, we can directly do `x = x * 2 ^ (-127 + 15)`. If the input is +/-inf or Nan, the output remains +/-inf or NaN. The output can become subnormal number, which is fine, since we can just shift the bits to their correct positions (as subnormal numbers).

In Pytorch, we can implement it as follows

```python
import torch

def fp32_to_fp16(tensor: torch.Tensor):
    tensor = tensor * 2.0 ** (-127 + 15)
    bits = tensor.view(torch.int32)

    sign = ((bits >> 31) & 0x1) << 15
    exp_and_man = (bits >> 13) & 0x7FFF
    result = sign | exp_and_man
    result = result.to(torch.uint16).view(torch.float16)
    return result

x = torch.randn(1)
print(fp32_to_fp16(x).item(), x.half().item())
```

There is no rounding logic in the code above yet, so the result might differ by 1 bit. We can extend this to FP32->FP6 with ease.

For FP16/BF16->FP6, it's usually better to upcast the input to FP32 first, then do FP32->FP6. On CPU, we don't have FP16/BF16 multiplication, so it's often done by bit manipulation, which is slow. On CUDA, we have FP16/BF16 multiplication, but the CUDA kernel is often memory-bound, so it's easier to upcast the input to FP32 first anyway.
