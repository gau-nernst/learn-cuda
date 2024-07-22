__device__ void adam_device(float *p,
                            const float *grad,
                            float step,
                            float *exp_avg,
                            float *exp_avg_sq,
                            float lr,
                            float beta1,
                            float beta2,
                            float eps,
                            float weight_decay,
                            float bias_correction1,
                            float bias_correction2) {
  float p_val = p[idx];
  float grad_val = grad[idx];
  float exp_avg_val = exp_avg[idx];
  float exp_avg_sq_val = exp_avg_sq[idx];

  grad_val += weight_decay * p_val;
  exp_avg_val = beta1 * exp_avg_val + (1.0f - beta1) * grad_val;
  exp_avg_sq_val = beta2 * exp_avg_sq_val + (1.0f - beta2) * grad_val * grad_val;

  p_val -= lr * exp_avg_val * bias_correction1 / (sqrt(exp_avg_sq_val * bias_correction2) + eps);

  p[idx] = p_val;
  exp_avg[idx] = exp_avg_val;
  exp_avg_sq[idx] = exp_avg_sq_val;
}

// https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
__global__ void adam_single(float *p,
                            const float *grad,
                            float step,
                            float *exp_avg,
                            float *exp_avg_sq,
                            int N,
                            float lr,
                            float beta1,
                            float beta2,
                            float eps,
                            float weight_decay,
                            int TILE_SIZE) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int tile_id = blockIdx.x;

  const int offset = tile_id * TILE_SIZE;
  p += offset;
  grad += offset;
  exp_avg += offset;
  exp_avg_sq += offset;

  const float bias_correction1 = 1.0f / (1.0f - pow(beta1, step));
  const float bias_correction2 = 1.0f / (1.0f - pow(beta2, step));

  for (int idx = tid; idx < TILE_SIZE; idx += BLOCK_SIZE)
    if (offset + idx < N)
      adam_device(p + idx,
                  grad + idx,
                  step,
                  exp_avg + idx,
                  exp_avg_sq + idx,
                  lr,
                  beta1,
                  beta2,
                  eps,
                  weight_decay,
                  bias_correction1,
                  bias_correction2);
}

__global__ void adam_vec4(float *p,
                          const float *grad,
                          float step,
                          float *exp_avg,
                          float *exp_avg_sq,
                          int N,
                          float lr,
                          float beta1,
                          float beta2,
                          float eps,
                          float weight_decay,
                          int TILE_SIZE) {
  const int tid = threadIdx.x;
  const int BLOCK_SIZE = blockDim.x;
  const int tile_id = blockIdx.x;

  const int offset = tile_id * TILE_SIZE;
  p += offset;
  grad += offset;
  exp_avg += offset;
  exp_avg_sq += offset;

  const float bias_correction1 = 1.0f / (1.0f - pow(beta1, step));
  const float bias_correction2 = 1.0f / (1.0f - pow(beta2, step));

  // NOTE: assume N % 4 == 0
  for (int idx = tid * 4; idx < TILE_SIZE; idx += BLOCK_SIZE * 4)
    if (offset + idx < N) {
      float4 p_vec4 = reinterpret_cast<float4 *>(&p[idx])[0];
      float4 grad_vec4 = reinterpret_cast<const float4 *>(&grad[idx])[0];
      float4 exp_avg_vec4 = reinterpret_cast<float4 *>(&exp_avg[idx])[0];
      float4 exp_avg_sq_vec4 = reinterpret_cast<float4 *>(&exp_avg_sq[idx])[0];

      adam_device(&p_vec4.x, &grad_vec4.x, step, &exp_avg_vec4.x, &exp_avg_sq.x, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);
      adam_device(&p_vec4.y, &grad_vec4.y, step, &exp_avg_vec4.y, &exp_avg_sq.y, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);
      adam_device(&p_vec4.z, &grad_vec4.z, step, &exp_avg_vec4.z, &exp_avg_sq.z, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);
      adam_device(&p_vec4.w, &grad_vec4.w, step, &exp_avg_vec4.w, &exp_avg_sq.w, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);

      reinterpret_cast<float4 *>(&p[idx])[0] = p_vec4;
      reinterpret_cast<float4 *>(&exp_avg[idx])[0] = exp_avg_vec4;
      reinterpret_cast<float4 *>(&exp_avg_sq[idx])[0] = exp_avg_sq_vec4;
    }
}
