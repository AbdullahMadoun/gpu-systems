#include <cuda_runtime.h>
#include <stdio.h>

extern "C" void launch_kernel(float *X, float *L, float *R, int B, int S, int D,
                              int D_ff) {

  float *X_d;
  float *L_d;
  float *R_d;

  int x_size = B * S * D * sizeof(float);
  int l_size = D * D_ff * sizeof(float);
  cudaMalloc((void **)&X_d, x_size);
  cudaMemcpy(X, X_d, x_size, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&L_d, l_size);
  cudaMemcpy(L, L_d, l_size, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&R_d, l_size);

  dim3 dimBlock(16, 16, 1)
}