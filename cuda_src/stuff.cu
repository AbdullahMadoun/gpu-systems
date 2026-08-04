#include <cuda_runtime.h>
#include <math.h>

__global__ void SwiGLU_kernel(float *P_in, float *P_out, int width,
                              int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {

    int gray_offset = (row * width + col);
    int rgb_offset = gray_offset * 3;

    float r = P_in[rgb_offset];
    float g = P_in[rgb_offset + 1];
    float b = P_in[rgb_offset + 2];

    P_out[gray_offset] = 0.21f * r + 0.71f * g + 0.07f * b;
  }
}

__device__ void MatrixMultiplication(float *M, float *N, float *R, int d,
                                     int dff, int s) {}

extern "C" void SwiGLU(float *x, float *up_proj, float *down_proj,
                       float *gate_proj, float *result, int B, int S, int D,
                       int Dff) {
  float *x_d;
  float *up_proj_d;
  float *down_proj_d;
  float *gate_proj_d;
  float *result_d;

  int x_size = B * S * D * sizeof(float);
  int w_size = D * Dff * sizeof(float);

  cudaMalloc((void **)&x_d, x_size);
  cudaMemcpy(x_d, x, x_size, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&up_proj_d, w_size);
  cudaMemcpy(up_proj_d, up_proj, w_size, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&down_proj_d, w_size);
  cudaMemcpy(down_proj_d, down_proj, w_size, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&gate_proj_d, w_size);
  cudaMemcpy(gate_proj_d, gate_proj, w_size, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&result_d, x_size);

  dim3 dimBlock(16, 16);
  dim3 dimGrid((D + 15) / 16, (S + 15) / 16);

  SwiGLU_kernel<<<dimGrid, dimBlock>>>(x_d, result_d, D, S);

  cudaMemcpy(result, result_d, x_size, cudaMemcpyDeviceToHost);
  cudaFree(x_d);
  cudaFree(up_proj_d);
  cudaFree(down_proj_d);
  cudaFree(gate_proj_d);
  cudaFree(result_d);
}
