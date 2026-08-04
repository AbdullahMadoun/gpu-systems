#include <cuda_runtime.h>
#include <stdio.h>

__global__ void tiled_matmul(float *A, float *B, float *R, int M, int N,
                             int P) {
  int TILE_SIZE = 32;
  __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
  __shared__ float B_tile[TILE_SIZE][TILE_SIZE];
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  float r_value = 0.0f;

  for (int i = 0; i < (N + TILE_SIZE - 1) / TILE_SIZE; ++i) {

    if (row < M && (i * TILE_SIZE + tx) < N) {
      A_tile[ty][tx] = A[row * N + (i * TILE_SIZE + tx)];
    } else {
      A_tile[ty][tx] = 0.0f;
    }
    if (col < P && (i * TILE_SIZE + ty) < N) {
      B_tile[ty][tx] = B[(i * TILE_SIZE + ty) * P + col];
    } else {
      B_tile[ty][tx] = 0.0f;
    }

    __syncthreads();

    for (int j = 0; j < TILE_SIZE; ++j) {
      r_value += A_tile[ty][j] * B_tile[j][tx];
    }
    __syncthreads();
  }

  if (row < M && col < P) {
    R[row * P + col] = r_value;
  }
}
void launch_tiled_matmul(float *A, float *B, float *R, int M, int N, int P) {
  float *A_d;
  float *B_d;
  float *R_d;
  int A_size = M * N * sizeof(float);
  int B_size = N * P * sizeof(float);
  int R_size = M * P * sizeof(float);
  cudaMalloc((void **)&A_d, A_size);
  cudaMemcpy(A_d, A, A_size, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&B_d, B_size);
  cudaMemcpy(B_d, B, B_size, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&R_d, R_size);

  int threads = 32;
  dim3 dimBlock(threads, threads);
  dim3 dimGrid((threads - 1 + P) / threads, (threads - 1 + M) / threads);
  tiled_matmul<<<dimGrid, dimBlock>>>(float *A, float *B, float *R, int M,
                                      int N, int P);

  cudaMemcpy(R_d, R, R_size, cudaMemcpyDeviceToHost);
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(R_d);
}