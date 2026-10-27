#include <cuda_runtime.h>
#include <stdio.h>

const int TILE_SIZE = 32;

__global__ void naive_matmul(float *A, float *B, float *R, int M, int N,
                             int P) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  float r_value = 0.0f;

  if (row < M && col < P) {
    for (int i = 0; i < N; ++i) {
      r_value += A[row * N + i] * B[i * P + col];
    }
    R[row * P + col] = r_value;
  }
}

__global__ void tiled_matmul(float *A, float *B, float *R, int M, int N,
                             int P) {
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
  size_t A_size = (size_t)M * N * sizeof(float);
  size_t B_size = (size_t)N * P * sizeof(float);
  size_t R_size = (size_t)M * P * sizeof(float);
  cudaMalloc((void **)&A_d, A_size);
  cudaMemcpy(A_d, A, A_size, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&B_d, B_size);
  cudaMemcpy(B_d, B, B_size, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&R_d, R_size);

  dim3 dimBlock(TILE_SIZE, TILE_SIZE);
  dim3 dimGrid((P + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
  tiled_matmul<<<dimGrid, dimBlock>>>(A_d, B_d, R_d, M, N, P);

  cudaMemcpy(R, R_d, R_size, cudaMemcpyDeviceToHost);
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(R_d);
}
