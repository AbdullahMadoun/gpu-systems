import torch
from torch.utils.cpp_extension import load_inline

_GRAYSCALE_EXT = None
_RGB_BLUR_EXT = None

def get_grayscale_extension():
    """Lazily compiles and returns the PyTorch C++/CUDA inline extension for RGB to Grayscale."""
    global _GRAYSCALE_EXT
    if _GRAYSCALE_EXT is not None:
        return _GRAYSCALE_EXT

    cuda_source = """
    #include <cuda_runtime.h>
    #include <math.h>

    __global__ void to_grayscale_kernel(float* P_in, float* P_out, int width, int height) {
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

    extern "C" void launch_to_grayscale(float* P_in, float* P_out, int width, int height) {
        float *P_in_d, *P_out_d;
        size_t input_size_bytes = width * height * 3 * sizeof(float);
        size_t output_size_bytes = width * height * sizeof(float);

        cudaMalloc((void**) &P_in_d, input_size_bytes);
        cudaMemcpy(P_in_d, P_in, input_size_bytes, cudaMemcpyHostToDevice);
        cudaMalloc((void**) &P_out_d, output_size_bytes);

        dim3 dimBlock(32, 32);
        dim3 dimGrid((width + 31) / 32, (height + 31) / 32);

        to_grayscale_kernel<<<dimGrid, dimBlock>>>(P_in_d, P_out_d, width, height);

        cudaMemcpy(P_out, P_out_d, output_size_bytes, cudaMemcpyDeviceToHost);
        cudaFree(P_in_d);
        cudaFree(P_out_d);
    }
    """

    cpp_source = """
    #include <torch/extension.h>
    #include <vector>

    extern "C" void launch_to_grayscale(float* P_in, float* P_out, int width, int height);

    torch::Tensor to_grayscale(torch::Tensor P_in) {
        torch::Tensor cpu_tensor = P_in.to(torch::kCPU).contiguous();
        float* P_in_float = cpu_tensor.data_ptr<float>();
        int height = cpu_tensor.size(0);
        int width = cpu_tensor.size(1);
        float* P_out = new float[height * width];

        launch_to_grayscale(P_in_float, P_out, width, height);

        std::vector<int64_t> shape = {height, width};
        torch::Tensor output_tensor = torch::from_blob(
            P_out, shape, torch::TensorOptions().dtype(torch::kFloat32)
        ).clone();

        delete[] P_out;
        return output_tensor;
    }
    """

    _GRAYSCALE_EXT = load_inline(
        name="custom_grayscale_ext",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=["to_grayscale"],
        verbose=False,
    )
    return _GRAYSCALE_EXT

def get_rgb_blur_extension():
    """Lazily compiles and returns the PyTorch C++/CUDA inline extension for 3-channel RGB Box Blur."""
    global _RGB_BLUR_EXT
    if _RGB_BLUR_EXT is not None:
        return _RGB_BLUR_EXT

    cuda_source = """
    #include <cuda_runtime.h>
    #include <math.h>

    __global__ void box_blur_rgb_kernel(float* P_in, float* P_out, int width, int height) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int BlurSize = 3;

        if (col < width && row < height) {
            float r_val = 0.0f, g_val = 0.0f, b_val = 0.0f;
            int pixels = 0;

            for (int blurRow = -BlurSize; blurRow <= BlurSize; ++blurRow) {
                for (int blurCol = -BlurSize; blurCol <= BlurSize; ++blurCol) {
                    int currRow = row + blurRow;
                    int currCol = col + blurCol;
                    if (currRow >= 0 && currRow < height && currCol >= 0 && currCol < width) {
                        int pixel_idx = (currRow * width + currCol) * 3;
                        r_val += P_in[pixel_idx + 0];
                        g_val += P_in[pixel_idx + 1];
                        b_val += P_in[pixel_idx + 2];
                        ++pixels;
                    }
                }
            }

            int out_idx = (row * width + col) * 3;
            P_out[out_idx + 0] = r_val / pixels;
            P_out[out_idx + 1] = g_val / pixels;
            P_out[out_idx + 2] = b_val / pixels;
        }
    }

    extern "C" void launch_rgb_blur(float* P_in, float* P_out, int width, int height) {
        float *P_in_d, *P_out_d;
        size_t size_bytes = width * height * 3 * sizeof(float);

        cudaMalloc((void**) &P_in_d, size_bytes);
        cudaMemcpy(P_in_d, P_in, size_bytes, cudaMemcpyHostToDevice);
        cudaMalloc((void**) &P_out_d, size_bytes);

        dim3 dimBlock(32, 32);
        dim3 dimGrid((width + 31) / 32, (height + 31) / 32);

        box_blur_rgb_kernel<<<dimGrid, dimBlock>>>(P_in_d, P_out_d, width, height);

        cudaMemcpy(P_out, P_out_d, size_bytes, cudaMemcpyDeviceToHost);
        cudaFree(P_in_d);
        cudaFree(P_out_d);
    }
    """

    cpp_source = """
    #include <torch/extension.h>
    #include <vector>

    extern "C" void launch_rgb_blur(float* P_in, float* P_out, int width, int height);

    torch::Tensor rgb_blur(torch::Tensor P_in) {
        torch::Tensor cpu_tensor = P_in.to(torch::kCPU).contiguous();
        float* P_in_float = cpu_tensor.data_ptr<float>();
        int height = cpu_tensor.size(0);
        int width = cpu_tensor.size(1);
        float* P_out = new float[height * width * 3];

        launch_rgb_blur(P_in_float, P_out, width, height);

        std::vector<int64_t> shape = {height, width, 3};
        torch::Tensor output_tensor = torch::from_blob(
            P_out, shape, torch::TensorOptions().dtype(torch::kFloat32)
        ).clone();

        delete[] P_out;
        return output_tensor;
    }
    """

    _RGB_BLUR_EXT = load_inline(
        name="custom_rgb_blur_ext",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=["rgb_blur"],
        verbose=False,
    )
    return _RGB_BLUR_EXT
