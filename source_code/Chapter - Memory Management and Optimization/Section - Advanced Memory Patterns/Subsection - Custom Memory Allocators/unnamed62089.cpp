#include <iostream>
#include <cuda_runtime.h>

// Custom memory allocator class
class CustomAllocator {
public:
    CustomAllocator(size_t poolSize) : poolSize(poolSize) {
        cudaMalloc(&devicePool, poolSize); // Allocate memory pool on device
    }

    ~CustomAllocator() {
        cudaFree(devicePool); // Free memory pool on destruction
    }

    void* allocate(size_t size) {
        if (currentOffset + size > poolSize) {
            throw std::bad_alloc(); // Check if enough space is available
        }
        void* ptr = static_cast<char*>(devicePool) + currentOffset;
        currentOffset += size; // Update offset
        return ptr;
    }

    void reset() {
        currentOffset = 0; // Reset allocator to reuse memory pool
    }

private:
    void* devicePool = nullptr;
    size_t poolSize;
    size_t currentOffset = 0;
};

__global__ void kernel(int* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = idx; // Simple kernel to initialize array
    }
}

int main() {
    const int N = 1024;
    CustomAllocator allocator(N * sizeof(int)); // Create custom allocator

    int* d_data = static_cast<int*>(allocator.allocate(N * sizeof(int)));

    kernel<<<(N + 255) / 256, 256>>>(d_data, N); // Launch kernel

    int h_data[N];
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        std::cout << h_data[i] << " "; // Print results
    }

    allocator.reset(); // Reset allocator for reuse
    return 0;
}