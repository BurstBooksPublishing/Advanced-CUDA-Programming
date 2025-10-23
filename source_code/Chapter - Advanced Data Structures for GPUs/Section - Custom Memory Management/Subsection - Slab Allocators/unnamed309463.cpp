#include <iostream>
#include <cuda_runtime.h>

#define SLAB_SIZE 1024
#define NUM_SLABS 10

__device__ void* slab_alloc(int* slab_ptrs, int* slab_usage, int size) {
    for (int i = 0; i < NUM_SLABS; i++) {
        if (slab_usage[i] + size <= SLAB_SIZE) {
            void* ptr = (void*)(slab_ptrs[i] + slab_usage[i]);
            slab_usage[i] += size;
            return ptr;
        }
    }
    return NULL; // No available slab
}

__global__ void kernel(int* slab_ptrs, int* slab_usage) {
    int* data = (int*)slab_alloc(slab_ptrs, slab_usage, sizeof(int) * 256);
    if (data) {
        for (int i = 0; i < 256; i++) {
            data[i] = i; // Initialize data
        }
    }
}

int main() {
    int* d_slab_ptrs, *d_slab_usage;
    cudaMalloc(&d_slab_ptrs, NUM_SLABS * sizeof(int));
    cudaMalloc(&d_slab_usage, NUM_SLABS * sizeof(int));

    int h_slab_ptrs[NUM_SLABS], h_slab_usage[NUM_SLABS] = {0};

    for (int i = 0; i < NUM_SLABS; i++) {
        cudaMalloc(&h_slab_ptrs[i], SLAB_SIZE);
    }

    cudaMemcpy(d_slab_ptrs, h_slab_ptrs, NUM_SLABS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_slab_usage, h_slab_usage, NUM_SLABS * sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(d_slab_ptrs, d_slab_usage);
    cudaDeviceSynchronize();

    for (int i = 0; i < NUM_SLABS; i++) {
        cudaFree(h_slab_ptrs[i]);
    }

    cudaFree(d_slab_ptrs);
    cudaFree(d_slab_usage);
    
    return 0;
}