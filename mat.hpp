#ifndef CUDA_MAT_MAT_HPP
#define CUDA_MAT_MAT_HPP

#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>

#define CHECK(exp) \
if (exp != cudaSuccess) { \
    auto err = cudaGetLastError(); \
    printf("[ERROR] [%s:%d] %s(%d).\n", __FILE__, __LINE__, cudaGetErrorString(err), err); \
    exit(-1); \
}

using namespace std;


template<class T>
class Mat {
public:
    Mat(size_t r, size_t c, int device = -1) : device(device), r(r), c(c), n(r * c) {
        if (device < 0) {
            CHECK(cudaMallocHost(&cpu_data, n * sizeof(T)));
            gpu_data = nullptr;
        } else {
            cudaSetDevice(device);
            CHECK(cudaMalloc(&gpu_data, n * sizeof(float)));
            cpu_data = nullptr;
        }
    }

private:
    int device;
    T *cpu_data, *gpu_data;
    size_t r, c, n;
};


#endif //CUDA_MAT_MAT_HPP
