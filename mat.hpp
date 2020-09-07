#ifndef CUDA_MAT_MAT_HPP
#define CUDA_MAT_MAT_HPP

#include <iostream>
#include <exception>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

#define CHECK(exp) \
if (exp != cudaSuccess) { \
    auto err = cudaGetLastError(); \
    printf("[ERROR] [%s:%d] %s(%d).\n", __FILE__, __LINE__, cudaGetErrorString(err), err); \
    exit(-1); \
}

template<class T= float>
class Mat;

template<class T>
void swap(Mat<T> &a, Mat<T> &b) {
  std::swap(a.row, b.row);
  std::swap(a.col, b.col);
  std::swap(a.size, b.size);
  std::swap(a.cpu_data, b.cpu_data);
  std::swap(a.gpu_data, b.gpu_data);
}

template<class T>
ostream &operator<<(ostream &stream, const Mat<T> &mat) {
  if (mat.cpu_data == nullptr) {
    return stream;
  }

  T *p = mat.cpu_data;
  for (size_t r = 0; r < mat.row; r++) {
    for (size_t c = 0; c < mat.col; c++) {
      stream << *p++ << " ";
    }
    stream << "\n";
  }

  return stream;
}

template<class T>
class Mat {
  template<class K>
  friend void swap(Mat<K> &a, Mat<K> &b);

  template<class K>
  friend ostream &operator<<(ostream &stream, const Mat<K> &mat);

public:
  Mat() : row(0), col(0), size(0), cpu_data(nullptr), gpu_data(nullptr) {};

  Mat(size_t r, size_t c, bool use_cuda = false) : row(r), col(c), size(r * c * sizeof(T)) {
    if (!use_cuda) {
      CHECK(cudaMallocHost(&cpu_data, size));
      gpu_data = nullptr;
    } else {
      cpu_data = nullptr;
      CHECK(cudaMalloc(&gpu_data, size));
    }
  }

  Mat(const initializer_list<T> &init_data) : Mat(1, init_data.size()) {
    T *q = cpu_data;
    for (auto p = init_data.begin(); p != init_data.end(); p++, q++) {
      *q = *p;
    }
  }

  Mat(const initializer_list<initializer_list<T>> &init_data) : Mat(init_data.size(), init_data.begin()->size()) {
    T *q = cpu_data;
    for (auto p1 = init_data.begin(); p1 != init_data.end(); p1++) {
      if (p1->size() != col) {
        stringstream ss;
        ss << "The expected col is " << col << ", but the actual value is " << p1->size();
        throw invalid_argument(ss.str().c_str());
      }

      for (auto p2 = p1->begin(); p2 != p1->end(); p2++, q++) {
        *q = *p2;
      }
    }
  }

  Mat(const Mat &rhs) noexcept : row(rhs.row), col(rhs.col), size(rhs.size) {
    if (rhs.cpu_data) {
      CHECK(cudaMallocHost(&cpu_data, size));
      CHECK(cudaMemcpy(cpu_data, rhs.cpu_data, size, cudaMemcpyHostToHost));
    } else cpu_data = nullptr;

    if (rhs.gpu_data) {
      CHECK(cudaMalloc(&gpu_data, size));
      CHECK(cudaMemcpy(gpu_data, rhs.gpu_data, size, cudaMemcpyDeviceToDevice));
    } else gpu_data = nullptr;
  }

  Mat(Mat &&rhs) noexcept : row(rhs.row), col(rhs.col), size(rhs.size), cpu_data(rhs.cpu_data), gpu_data(rhs.gpu_data) {
    rhs.col = rhs.row = rhs.row = 0;
    rhs.gpu_data = nullptr;
    rhs.cpu_data = nullptr;
  }

  Mat &operator=(const Mat &rhs) {
    Mat tmp(rhs);
    swap(tmp, *this);
    return *this;
  }

  ~Mat() {
    if (cpu_data)cudaFreeHost(cpu_data);
    if (gpu_data)cudaFree(gpu_data);
  }

  Mat &cuda() {
    if (!gpu_data && cpu_data) {
      CHECK(cudaMalloc(&gpu_data, size));
      CHECK(cudaMemcpy(gpu_data, cpu_data, size, cudaMemcpyHostToHost));
    }
    return *this;
  }

  Mat &cpu() {
    if (!cpu_data && gpu_data) {
      CHECK(cudaMallocHost(&cpu_data, size));
      CHECK(cudaMemcpy(gpu_data, cpu_data, size, cudaMemcpyDeviceToHost));
    }
    return *this;
  }

private:
  T *cpu_data, *gpu_data;
  size_t row, col, size;
};

#endif //CUDA_MAT_MAT_HPP
