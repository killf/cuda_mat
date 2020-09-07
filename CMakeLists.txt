cmake_minimum_required(VERSION 3.17)
project(cuda_mat)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDAToolkit)

add_executable(cuda_mat main.cpp mat.hpp)
target_link_libraries(cuda_mat CUDA::cudart)

# FindCUDAToolkit: https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html