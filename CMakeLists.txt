cmake_minimum_required(VERSION 3.17)
project(cuda_mat CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

add_executable(cuda_mat main.cpp mat.hpp)
target_link_libraries(cuda_mat cudart)