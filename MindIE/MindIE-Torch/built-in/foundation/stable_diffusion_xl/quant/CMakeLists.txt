cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(MIE_LLM_quant_ops)

find_package(Torch REQUIRED)

set(TORCH_ROOT "/usr/local/libtorch2.0.0" CACHE STRING "")
MESSAGE("Torch root : ${TORCH_ROOT}")

add_library(quant_ops SHARED quant_ops.cpp)

target_compile_features(quant_ops PRIVATE cxx_std_17)

target_link_libraries(quant_ops PUBLIC
    c10
    torch
    torch_cpu)