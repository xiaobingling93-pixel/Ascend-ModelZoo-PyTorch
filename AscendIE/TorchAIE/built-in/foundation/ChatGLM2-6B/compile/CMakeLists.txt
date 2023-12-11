PROJECT(SAMPLE)
CMAKE_MINIMUM_REQUIRED(VERSION 3.13)

add_compile_options(-fPIE -fstack-protector-all -fPIC -w -Wfatal-errors -O2)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

IF(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ARCH aarch64-linux)
ELSEIF(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ARCH x86_64-linux)
ENDIF()

# Configurable parameters
set(ASCEND_HOME "/usr/local/Ascend")
MESSAGE("Ascend home: ${ASCEND_HOME}")

set(ASCEND_TORCH_AIE "${TORCH_AIE_PATH}")
MESSAGE("Ascend torch_aie root: ${ASCEND_TORCH_AIE}")

set(TORCH_ROOT "${TORCH_ROOT_PATH}")
MESSAGE("Torch root : ${TORCH_ROOT}")

set(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

add_executable(sample chatglm2_test.cpp)
SET(EXECUTABLE_OUTPUT_PATH ${ROOT_DIR})

target_include_directories(sample PUBLIC
    ${TORCH_ROOT}/include/
    ${TORCH_ROOT}/include/torch/csrc/api/include
    ${ASCEND_TORCH_AIE}/include
)

target_link_directories(sample PUBLIC
    ${TORCH_ROOT}/lib
    ${TORCH_ROOT}/../torch.libs
    ${ASCEND_TORCH_AIE}/lib
)

target_link_libraries (sample PUBLIC
    torch_aie 
    c10 
    torch 
    torch_cpu
)

install(TARGETS sample DESTINATION ${ROOT_DIR})

