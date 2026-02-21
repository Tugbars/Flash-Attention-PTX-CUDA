cmake_minimum_required(VERSION 3.20)
project(flash_attention_cuda CUDA CXX)

# ============================================================================
# Build Configuration
# ============================================================================
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90;120")

# For fastest builds targeting only your GPU, uncomment one:
# set(CMAKE_CUDA_ARCHITECTURES "120")  # RTX 5080 (Blackwell)
# set(CMAKE_CUDA_ARCHITECTURES "89")   # RTX 4090 (Ada)
# set(CMAKE_CUDA_ARCHITECTURES "86")   # RTX 3090 (Ampere)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ============================================================================
# Dependencies
# ============================================================================
find_package(CUDAToolkit REQUIRED)

# ============================================================================
# Flash Attention Library
# ============================================================================
add_library(flash_attention STATIC
    kernels/flash_attention.cu
)

target_include_directories(flash_attention PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(flash_attention PUBLIC
    CUDA::cudart
)

target_compile_options(flash_attention PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        $<$<NOT:$<PLATFORM_ID:Windows>>:-Xcompiler=-fPIC>
    >
)

# ============================================================================
# Demo Executable
# ============================================================================
add_executable(flash_demo src/demo.cu)
target_link_libraries(flash_demo PRIVATE flash_attention)
target_include_directories(flash_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# ============================================================================
# Info
# ============================================================================
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")