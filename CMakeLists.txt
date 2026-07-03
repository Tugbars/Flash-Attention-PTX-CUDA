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
    kernels/fa_autotune.cu
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
# Demo — Runs kernel, dumps visualization data
# ============================================================================
add_executable(flash_demo src/demo.cu)
target_link_libraries(flash_demo PRIVATE flash_attention)
target_include_directories(flash_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# ============================================================================
# Benchmark — Correctness tests + performance suite
# ============================================================================
add_executable(flash_bench src/bench.cu)
target_link_libraries(flash_bench PRIVATE flash_attention)
target_include_directories(flash_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# ============================================================================
# Strict correctness gate — std=1 inputs, double reference, noise-floor
# thresholds. Compiles the decode kernel directly (it is not part of the
# flash_attention library). Run after any kernel change; exit code 0 = pass.
# ============================================================================
add_executable(fa_validate
    tests/fa_validate.cu
    kernels/flash_attention_decode.cu
)
target_link_libraries(fa_validate PRIVATE flash_attention)
target_include_directories(fa_validate PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_options(fa_validate PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Varlen (ragged-batch / chunked prefill) gate.
add_executable(fa_varlen_validate tests/fa_varlen_validate.cu)
target_link_libraries(fa_varlen_validate PRIVATE flash_attention)
target_include_directories(fa_varlen_validate PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Paged-KV decode gate (shuffled pages, poisoned pool).
add_executable(fa_paged_validate
    tests/fa_paged_validate.cu
    kernels/flash_attention_decode.cu
)
target_link_libraries(fa_paged_validate PRIVATE flash_attention)
target_include_directories(fa_paged_validate PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# FP8 KV-cache gate (writer bit-check + quantization-aware reference).
add_executable(fa_fp8_validate
    tests/fa_fp8_validate.cu
    kernels/flash_attention_decode.cu
)
target_link_libraries(fa_fp8_validate PRIVATE flash_attention)
target_include_directories(fa_fp8_validate PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# INT4_G32 KV-cache gate (group-wise scales, writer bit-check).
add_executable(fa_int4_validate
    tests/fa_int4_validate.cu
    kernels/flash_attention_decode.cu
)
target_link_libraries(fa_int4_validate PRIVATE flash_attention)
target_include_directories(fa_int4_validate PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# ============================================================================
# Info
# ============================================================================
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")