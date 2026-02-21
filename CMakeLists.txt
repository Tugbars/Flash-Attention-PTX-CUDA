cmake_minimum_required(VERSION 3.20)
project(cuda_transformer CUDA CXX)

# ============================================================================
# Build Configuration
# ============================================================================
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")  # Ampere, Ada, Hopper

# Aggressive optimization
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# ============================================================================
# Dependencies
# ============================================================================

# CUDA
find_package(CUDAToolkit REQUIRED)

# CUTLASS (header-only — expects CUTLASS_DIR to be set or found in standard paths)
# Clone: git clone https://github.com/NVIDIA/cutlass.git external/cutlass
set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/external/cutlass" CACHE PATH "Path to CUTLASS")
option(USE_CUTLASS "Enable CUTLASS GEMM kernels" ON)

if(USE_CUTLASS AND EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(STATUS "CUTLASS found at: ${CUTLASS_DIR}")
    add_compile_definitions(USE_CUTLASS)
    set(CUTLASS_INCLUDE_DIR "${CUTLASS_DIR}/include")
else()
    message(WARNING "CUTLASS not found at ${CUTLASS_DIR}, falling back to cuBLAS only")
    set(USE_CUTLASS OFF)
    set(CUTLASS_INCLUDE_DIR "")
endif()

# ============================================================================
# Library Target
# ============================================================================
add_library(cuda_transformer STATIC
    src/transformer.cu
    kernels/flash_attention.cu
    kernels/fused_layernorm.cu
    kernels/rotary_embedding.cu
    kernels/activation_kernels.cu
)

target_include_directories(cuda_transformer PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUTLASS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(cuda_transformer PUBLIC
    CUDA::cudart
    CUDA::cublas
)

# Enable extended lambda and relaxed constexpr for CUTLASS
target_compile_options(cuda_transformer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        $<$<NOT:$<PLATFORM_ID:Windows>>:-Xcompiler=-fPIC>
    >
)

# ============================================================================
# Benchmark / Demo Executable
# ============================================================================
add_executable(transformer_bench src/main.cu)
target_link_libraries(transformer_bench PRIVATE cuda_transformer)
target_include_directories(transformer_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUTLASS_INCLUDE_DIR}
)

# ============================================================================
# Install
# ============================================================================
install(TARGETS cuda_transformer ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

# ============================================================================
# Info
# ============================================================================
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUTLASS enabled: ${USE_CUTLASS}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")