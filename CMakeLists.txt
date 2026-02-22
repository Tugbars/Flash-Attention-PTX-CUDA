cmake_minimum_required(VERSION 3.20)
project(cuda_transformer CUDA CXX)

# ============================================================================
# Build Configuration
# ============================================================================
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90;120")  # Ampere, Ada, Hopper, Blackwell

# For faster builds targeting only your GPU, uncomment:
# set(CMAKE_CUDA_ARCHITECTURES "120")  # RTX 5080 Blackwell only

# Aggressive optimization
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# ============================================================================
# Dependencies
# ============================================================================

# CUDA
find_package(CUDAToolkit REQUIRED)

# CUTLASS (header-only — optional, for fused epilogues and FP8)
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

# FP8 support — requires sm_89+ (Ada/Blackwell) and CUTLASS
option(ENABLE_FP8 "Enable FP8 E4M3 quantized GEMM kernels" ON)
if(ENABLE_FP8)
    if(NOT USE_CUTLASS)
        message(WARNING "FP8 requires CUTLASS — disabling FP8")
        set(ENABLE_FP8 OFF)
    else()
        message(STATUS "FP8 E4M3 quantization enabled")
        add_compile_definitions(ENABLE_FP8)
    endif()
endif()

# ============================================================================
# Kernel Library (static — linked by all executables)
#
# Contains all GPU kernels: Flash Attention, LayerNorm, RoPE, activations.
# Each executable links against this to avoid recompiling kernels.
# ============================================================================
set(KERNEL_SOURCES
    kernels/flash_attention.cu
    kernels/fused_layernorm.cu
    kernels/rotary_embedding.cu
    kernels/activation_kernels.cu
)

add_library(cuda_kernels STATIC ${KERNEL_SOURCES})

target_include_directories(cuda_kernels PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUTLASS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(cuda_kernels PUBLIC
    CUDA::cudart
    CUDA::cublas
)

target_compile_options(cuda_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        $<$<NOT:$<PLATFORM_ID:Windows>>:-Xcompiler=-fPIC>
    >
)

# ============================================================================
# Inference Binary
#
#   ./inference model.forge "Your prompt here" [--temp 0.7] [--top-k 50] ...
#
# Loads .forge weights + .forge.vocab tokenizer, runs prefill + autoregressive
# decode with CUDA graph acceleration. Supports greedy and sampled generation.
# ============================================================================
add_executable(inference src/main_inference.cu)
target_link_libraries(inference PRIVATE cuda_kernels)
target_include_directories(inference PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(inference PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
)

# ============================================================================
# Training Binary (frozen backbone, lm_head only)
#
#   ./train model.forge --data train.txt --lr 1e-4 --steps 200
#
# Proof-of-concept fine-tuning. All transformer layers are frozen; only the
# final lm_head projection is updated via AdamW. Includes cross-entropy loss,
# backward GEMM through lm_head, and FP32 optimizer state.
# ============================================================================
add_executable(train src/train_lmhead.cu)
target_link_libraries(train PRIVATE cuda_kernels)
target_include_directories(train PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(train PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
)

# ============================================================================
# Test / Benchmark Suite
#
#   ./tests
#
# 26 accuracy tests (GEMM, LayerNorm, RoPE, activations, Flash Attention MHA
# + GQA) against FP32 CPU references. Kernel benchmarks (FA, GEMM, LN).
# Forward pass eager vs CUDA graph. MHA vs GQA throughput comparison.
# Decode step eager vs graph (full pipeline including lm_head + logits copy).
# ============================================================================
add_executable(tests src/main.cu)
target_link_libraries(tests PRIVATE cuda_kernels)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
)

# ============================================================================
# Note: transformer_block.cu is a library file (no main), included in the
# kernel lib or compiled into executables that need it. bench_forward
# functionality is covered by the tests executable's forward pass benchmarks.
# ============================================================================

# ============================================================================
# FP8 Benchmark (optional — requires CUTLASS + sm_89+)
# ============================================================================
if(ENABLE_FP8)
    add_executable(bench_fp8 src/bench_fp8.cu)
    target_link_libraries(bench_fp8 PRIVATE cuda_kernels)
    target_include_directories(bench_fp8 PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUTLASS_INCLUDE_DIR}
    )
    target_compile_options(bench_fp8 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
    )
endif()

# ============================================================================
# Install
# ============================================================================
install(TARGETS inference train tests RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(FILES convert.py DESTINATION bin)

# ============================================================================
# Info
# ============================================================================
message(STATUS "")
message(STATUS "  ┌────────────────────────────────────────┐")
message(STATUS "  │       CUDA Transformer Engine           │")
message(STATUS "  └────────────────────────────────────────┘")
message(STATUS "  CUDA architectures:  ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUTLASS enabled:     ${USE_CUTLASS}")
message(STATUS "  FP8 E4M3 enabled:    ${ENABLE_FP8}")
message(STATUS "  Build type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "  Targets:")
message(STATUS "    inference      — End-to-end text generation")
message(STATUS "    train          — Fine-tune lm_head (frozen backbone)")
message(STATUS "    tests          — 26 accuracy tests + kernel benchmarks")
message(STATUS "")
message(STATUS "  Quick start:")
message(STATUS "    cmake -B build -DCMAKE_BUILD_TYPE=Release")
message(STATUS "    cmake --build build -j$(nproc)")
message(STATUS "    python convert.py --model meta-llama/Llama-3.2-1B -o llama.forge")
message(STATUS "    ./build/inference llama.forge \"Once upon a time\"")
message(STATUS "")