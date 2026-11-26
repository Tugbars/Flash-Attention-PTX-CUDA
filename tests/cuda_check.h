#pragma once

/**
 * @file cuda_check.h
 * @brief Abort the process on any CUDA error. For the test, bench and demo
 *        executables only.
 *
 * Library code returns cudaError_t instead of using this.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/// Evaluate @p call and exit with a message if it did not return cudaSuccess.
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err_));                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
