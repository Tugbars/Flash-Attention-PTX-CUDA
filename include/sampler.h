#pragma once

// ============================================================================
// sampler.h — Token Sampling from Logits
//
// Implements the standard sampling pipeline:
//   1. Repetition penalty (optional)
//   2. Temperature scaling
//   3. Top-K filtering
//   4. Top-P (nucleus) filtering
//   5. Softmax → probability distribution
//   6. Weighted random sample (or argmax for temp=0)
//
// All operations run on CPU. Copying vocab_size floats (~512 KB for 128K vocab)
// from GPU takes ~25 μs — irrelevant for decode where we sample once per step.
//
// Usage:
//   Sampler sampler;
//   sampler.temperature = 0.7f;
//   sampler.top_k = 50;
//   sampler.top_p = 0.9f;
//   int32_t next_token = sampler.sample(logits, vocab_size);
// ============================================================================

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace transformer {

struct Sampler {
    float    temperature     = 0.7f;     // 0 = greedy (argmax)
    int32_t  top_k           = 50;       // 0 = disabled
    float    top_p           = 0.9f;     // 1.0 = disabled
    float    rep_penalty     = 1.0f;     // 1.0 = disabled
    uint64_t seed            = 0;        // 0 = random seed

    // Internal state
    std::mt19937 rng_;
    bool rng_initialized_ = false;

    void init_rng() {
        if (seed != 0) {
            rng_.seed(static_cast<uint32_t>(seed));
        } else {
            std::random_device rd;
            rng_.seed(rd());
        }
        rng_initialized_ = true;
    }

    // ========================================================================
    // Sample next token from logits
    //
    // logits:     [vocab_size] float array (CPU memory)
    // vocab_size: number of entries
    // context:    recent token IDs for repetition penalty (can be nullptr)
    // ctx_len:    number of tokens in context
    // ========================================================================
    int32_t sample(
        const float* logits,
        int32_t vocab_size,
        const int32_t* context = nullptr,
        int32_t ctx_len = 0)
    {
        if (!rng_initialized_) init_rng();

        // Work on a copy
        std::vector<float> probs(logits, logits + vocab_size);

        // 1. Repetition penalty
        if (rep_penalty != 1.0f && context && ctx_len > 0) {
            apply_repetition_penalty(probs.data(), vocab_size, context, ctx_len);
        }

        // 2. Greedy (temperature = 0)
        if (temperature == 0.0f || temperature < 1e-8f) {
            return static_cast<int32_t>(
                std::max_element(probs.begin(), probs.end()) - probs.begin());
        }

        // 3. Temperature scaling
        float inv_temp = 1.0f / temperature;
        for (auto& p : probs) p *= inv_temp;

        // 4. Build sorted index for top-k/top-p
        std::vector<int32_t> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0);

        // Partial sort for top-k (avoid full sort of 128K elements)
        int32_t k = (top_k > 0 && top_k < vocab_size) ? top_k : vocab_size;
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
            [&probs](int32_t a, int32_t b) { return probs[a] > probs[b]; });

        // 5. Softmax over top-k candidates
        float max_logit = probs[indices[0]];
        float sum = 0.0f;
        std::vector<float> candidate_probs(k);
        for (int32_t i = 0; i < k; i++) {
            candidate_probs[i] = expf(probs[indices[i]] - max_logit);
            sum += candidate_probs[i];
        }
        float inv_sum = 1.0f / sum;
        for (int32_t i = 0; i < k; i++) {
            candidate_probs[i] *= inv_sum;
        }

        // 6. Top-P (nucleus) filtering
        int32_t n_candidates = k;
        if (top_p < 1.0f && top_p > 0.0f) {
            float cumsum = 0.0f;
            for (int32_t i = 0; i < k; i++) {
                cumsum += candidate_probs[i];
                if (cumsum >= top_p) {
                    n_candidates = i + 1;
                    break;
                }
            }
            // Re-normalize
            sum = 0.0f;
            for (int32_t i = 0; i < n_candidates; i++) sum += candidate_probs[i];
            inv_sum = 1.0f / sum;
            for (int32_t i = 0; i < n_candidates; i++) candidate_probs[i] *= inv_sum;
        }

        // 7. Weighted random sample
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng_);
        float cumsum = 0.0f;
        for (int32_t i = 0; i < n_candidates; i++) {
            cumsum += candidate_probs[i];
            if (r <= cumsum) {
                return indices[i];
            }
        }

        // Fallback (numerical edge case)
        return indices[n_candidates - 1];
    }

private:
    void apply_repetition_penalty(
        float* logits, int32_t vocab_size,
        const int32_t* context, int32_t ctx_len)
    {
        // Penalize tokens that appeared in recent context
        for (int32_t i = 0; i < ctx_len; i++) {
            int32_t id = context[i];
            if (id < 0 || id >= vocab_size) continue;
            if (logits[id] > 0) {
                logits[id] /= rep_penalty;
            } else {
                logits[id] *= rep_penalty;
            }
        }
    }
};

} // namespace transformer
