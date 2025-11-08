/**
 * SIMD Utilities Implementation
 *
 * Optimized distance computations using platform-specific SIMD instructions
 */

#include "kmeans_seeding/simd_utils.hpp"
#include <cstring>

// Detect architecture and include appropriate intrinsics
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef __AVX2__
        #define USE_AVX2
        #include <immintrin.h>
    #elif __SSE2__
        #define USE_SSE2
        #include <emmintrin.h>
    #endif
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #define USE_NEON
    #include <arm_neon.h>
#endif

namespace rs_kmeans {
namespace simd {

// ============================ AVX2 Implementation ============================
#ifdef USE_AVX2

float squared_distance_simd(const float* a, const float* b, int d) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;

    // Process 8 floats at a time with AVX2
    for (; i + 7 < d; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);  // FMA: sum += diff * diff
    }

    // Horizontal sum of 8 floats
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(sum_low, sum_high);

    // Sum 4 floats in sum128
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);

    float result = _mm_cvtss_f32(sum128);

    // Handle remainder
    for (; i < d; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }

    return result;
}

float dot_product_simd(const float* a, const float* b, int d) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;

    for (; i + 7 < d; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(sum_low, sum_high);

    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);

    float result = _mm_cvtss_f32(sum128);

    for (; i < d; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

float norm_squared_simd(const float* a, int d) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;

    for (; i + 7 < d; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        sum = _mm256_fmadd_ps(va, va, sum);
    }

    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(sum_low, sum_high);

    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);

    float result = _mm_cvtss_f32(sum128);

    for (; i < d; ++i) {
        result += a[i] * a[i];
    }

    return result;
}

// ============================ NEON Implementation (ARM) ============================
#elif defined(USE_NEON)

float squared_distance_simd(const float* a, const float* b, int d) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    int i = 0;

    // Process 4 floats at a time with NEON
    for (; i + 3 < d; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t diff = vsubq_f32(va, vb);
        sum = vmlaq_f32(sum, diff, diff);  // sum += diff * diff
    }

    // Horizontal sum
    float32x2_t sum_low = vget_low_f32(sum);
    float32x2_t sum_high = vget_high_f32(sum);
    float32x2_t sum_pair = vadd_f32(sum_low, sum_high);
    float result = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);

    // Handle remainder
    for (; i < d; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }

    return result;
}

float dot_product_simd(const float* a, const float* b, int d) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    int i = 0;

    for (; i + 3 < d; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        sum = vmlaq_f32(sum, va, vb);
    }

    float32x2_t sum_low = vget_low_f32(sum);
    float32x2_t sum_high = vget_high_f32(sum);
    float32x2_t sum_pair = vadd_f32(sum_low, sum_high);
    float result = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);

    for (; i < d; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

float norm_squared_simd(const float* a, int d) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    int i = 0;

    for (; i + 3 < d; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        sum = vmlaq_f32(sum, va, va);
    }

    float32x2_t sum_low = vget_low_f32(sum);
    float32x2_t sum_high = vget_high_f32(sum);
    float32x2_t sum_pair = vadd_f32(sum_low, sum_high);
    float result = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);

    for (; i < d; ++i) {
        result += a[i] * a[i];
    }

    return result;
}

// ============================ Scalar Fallback ============================
#else

float squared_distance_simd(const float* a, const float* b, int d) {
    float result = 0.0f;
    for (int i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

float dot_product_simd(const float* a, const float* b, int d) {
    float result = 0.0f;
    for (int i = 0; i < d; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

float norm_squared_simd(const float* a, int d) {
    float result = 0.0f;
    for (int i = 0; i < d; ++i) {
        result += a[i] * a[i];
    }
    return result;
}

#endif

} // namespace simd
} // namespace rs_kmeans
