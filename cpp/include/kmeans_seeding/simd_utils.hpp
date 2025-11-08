/**
 * SIMD Utilities for Fast Distance Computations
 *
 * Provides optimized distance calculation using AVX2/NEON depending on architecture
 */

#ifndef KMEANS_SEEDING_SIMD_UTILS_HPP
#define KMEANS_SEEDING_SIMD_UTILS_HPP

#include <cstddef>

namespace rs_kmeans {
namespace simd {

/**
 * Compute squared Euclidean distance between two points using SIMD
 *
 * @param a First point
 * @param b Second point
 * @param d Dimensionality
 * @return Squared L2 distance
 */
float squared_distance_simd(const float* a, const float* b, int d);

/**
 * Compute dot product using SIMD
 */
float dot_product_simd(const float* a, const float* b, int d);

/**
 * Compute L2 norm squared using SIMD
 */
float norm_squared_simd(const float* a, int d);

} // namespace simd
} // namespace rs_kmeans

#endif // KMEANS_SEEDING_SIMD_UTILS_HPP
