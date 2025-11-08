/**
 * PRONE Boosted Implementation
 *
 * Implements the full PRONE + coreset + weighted k-means++ pipeline
 */

#include "prone_boosted.hpp"
#include "kmeans_seeding/prone.hpp"
#include <iostream>

namespace prone_boosted {

/**
 * Full PRONE (boosted) pipeline
 *
 * 1. Run PRONE (standard) to get initial k centers
 * 2. Construct coreset using sensitivity sampling
 * 3. Run weighted k-means++ on coreset
 * 4. Return final centers and compute assignments
 */
std::pair<std::vector<float>, std::vector<int>>
run_prone_boosted(const std::vector<float>& data,
                  int n, int d, int k,
                  double alpha,
                  int random_seed) {

    // Step 1: Run PRONE (standard) to get initial clustering
    rs_kmeans::PRONE prone(rs_kmeans::ProjectionType::STANDARD);
    prone.preprocess(data, n, d);
    auto prone_result = prone.cluster(k, random_seed);

    const auto& prone_centers = prone_result.first;  // k × d flattened array
    const auto& prone_assignments = prone_result.second;  // length n

    // Convert prone_centers to indices by finding nearest points in dataset
    // (PRONE returns actual center coordinates, not indices)
    std::vector<int> center_indices(k);
    for (int c = 0; c < k; c++) {
        const float* center = &prone_centers[c * d];

        // Find closest point in dataset to this center
        int best_idx = 0;
        float best_dist = std::numeric_limits<float>::max();

        for (int i = 0; i < n; i++) {
            const float* point = &data[i * d];
            float dist = squared_distance(point, center, d);

            if (dist < best_dist) {
                best_dist = dist;
                best_idx = i;
            }
        }

        center_indices[c] = best_idx;
    }

    // Step 2: Construct coreset using sensitivity sampling
    Coreset coreset = construct_coreset(data, n, d, center_indices, prone_assignments, k, alpha, random_seed + 1);

    // Step 3: Run weighted k-means++ on coreset
    std::vector<int> final_center_indices = weighted_kmeanspp(data, d, coreset, k, random_seed + 2);

    // Step 4: Extract final centers and compute assignments
    std::vector<float> final_centers(k * d);
    for (int c = 0; c < k; c++) {
        int center_idx = final_center_indices[c];
        for (int j = 0; j < d; j++) {
            final_centers[c * d + j] = data[center_idx * d + j];
        }
    }

    // Compute final assignments to nearest center
    std::vector<int> final_assignments(n);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        const float* point = &data[i * d];

        int best_cluster = 0;
        float best_dist = std::numeric_limits<float>::max();

        for (int c = 0; c < k; c++) {
            const float* center = &final_centers[c * d];
            float dist = squared_distance(point, center, d);

            if (dist < best_dist) {
                best_dist = dist;
                best_cluster = c;
            }
        }

        final_assignments[i] = best_cluster;
    }

    return {final_centers, final_assignments};
}

} // namespace prone_boosted
