/**
 * PRONE Boosted: PRONE + Sensitivity Sampling Coreset + Weighted k-means++
 *
 * Implements the two-stage pipeline from Section 7.3 of the PRONE paper:
 * 1. Run PRONE to get initial k centers and assignments
 * 2. Use sensitivity sampling to construct coreset of size αn
 * 3. Run weighted k-means++ on coreset to get final centers
 * 4. Compute final cost on full dataset
 *
 * Based on:
 * - "Fast k-means++ Initialization via Sampling" (PRONE paper)
 * - "Scalable K-Means++" (Bachem, Lucic, Krause - sensitivity sampling)
 */

#ifndef PRONE_BOOSTED_HPP
#define PRONE_BOOSTED_HPP

#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>

namespace prone_boosted {

/**
 * Coreset structure holding sampled point indices and their weights
 */
struct Coreset {
    std::vector<int> indices;      // Indices of sampled points in original dataset
    std::vector<double> weights;   // Weights for each sampled point
};

/**
 * Compute squared Euclidean distance between two points
 */
inline float squared_distance(const float* a, const float* b, int d) {
    float dist = 0.0f;
    for (int j = 0; j < d; j++) {
        float diff = a[j] - b[j];
        dist += diff * diff;
    }
    return dist;
}

/**
 * Construct coreset using sensitivity sampling
 *
 * Algorithm from Bachem et al. "Scalable K-Means++"
 *
 * @param data Flattened row-major data (n × d)
 * @param n Number of points
 * @param d Dimensionality
 * @param center_indices Indices of initial centers from PRONE
 * @param assignments Cluster assignments from PRONE
 * @param k Number of clusters
 * @param alpha Coreset size parameter (coreset_size = alpha * n)
 * @param random_seed Random seed
 * @return Coreset structure with sampled indices and weights
 */
inline Coreset construct_coreset(const std::vector<float>& data,
                          int n, int d,
                          const std::vector<int>& center_indices,
                          const std::vector<int>& assignments,
                          int k,
                          double alpha,
                          int random_seed) {
    std::mt19937 rng(random_seed);

    // Step 1: Compute distances from each point to its assigned center
    std::vector<double> distances(n);
    std::vector<double> cluster_costs(k, 0.0);
    std::vector<int> cluster_sizes(k, 0);

    for (int i = 0; i < n; i++) {
        int cluster_id = assignments[i];
        int center_idx = center_indices[cluster_id];

        const float* point = &data[i * d];
        const float* center = &data[center_idx * d];

        double dist_sq = squared_distance(point, center, d);
        distances[i] = dist_sq;
        cluster_costs[cluster_id] += dist_sq;
        cluster_sizes[cluster_id]++;
    }

    // Step 2: Compute sensitivity for each point
    // Formula: q(x) = α/(2*cost(X)) * d²(x,C) + α/(2k) * cost(C_i)/|C_i|
    // where C_i is the cluster containing x

    const double ALPHA_CONSTANT = 16.0 * (std::log(k) + 2.0);
    double total_cost = std::accumulate(cluster_costs.begin(), cluster_costs.end(), 0.0);
    double avg_cost = total_cost / n;

    std::vector<double> sensitivity(n);
    double total_sensitivity = 0.0;

    for (int i = 0; i < n; i++) {
        int cluster_id = assignments[i];

        // First term: point-specific contribution
        double term1 = (ALPHA_CONSTANT / avg_cost) * distances[i];

        // Second term: cluster-specific contribution
        double term2 = 0.0;
        if (cluster_sizes[cluster_id] > 0) {
            term2 = (2.0 * ALPHA_CONSTANT / k) * (cluster_costs[cluster_id] / cluster_sizes[cluster_id]);
        }

        sensitivity[i] = (term1 + term2) / (4.0 * n);
        total_sensitivity += sensitivity[i];
    }

    // Normalize sensitivity to get probability distribution
    for (int i = 0; i < n; i++) {
        sensitivity[i] /= total_sensitivity;
    }

    // Step 3: Sample coreset according to sensitivity distribution
    int coreset_size = static_cast<int>(alpha * n);
    coreset_size = std::max(k, std::min(coreset_size, n));  // Clamp to [k, n]

    std::discrete_distribution<int> distribution(sensitivity.begin(), sensitivity.end());

    Coreset coreset;
    coreset.indices.reserve(coreset_size);
    coreset.weights.reserve(coreset_size);

    for (int i = 0; i < coreset_size; i++) {
        int sampled_idx = distribution(rng);
        coreset.indices.push_back(sampled_idx);

        // Weight = total_sensitivity / (coreset_size * sensitivity[sampled_idx])
        double weight = total_sensitivity / (coreset_size * sensitivity[sampled_idx]);
        coreset.weights.push_back(weight);
    }

    return coreset;
}

/**
 * Weighted k-means++ on coreset
 *
 * Modified k-means++ that respects point weights from coreset construction
 *
 * @param data Flattened row-major data (n × d)
 * @param d Dimensionality
 * @param coreset Coreset with indices and weights
 * @param k Number of clusters
 * @param random_seed Random seed
 * @return Vector of k center indices in the ORIGINAL dataset
 */
inline std::vector<int> weighted_kmeanspp(const std::vector<float>& data,
                                    int d,
                                    const Coreset& coreset,
                                    int k,
                                    int random_seed) {
    std::mt19937 rng(random_seed);
    int coreset_size = coreset.indices.size();

    std::vector<int> centers;
    centers.reserve(k);

    // Step 1: Choose first center uniformly at random from coreset
    std::uniform_int_distribution<int> uniform_dist(0, coreset_size - 1);
    int first_center_idx = uniform_dist(rng);
    centers.push_back(coreset.indices[first_center_idx]);

    // Step 2: Iteratively select remaining k-1 centers using weighted D²
    std::vector<double> min_dist_sq(coreset_size, std::numeric_limits<double>::max());

    for (int c = 1; c < k; c++) {
        // Update distances to nearest center for all coreset points
        const float* new_center = &data[centers.back() * d];

        for (int i = 0; i < coreset_size; i++) {
            int point_idx = coreset.indices[i];
            const float* point = &data[point_idx * d];

            double dist_sq = squared_distance(point, new_center, d);
            min_dist_sq[i] = std::min(min_dist_sq[i], dist_sq);
        }

        // Compute weighted D² distribution
        std::vector<double> weighted_d2(coreset_size);
        for (int i = 0; i < coreset_size; i++) {
            weighted_d2[i] = coreset.weights[i] * min_dist_sq[i];
        }

        // Sample next center proportional to weighted D²
        std::discrete_distribution<int> d2_dist(weighted_d2.begin(), weighted_d2.end());
        int next_center_idx = d2_dist(rng);
        centers.push_back(coreset.indices[next_center_idx]);
    }

    return centers;
}

/**
 * Full PRONE (boosted) pipeline
 *
 * @param data Flattened row-major data (n × d)
 * @param n Number of points
 * @param d Dimensionality
 * @param k Number of clusters
 * @param alpha Coreset size parameter (coreset_size = alpha * n)
 * @param random_seed Random seed
 * @return Pair of (centers, assignments)
 *         - centers: flattened k × d array of center coordinates
 *         - assignments: length n array of final cluster assignments
 */
std::pair<std::vector<float>, std::vector<int>>
run_prone_boosted(const std::vector<float>& data,
                  int n, int d, int k,
                  double alpha,
                  int random_seed);

} // namespace prone_boosted

#endif // PRONE_BOOSTED_HPP
