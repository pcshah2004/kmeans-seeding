/**
 * Lightweight Coreset for k-means clustering
 *
 * Based on "Scalable k-Means Clustering via Lightweight Coresets"
 * Bachem, Lucic, Krause (KDD 2018)
 *
 * Creates a weighted coreset in 2 passes through the data using
 * importance sampling based on distance to the mean.
 */

#ifndef KMEANS_SEEDING_LIGHTWEIGHT_HPP
#define KMEANS_SEEDING_LIGHTWEIGHT_HPP

#include <vector>
#include <random>
#include <cmath>

namespace rs_kmeans {

class Lightweight {
public:
    Lightweight() = default;

    /**
     * Preprocess the data: compute mean and sampling probabilities
     *
     * @param data Flat array of n*d floats (row-major)
     * @param n Number of points
     * @param d Dimensionality
     */
    void preprocess(const std::vector<float>& data, int n, int d);

    /**
     * Construct coreset and run k-means++ on it
     *
     * @param k Number of clusters
     * @param m Coreset size (if -1, use theoretical bound)
     * @param epsilon Approximation parameter (default 0.1)
     * @param random_seed Random seed
     * @return Pair of (centers, labels) where centers is k*d floats, labels is n ints
     */
    std::pair<std::vector<float>, std::vector<int>> cluster(
        int k, int m = -1, double epsilon = 0.1, int random_seed = 42);

private:
    std::vector<float> data_;
    std::vector<float> mean_;
    std::vector<double> sampling_probs_;
    std::vector<double> squared_distances_to_mean_;
    int n_;
    int d_;

    // Compute squared Euclidean distance
    double squared_distance(const float* a, const float* b, int d) const;

    // Sample coreset using importance sampling
    struct WeightedPoint {
        int index;     // Index in original dataset
        double weight; // Coreset weight
    };

    std::vector<WeightedPoint> sample_coreset(int m, std::mt19937& rng);

    // Run weighted k-means++ on coreset
    std::vector<int> weighted_kmeanspp(
        const std::vector<WeightedPoint>& coreset,
        int k,
        std::mt19937& rng);

    // Assign points to nearest centers
    std::vector<int> assign_to_centers(const std::vector<int>& center_indices);
};

} // namespace rs_kmeans

#endif // KMEANS_SEEDING_LIGHTWEIGHT_HPP
