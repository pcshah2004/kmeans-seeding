/**
 * PRONE: PRojected ONE-dimensional k-means++ seeding
 *
 * Implements the algorithms from "Fast k-means++ Initialization via Sampling"
 * Uses random 1D projection to reduce k-means++ seeding from O(nkd) to O(nnz(X) + n log n)
 *
 * Key algorithmic components:
 * 1. Random Gaussian projection to 1D
 * 2. Efficient 1D k-means++ seeding with dynamic tree structure
 * 3. Three projection variants: standard, variance-weighted, covariance-weighted
 */

#ifndef KMEANS_SEEDING_PRONE_HPP
#define KMEANS_SEEDING_PRONE_HPP

#include <vector>
#include <utility>
#include <random>

namespace rs_kmeans {

/**
 * Projection type for PRONE algorithm
 */
enum class ProjectionType {
    STANDARD,          // Standard Gaussian projection (default)
    VARIANCE_WEIGHTED, // Variance-weighted projection
    COVARIANCE         // Covariance-weighted projection
};

/**
 * Dynamic binary tree for efficient D² sampling in 1D
 *
 * Maintains a complete binary tree where:
 * - Leaves store squared distances to nearest center
 * - Internal nodes store sum of subtree weights
 *
 * Operations:
 * - sum(): O(1) - returns total weight
 * - find(value): O(log n) - samples from D² distribution
 * - update(lower, upper): O(log n) - updates tree after distance changes
 */
class DynamicTree {
public:
    explicit DynamicTree(const std::vector<float>& distances);

    // Sample an index from the D² distribution
    size_t find(float value) const;

    // Update the tree after distances[lower..upper] have changed
    void update(const std::vector<float>& distances, size_t lower, size_t upper);

    // Get total sum of all distances
    float sum() const { return tree_.empty() ? 0.0f : tree_[0]; }

private:
    static size_t left_child(size_t index) { return 2 * index + 1; }
    static size_t right_child(size_t index) { return 2 * index + 2; }
    static size_t parent(size_t index) { return (index - 1) / 2; }

    size_t size_;
    const std::vector<float>* distances_; // Pointer to distance array
    std::vector<float> tree_;              // Binary tree storing sums
};

/**
 * PRONE: Fast k-means++ seeding via 1D projection
 *
 * Main class implementing the PRONE algorithm with three variants
 */
class PRONE {
public:
    PRONE() : projection_type_(ProjectionType::STANDARD) {}
    explicit PRONE(ProjectionType type) : projection_type_(type) {}

    /**
     * Preprocess the dataset
     * @param data Flattened row-major data (n × d)
     * @param n Number of points
     * @param d Dimensionality
     */
    void preprocess(const std::vector<float>& data, int n, int d);

    /**
     * Run PRONE k-means++ seeding
     *
     * @param k Number of clusters
     * @param random_seed Random seed (-1 for random)
     * @return Pair of (centers, assignments)
     *         - centers: flattened k × d array of center coordinates
     *         - assignments: length n array of cluster assignments
     */
    std::pair<std::vector<float>, std::vector<int>> cluster(int k, int random_seed = -1);

    /**
     * Get the projection type being used
     */
    ProjectionType get_projection_type() const { return projection_type_; }

    /**
     * Set the projection type
     */
    void set_projection_type(ProjectionType type) { projection_type_ = type; }

private:
    // Project data to 1D using specified projection type
    std::vector<float> project_to_1d() const;

    // Standard Gaussian projection
    std::vector<float> project_standard() const;

    // Variance-weighted projection
    std::vector<float> project_variance_weighted() const;

    // Covariance-weighted projection
    std::vector<float> project_covariance() const;

    // Efficient 1D k-means++ on sorted projected data
    // Returns (center_indices, assignments)
    std::pair<std::vector<size_t>, std::vector<int>>
    efficient_kmeans_1d(std::vector<float>& projected_data,
                        std::vector<size_t>& sorted_indices,
                        int k,
                        std::mt19937& rng);

    // Compute exact distances in original d-dimensional space
    void compute_exact_distances(const std::vector<size_t>& center_indices,
                                 const std::vector<int>& assignments,
                                 std::vector<int>& final_assignments) const;

    std::vector<float> data_;   // Original data (n × d, row-major)
    int n_;                     // Number of points
    int d_;                     // Dimensionality
    ProjectionType projection_type_;
};

} // namespace rs_kmeans

#endif // KMEANS_SEEDING_PRONE_HPP
