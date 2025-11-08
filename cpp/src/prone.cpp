/**
 * PRONE: PRojected ONE-dimensional k-means++ seeding
 *
 * Implementation based on the paper and fast-coresets repository
 */

#include "kmeans_seeding/prone.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rs_kmeans {

// ============================================================================
// DynamicTree Implementation
// ============================================================================

DynamicTree::DynamicTree(const std::vector<float>& distances)
    : size_(distances.size() - 1), distances_(&distances), tree_(distances.size() - 1) {

    if (size_ == 0) {
        return;
    }

    // Build tree bottom-up
    for (size_t i = size_ - 1; i > 0; i--) {
        if (right_child(i) < size_) {
            // Both children are internal nodes
            tree_[i] = tree_[left_child(i)] + tree_[right_child(i)];
        } else if (left_child(i) < size_) {
            // Left child is internal, right is leaf
            tree_[i] = tree_[left_child(i)] + distances[right_child(i) - size_];
        } else {
            // Both children are leaves
            tree_[i] = distances[left_child(i) - size_] + distances[right_child(i) - size_];
        }
    }

    // Fix the root
    if (size_ > 2) {
        // Both children are internal
        tree_[0] = tree_[1] + tree_[2];
    } else if (size_ > 1) {
        // Left internal, right leaf
        tree_[0] = tree_[1] + distances[2 - size_];
    } else {
        // Both leaves
        tree_[0] = distances[1 - size_] + distances[2 - size_];
    }
}

size_t DynamicTree::find(float value) const {
    if (size_ == 0) {
        throw std::runtime_error("Cannot sample from empty tree");
    }

    size_t index = 0;
    while (index < size_) {
        size_t left = left_child(index);
        size_t right = right_child(index);

        if (right >= size_) {
            // Right child is a leaf
            if (left >= size_) {
                // Both are leaves - compare and return
                float left_val = (*distances_)[left - size_];
                if (value < left_val) {
                    return left - size_;
                } else {
                    return right - size_;
                }
            } else {
                // Left is internal, right is leaf
                if (value > tree_[left]) {
                    return right - size_;
                } else {
                    index = left;
                }
            }
        } else {
            // Both children are internal nodes
            if (value < tree_[left]) {
                index = left;
            } else {
                value -= tree_[left];
                index = right;
            }
        }
    }

    throw std::runtime_error("Tree traversal error - should not reach here");
}

void DynamicTree::update(const std::vector<float>& distances, size_t lower, size_t upper) {
    if (size_ == 0) return;

    // Convert to tree indices
    upper += size_;
    lower += size_;

    do {
        upper = parent(upper);
        lower = parent(lower);

        // Update all nodes in range [lower, upper]
        for (size_t i = upper; i >= lower && i < tree_.size(); i--) {
            size_t left = left_child(i);
            size_t right = right_child(i);

            tree_[i] = 0.0f;

            // Add left child contribution
            if (left < size_) {
                tree_[i] += tree_[left];
            } else {
                tree_[i] += distances[left - size_];
            }

            // Add right child contribution
            if (right < size_) {
                tree_[i] += tree_[right];
            } else {
                tree_[i] += distances[right - size_];
            }

            if (i == 0) break; // Prevent underflow
        }
    } while (lower != 0);
}

// ============================================================================
// PRONE Implementation
// ============================================================================

void PRONE::preprocess(const std::vector<float>& data, int n, int d) {
    if (n <= 0 || d <= 0) {
        throw std::invalid_argument("Invalid dimensions: n and d must be positive");
    }
    if (data.size() != static_cast<size_t>(n * d)) {
        throw std::invalid_argument("Data size mismatch: expected n*d elements");
    }

    data_ = data;
    n_ = n;
    d_ = d;
}

std::pair<std::vector<float>, std::vector<int>> PRONE::cluster(int k, int random_seed) {
    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }
    if (k > n_) {
        throw std::invalid_argument("k cannot exceed number of points");
    }
    if (data_.empty()) {
        throw std::runtime_error("Must call preprocess() before cluster()");
    }

    // Initialize random number generator
    std::mt19937 rng;
    if (random_seed < 0) {
        std::random_device rd;
        rng.seed(rd());
    } else {
        rng.seed(static_cast<unsigned>(random_seed));
    }

    // Project data to 1D
    std::vector<float> projected = project_to_1d();

    // Create sorted indices
    std::vector<size_t> sorted_indices(n_);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&projected](size_t a, size_t b) { return projected[a] < projected[b]; });

    // Sort the projected data
    std::vector<float> sorted_projected(n_);
    for (int i = 0; i < n_; i++) {
        sorted_projected[i] = projected[sorted_indices[i]];
    }

    // Run efficient 1D k-means++
    auto [center_indices_1d, assignments_1d] =
        efficient_kmeans_1d(sorted_projected, sorted_indices, k, rng);

    // Compute exact assignments in original space
    std::vector<int> final_assignments(n_);
    compute_exact_distances(center_indices_1d, assignments_1d, final_assignments);

    // Extract center coordinates
    std::vector<float> centers(k * d_);
    for (int i = 0; i < k; i++) {
        size_t center_idx = center_indices_1d[i];
        for (int j = 0; j < d_; j++) {
            centers[i * d_ + j] = data_[center_idx * d_ + j];
        }
    }

    return {centers, final_assignments};
}

std::vector<float> PRONE::project_to_1d() const {
    switch (projection_type_) {
        case ProjectionType::STANDARD:
            return project_standard();
        case ProjectionType::VARIANCE_WEIGHTED:
            return project_variance_weighted();
        case ProjectionType::COVARIANCE:
            return project_covariance();
        default:
            return project_standard();
    }
}

std::vector<float> PRONE::project_standard() const {
    // Standard Gaussian projection: y = X * g, where g ~ N(0, I_d)
    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<float> normal(0.0f, 1.0f);

    // Generate random Gaussian vector
    std::vector<float> g(d_);
    for (int i = 0; i < d_; i++) {
        g[i] = normal(rng);
    }

    // Compute projection: projected[i] = sum_j data[i,j] * g[j]
    std::vector<float> projected(n_);

    #pragma omp parallel for if(n_ > 1000)
    for (int i = 0; i < n_; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d_; j++) {
            sum += data_[i * d_ + j] * g[j];
        }
        projected[i] = sum;
    }

    return projected;
}

std::vector<float> PRONE::project_variance_weighted() const {
    // Variance-weighted projection: weight each dimension by its standard deviation
    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<float> normal(0.0f, 1.0f);

    // Compute column standard deviations
    std::vector<float> stddev(d_, 0.0f);

    // Compute means
    std::vector<float> means(d_, 0.0f);
    for (int i = 0; i < n_; i++) {
        for (int j = 0; j < d_; j++) {
            means[j] += data_[i * d_ + j];
        }
    }
    for (int j = 0; j < d_; j++) {
        means[j] /= n_;
    }

    // Compute standard deviations
    for (int i = 0; i < n_; i++) {
        for (int j = 0; j < d_; j++) {
            float diff = data_[i * d_ + j] - means[j];
            stddev[j] += diff * diff;
        }
    }
    for (int j = 0; j < d_; j++) {
        stddev[j] = std::sqrt(stddev[j] / n_);
    }

    // Generate weighted random vector
    std::vector<float> g(d_);
    float norm = 0.0f;
    for (int i = 0; i < d_; i++) {
        g[i] = stddev[i] * normal(rng);
        norm += g[i] * g[i];
    }

    // Normalize
    norm = std::sqrt(norm);
    if (norm > 1e-10f) {
        for (int i = 0; i < d_; i++) {
            g[i] /= norm;
        }
    }

    // Compute projection
    std::vector<float> projected(n_);

    #pragma omp parallel for if(n_ > 1000)
    for (int i = 0; i < n_; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d_; j++) {
            sum += data_[i * d_ + j] * g[j];
        }
        projected[i] = sum;
    }

    return projected;
}

std::vector<float> PRONE::project_covariance() const {
    // Covariance-weighted projection: sample from empirical covariance
    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<float> normal(0.0f, 1.0f);

    // Compute column means
    std::vector<float> means(d_, 0.0f);
    for (int i = 0; i < n_; i++) {
        for (int j = 0; j < d_; j++) {
            means[j] += data_[i * d_ + j];
        }
    }
    for (int j = 0; j < d_; j++) {
        means[j] /= n_;
    }

    // Center the data (in-place would modify data_, so we compute on-the-fly)
    // Compute covariance matrix: Σ = (1/n) * X^T X (for centered X)
    std::vector<float> cov(d_ * d_, 0.0f);

    for (int i = 0; i < n_; i++) {
        for (int j = 0; j < d_; j++) {
            float x_ij = data_[i * d_ + j] - means[j];
            for (int k = 0; k < d_; k++) {
                float x_ik = data_[i * d_ + k] - means[k];
                cov[j * d_ + k] += x_ij * x_ik;
            }
        }
    }

    for (int i = 0; i < d_ * d_; i++) {
        cov[i] /= n_;
    }

    // Compute Cholesky decomposition: Σ = L L^T
    // Simplified implementation (assumes positive definite)
    std::vector<float> L(d_ * d_, 0.0f);

    for (int i = 0; i < d_; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++) {
                sum += L[i * d_ + k] * L[j * d_ + k];
            }

            if (i == j) {
                float val = cov[i * d_ + i] - sum;
                L[i * d_ + j] = (val > 0.0f) ? std::sqrt(val) : 0.0f;
            } else {
                L[i * d_ + j] = (L[j * d_ + j] > 1e-10f) ?
                    (cov[i * d_ + j] - sum) / L[j * d_ + j] : 0.0f;
            }
        }
    }

    // Generate standard normal vector
    std::vector<float> z(d_);
    for (int i = 0; i < d_; i++) {
        z[i] = normal(rng);
    }

    // Apply Cholesky factor: g = L * z
    std::vector<float> g(d_, 0.0f);
    for (int i = 0; i < d_; i++) {
        for (int j = 0; j <= i; j++) {
            g[i] += L[i * d_ + j] * z[j];
        }
    }

    // Normalize
    float norm = 0.0f;
    for (int i = 0; i < d_; i++) {
        norm += g[i] * g[i];
    }
    norm = std::sqrt(norm);
    if (norm > 1e-10f) {
        for (int i = 0; i < d_; i++) {
            g[i] /= norm;
        }
    }

    // Compute projection
    std::vector<float> projected(n_);

    #pragma omp parallel for if(n_ > 1000)
    for (int i = 0; i < n_; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d_; j++) {
            sum += data_[i * d_ + j] * g[j];
        }
        projected[i] = sum;
    }

    return projected;
}

std::pair<std::vector<size_t>, std::vector<int>>
PRONE::efficient_kmeans_1d(std::vector<float>& projected_data,
                           std::vector<size_t>& sorted_indices,
                           int k,
                           std::mt19937& rng) {

    // Select first center uniformly at random
    std::uniform_int_distribution<int> uniform(0, n_ - 1);
    size_t first_center = uniform(rng);

    std::vector<size_t> centers;
    centers.push_back(sorted_indices[first_center]);

    std::vector<int> assignments(n_, 0);

    // Initialize distances
    std::vector<float> distances(n_);
    for (int i = 0; i < n_; i++) {
        float diff = projected_data[i] - projected_data[first_center];
        distances[i] = diff * diff;
    }

    // Build dynamic tree
    DynamicTree tree(distances);

    // Main k-means++ loop
    for (int iter = 1; iter < k; iter++) {
        // Sample new center from D² distribution
        std::uniform_real_distribution<float> dist(0.0f, tree.sum());
        float sample = dist(rng);
        size_t new_center_idx = tree.find(sample);

        centers.push_back(sorted_indices[new_center_idx]);
        distances[new_center_idx] = 0.0f;
        assignments[new_center_idx] = iter;

        // Update distances in local neighborhood (key optimization!)
        int lower = static_cast<int>(new_center_idx) - 1;
        size_t upper = new_center_idx + 1;

        // Scan left
        while (lower >= 0) {
            float diff = projected_data[lower] - projected_data[new_center_idx];
            float candidate_dist = diff * diff;
            if (candidate_dist < distances[lower]) {
                distances[lower] = candidate_dist;
                assignments[lower] = iter;
                lower--;
            } else {
                break;
            }
        }

        // Scan right
        while (upper < static_cast<size_t>(n_)) {
            float diff = projected_data[upper] - projected_data[new_center_idx];
            float candidate_dist = diff * diff;
            if (candidate_dist < distances[upper]) {
                distances[upper] = candidate_dist;
                assignments[upper] = iter;
                upper++;
            } else {
                break;
            }
        }

        // Update tree
        tree.update(distances, lower + 1, upper - 1);
    }

    // Map assignments back to original indices
    std::vector<int> original_assignments(n_);
    for (int i = 0; i < n_; i++) {
        original_assignments[sorted_indices[i]] = assignments[i];
    }

    return {centers, original_assignments};
}

void PRONE::compute_exact_distances(const std::vector<size_t>& center_indices,
                                    const std::vector<int>& /* assignments */,
                                    std::vector<int>& final_assignments) const {

    int k = center_indices.size();

    #pragma omp parallel for if(n_ > 1000)
    for (int i = 0; i < n_; i++) {
        float min_dist = std::numeric_limits<float>::max();
        int best_cluster = 0;

        for (int c = 0; c < k; c++) {
            size_t center_idx = center_indices[c];
            float dist = 0.0f;

            for (int j = 0; j < d_; j++) {
                float diff = data_[i * d_ + j] - data_[center_idx * d_ + j];
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }

        final_assignments[i] = best_cluster;
    }
}

} // namespace rs_kmeans
