/**
 * Implementation of Lightweight Coreset for k-means
 */

#include "kmeans_seeding/lightweight.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rs_kmeans {

void Lightweight::preprocess(const std::vector<float>& data, int n, int d) {
    data_ = data;
    n_ = n;
    d_ = d;
    mean_.resize(d, 0.0f);
    sampling_probs_.resize(n);
    squared_distances_to_mean_.resize(n);

    // Pass 1: Compute mean
    #ifdef _OPENMP
    #pragma omp parallel
    {
        std::vector<float> local_mean(d, 0.0f);
        #pragma omp for nowait
        for (int i = 0; i < n; ++i) {
            const float* row = &data[i * d];
            for (int j = 0; j < d; ++j) {
                local_mean[j] += row[j];
            }
        }
        #pragma omp critical
        {
            for (int j = 0; j < d; ++j) {
                mean_[j] += local_mean[j];
            }
        }
    }
    #else
    for (int i = 0; i < n; ++i) {
        const float* row = &data[i * d];
        for (int j = 0; j < d; ++j) {
            mean_[j] += row[j];
        }
    }
    #endif

    for (int j = 0; j < d; ++j) {
        mean_[j] /= n;
    }

    // Pass 2: Compute squared distances to mean and sampling probabilities
    double total_squared_distance = 0.0;

    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:total_squared_distance)
    #endif
    for (int i = 0; i < n; ++i) {
        const float* point = &data[i * d];
        double dist_sq = squared_distance(point, mean_.data(), d);
        squared_distances_to_mean_[i] = dist_sq;
        total_squared_distance += dist_sq;
    }

    // Compute sampling probabilities: q(x) = 0.5 * (1/n) + 0.5 * d(x,μ)²/Σd(x',μ)²
    for (int i = 0; i < n; ++i) {
        double uniform_part = 0.5 * (1.0 / n);
        double distance_part = 0.5 * (squared_distances_to_mean_[i] / total_squared_distance);
        sampling_probs_[i] = uniform_part + distance_part;
    }
}

std::pair<std::vector<float>, std::vector<int>> Lightweight::cluster(
    int k, int m, double epsilon, int random_seed) {

    std::mt19937 rng(random_seed);

    // Compute coreset size if not provided
    if (m <= 0) {
        // Theoretical bound: O(dk log k / ε²)
        // Use c=100 as a conservative constant
        double log_k = std::log(k);
        m = static_cast<int>(std::ceil(100.0 * d_ * k * log_k / (epsilon * epsilon)));
        m = std::max(m, k * 10);  // At least 10x the number of clusters
        m = std::min(m, n_);       // But no more than the dataset size
    }

    // Sample coreset
    auto coreset = sample_coreset(m, rng);

    // Run weighted k-means++ on coreset
    auto center_indices = weighted_kmeanspp(coreset, k, rng);

    // Extract centers
    std::vector<float> centers(k * d_);
    for (int i = 0; i < k; ++i) {
        const float* point = &data_[center_indices[i] * d_];
        std::copy(point, point + d_, &centers[i * d_]);
    }

    // Assign all points to nearest centers
    auto labels = assign_to_centers(center_indices);

    return {centers, labels};
}

double Lightweight::squared_distance(const float* a, const float* b, int d) const {
    double sum = 0.0;
    for (int i = 0; i < d; ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

std::vector<Lightweight::WeightedPoint> Lightweight::sample_coreset(
    int m, std::mt19937& rng) {

    std::vector<WeightedPoint> coreset;
    coreset.reserve(m);

    // Create discrete distribution for sampling
    std::discrete_distribution<int> sampler(sampling_probs_.begin(), sampling_probs_.end());

    for (int i = 0; i < m; ++i) {
        int idx = sampler(rng);
        double weight = 1.0 / (m * sampling_probs_[idx]);
        coreset.push_back({idx, weight});
    }

    return coreset;
}

std::vector<int> Lightweight::weighted_kmeanspp(
    const std::vector<WeightedPoint>& coreset,
    int k,
    std::mt19937& rng) {

    std::vector<int> centers;
    centers.reserve(k);
    int coreset_size = coreset.size();

    // Distance squared from each coreset point to nearest center (weighted)
    std::vector<double> min_dist_sq(coreset_size, std::numeric_limits<double>::max());

    // Choose first center uniformly at random (weighted)
    std::vector<double> weights(coreset_size);
    for (int i = 0; i < coreset_size; ++i) {
        weights[i] = coreset[i].weight;
    }
    std::discrete_distribution<int> initial_sampler(weights.begin(), weights.end());
    int first_center_idx = coreset[initial_sampler(rng)].index;
    centers.push_back(first_center_idx);

    // Update distances to first center
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < coreset_size; ++i) {
        const float* point = &data_[coreset[i].index * d_];
        const float* center = &data_[first_center_idx * d_];
        min_dist_sq[i] = squared_distance(point, center, d_);
    }

    // Select remaining k-1 centers using D² weighting
    for (int iter = 1; iter < k; ++iter) {
        // Compute D² probabilities (weighted)
        std::vector<double> d2_probs(coreset_size);
        for (int i = 0; i < coreset_size; ++i) {
            d2_probs[i] = coreset[i].weight * min_dist_sq[i];
        }

        // Sample next center
        std::discrete_distribution<int> d2_sampler(d2_probs.begin(), d2_probs.end());
        int next_center_idx = coreset[d2_sampler(rng)].index;
        centers.push_back(next_center_idx);

        // Update minimum distances
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < coreset_size; ++i) {
            const float* point = &data_[coreset[i].index * d_];
            const float* center = &data_[next_center_idx * d_];
            double dist_sq = squared_distance(point, center, d_);
            min_dist_sq[i] = std::min(min_dist_sq[i], dist_sq);
        }
    }

    return centers;
}

std::vector<int> Lightweight::assign_to_centers(const std::vector<int>& center_indices) {
    std::vector<int> labels(n_);

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < n_; ++i) {
        const float* point = &data_[i * d_];
        double min_dist = std::numeric_limits<double>::max();
        int best_center = 0;

        for (size_t j = 0; j < center_indices.size(); ++j) {
            const float* center = &data_[center_indices[j] * d_];
            double dist = squared_distance(point, center, d_);
            if (dist < min_dist) {
                min_dist = dist;
                best_center = j;
            }
        }

        labels[i] = best_center;
    }

    return labels;
}

} // namespace rs_kmeans
