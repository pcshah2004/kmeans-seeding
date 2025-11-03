#include "fast_lsh.h"
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace fast_k_means {

// ==================== HadamardTransform Implementation ====================

int HadamardTransform::next_power_of_2(int n) {
    int p = 1;
    while (p < n) p *= 2;
    return p;
}

void HadamardTransform::fht(std::vector<double>& data) {
    int n = data.size();

    // Verify n is power of 2
    if ((n & (n - 1)) != 0) {
        throw std::runtime_error("FHT requires size to be power of 2");
    }

    // In-place fast Hadamard transform
    for (int h = 1; h < n; h *= 2) {
        for (int i = 0; i < n; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                double x = data[j];
                double y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
    }

    // Normalize by 1/sqrt(n)
    double norm = 1.0 / std::sqrt(n);
    for (int i = 0; i < n; i++) {
        data[i] *= norm;
    }
}

void HadamardTransform::ifht(std::vector<double>& data) {
    // For normalized Hadamard, inverse is same as forward
    fht(data);
}

// ==================== FastLSH Implementation ====================

FastLSH::FastLSH(int L, int k, int d, double w)
    : L_(L), k_(k), d_(d), w_(w), num_points_(0), rng_(42) {

    // Pad dimension to power of 2 for Hadamard transform
    d_padded_ = HadamardTransform::next_power_of_2(d);

    // Initialize hash tables
    hash_tables_.resize(L);
    for (int i = 0; i < L; i++) {
        initialize_hash_table(i);
    }
}

void FastLSH::initialize_hash_table(int table_idx) {
    auto& table = hash_tables_[table_idx];

    // D: diagonal ±1 (random sign flips)
    table.D.resize(d_padded_);
    std::uniform_int_distribution<int> sign_dist(0, 1);
    for (int i = 0; i < d_padded_; i++) {
        table.D[i] = sign_dist(rng_) ? 1 : -1;
    }

    // M: random permutation
    table.M.resize(d_padded_);
    for (int i = 0; i < d_padded_; i++) {
        table.M[i] = i;
    }
    std::shuffle(table.M.begin(), table.M.end(), rng_);

    // G: Gaussian N(0,1)
    table.G.resize(d_padded_);
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    for (int i = 0; i < d_padded_; i++) {
        table.G[i] = normal_dist(rng_);
    }

    // b: offset in [0, w]
    table.b.resize(d_padded_);
    std::uniform_real_distribution<double> offset_dist(0.0, w_);
    for (int i = 0; i < d_padded_; i++) {
        table.b[i] = offset_dist(rng_);
    }
}

std::vector<double> FastLSH::apply_transform(const std::vector<double>& point, int table_idx) {
    auto& table = hash_tables_[table_idx];

    // Pad point to d_padded with zeros
    std::vector<double> x(d_padded_, 0.0);
    for (int i = 0; i < d_; i++) {
        x[i] = point[i];
    }

    // Step 1: D (diagonal sign flips)
    for (int i = 0; i < d_padded_; i++) {
        x[i] *= table.D[i];
    }

    // Step 2: H (first Hadamard transform)
    HadamardTransform::fht(x);

    // Step 3: M (permutation)
    std::vector<double> x_perm(d_padded_);
    for (int i = 0; i < d_padded_; i++) {
        x_perm[i] = x[table.M[i]];
    }
    x = std::move(x_perm);

    // Step 4: G (Gaussian scaling)
    for (int i = 0; i < d_padded_; i++) {
        x[i] *= table.G[i];
    }

    // Step 5: H (second Hadamard transform)
    HadamardTransform::fht(x);

    // Step 6: Add offset b
    for (int i = 0; i < d_padded_; i++) {
        x[i] += table.b[i];
    }

    return x;
}

std::vector<std::vector<int>> FastLSH::compute_dhhash(const std::vector<double>& point) {
    std::vector<std::vector<int>> hashes(L_);

    for (int table_idx = 0; table_idx < L_; table_idx++) {
        // Apply transformation pipeline: D → H → M → G → H → +b
        std::vector<double> transformed = apply_transform(point, table_idx);

        // Compute hash values: ⌊transformed / w⌋
        // Sample k entries without replacement
        hashes[table_idx].resize(k_);

        // Use systematic sampling to get k diverse hash values
        int step = d_padded_ / k_;
        for (int i = 0; i < k_; i++) {
            int idx = i * step;
            hashes[table_idx][i] = static_cast<int>(std::floor(transformed[idx] / w_));
        }
    }

    return hashes;
}

void FastLSH::InsertPoint(int point_id, const std::vector<double>& point) {
    if (point.size() != static_cast<size_t>(d_)) {
        throw std::runtime_error("Point dimension mismatch");
    }

    // Compute DHHash for this point
    auto hashes = compute_dhhash(point);

    // Insert into each hash table
    for (int table_idx = 0; table_idx < L_; table_idx++) {
        auto& table = hash_tables_[table_idx];
        const auto& hash_key = hashes[table_idx];

        // Add to bucket (or create if doesn't exist)
        table.buckets[hash_key].push_back(point_id);
    }

    num_points_++;
}

std::vector<int> FastLSH::QueryPoint(const std::vector<double>& point, int max_candidates) {
    if (point.size() != static_cast<size_t>(d_)) {
        throw std::runtime_error("Point dimension mismatch");
    }

    // Compute DHHash for query point
    auto hashes = compute_dhhash(point);

    // Collect candidates from all hash tables
    std::unordered_map<int, int> candidate_counts;

    for (int table_idx = 0; table_idx < L_; table_idx++) {
        auto& table = hash_tables_[table_idx];
        const auto& hash_key = hashes[table_idx];

        // Find bucket
        auto it = table.buckets.find(hash_key);
        if (it != table.buckets.end()) {
            // Add all points in this bucket as candidates
            for (int point_id : it->second) {
                candidate_counts[point_id]++;
            }
        }
    }

    // Convert to vector and sort by count (points appearing in more tables are better)
    std::vector<std::pair<int, int>> candidates;
    for (const auto& pair : candidate_counts) {
        candidates.push_back({pair.second, pair.first});  // {count, point_id}
    }
    std::sort(candidates.rbegin(), candidates.rend());  // Descending by count

    // Extract point IDs up to max_candidates
    std::vector<int> result;
    int limit = std::min(max_candidates, static_cast<int>(candidates.size()));
    result.reserve(limit);
    for (int i = 0; i < limit; i++) {
        result.push_back(candidates[i].second);
    }

    return result;
}

} // namespace fast_k_means
