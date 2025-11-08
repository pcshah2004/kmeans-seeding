/**
 * Four-Method K-means++ Benchmark
 *
 * Benchmarks k-means++ initialization algorithms:
 * 1. RS-k-means++ (FastLSH) - Our optimized DHHash implementation
 * 2. RejectionSamplingLSH (Google 2020) - Multi-tree implementation
 * 3. AFK-MC² (MCMC) - Markov chain based sampling
 * 4. PRONE (Standard) - 1D projection-based seeding
 * 5. PRONE (Variance) - Variance-weighted projection
 * 6. PRONE (Covariance) - Covariance-weighted projection
 *
 * Results are written to CSV for plotting with Python.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cmath>
#include "kmeans_seeding/rs_kmeans.hpp"
#include "kmeans_seeding/rejection_sampling_lsh.h"
#include "kmeans_seeding/prone.hpp"

#ifdef HAS_FAISS
#include "kmeans_seeding/afkmc2.hpp"
#endif

// Load numpy array from .npy file (simple float32 format)
std::vector<float> load_npy(const std::string& filename, int& n, int& d) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Read numpy header (simplified - assumes float32, C order)
    char magic[6];
    file.read(magic, 6);

    uint8_t major_version, minor_version;
    file.read(reinterpret_cast<char*>(&major_version), 1);
    file.read(reinterpret_cast<char*>(&minor_version), 1);

    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);

    std::string header(header_len, ' ');
    file.read(&header[0], header_len);

    // Parse shape from header
    size_t shape_start = header.find("'shape': (");
    if (shape_start == std::string::npos) {
        shape_start = header.find("\"shape\": (");
    }

    if (shape_start != std::string::npos) {
        size_t paren_start = header.find('(', shape_start);
        size_t comma_pos = header.find(',', paren_start);
        size_t paren_end = header.find(')', comma_pos);

        n = std::stoi(header.substr(paren_start + 1, comma_pos - paren_start - 1));
        d = std::stoi(header.substr(comma_pos + 1, paren_end - comma_pos - 1));
    } else {
        throw std::runtime_error("Could not parse shape from numpy header");
    }

    // Read data
    std::vector<float> data(n * d);
    file.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));

    std::cout << "Loaded " << n << " x " << d << " array from " << filename << std::endl;
    return data;
}

// Convert flat data to vector<vector<double>> format for Google's algorithms
std::vector<std::vector<double>> toGoogleFormat(const std::vector<float>& data, int n, int d) {
    std::vector<std::vector<double>> result(n, std::vector<double>(d));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            result[i][j] = static_cast<double>(data[i * d + j]);
        }
    }
    return result;
}

// Compute k-means cost
double computeCost(const std::vector<float>& data, int n, int d,
                   const std::vector<float>& centers, int k) {
    double cost = 0.0;

    #pragma omp parallel for reduction(+:cost)
    for (int i = 0; i < n; i++) {
        float min_dist = std::numeric_limits<float>::max();
        for (int c = 0; c < k; c++) {
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = data[i * d + j] - centers[c * d + j];
                dist += diff * diff;
            }
            min_dist = std::min(min_dist, dist);
        }
        cost += min_dist;
    }
    return cost;
}

// Compute cost for Google format
double computeCostGoogle(const std::vector<std::vector<double>>& data,
                         const std::vector<int>& center_indices) {
    double cost = 0.0;
    int n = data.size();
    int k = center_indices.size();

    #pragma omp parallel for reduction(+:cost)
    for (int i = 0; i < n; i++) {
        double min_dist = std::numeric_limits<double>::max();
        for (int c = 0; c < k; c++) {
            int center_idx = center_indices[c];
            double dist = 0.0;
            for (size_t j = 0; j < data[i].size(); j++) {
                double diff = data[i][j] - data[center_idx][j];
                dist += diff * diff;
            }
            min_dist = std::min(min_dist, dist);
        }
        cost += min_dist;
    }
    return cost;
}

int main(int argc, char** argv) {
    // Configuration
    std::string data_file = "../embeddings/text/imdb_embeddings.npy";
    std::vector<int> k_values = {10, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000};
    const int random_seed = 42;

    // Allow custom data file from command line
    if (argc > 1) {
        data_file = argv[1];
    }

    // RejectionSamplingLSH parameters (from Google 2020 paper)
    const int number_of_trees = 10;
    const double scaling_factor = 4.0;
    const int number_greedy_rounds = 1;  // Must be >= 1 for algorithm to work
    const double boosting_prob_factor = 1.0;

    std::cout << "========================================\n";
    std::cout << "SIX-METHOD K-MEANS++ BENCHMARK\n";
    std::cout << "========================================\n";
    std::cout << "Methods:\n";
    std::cout << "  1. RS-k-means++ (FastLSH)\n";
    std::cout << "  2. RejectionSamplingLSH (Google 2020)\n";
#ifdef HAS_FAISS
    std::cout << "  3. AFK-MC² (MCMC)\n";
#else
    std::cout << "  3. AFK-MC² (SKIPPED - FAISS not available)\n";
#endif
    std::cout << "  4. PRONE (Standard)\n";
    std::cout << "  5. PRONE (Variance-weighted)\n";
    std::cout << "  6. PRONE (Covariance-weighted)\n";
    std::cout << "========================================\n\n";

    // Load dataset
    int n, d;
    std::vector<float> data;
    try {
        data = load_npy(data_file, n, d);
    } catch (const std::exception& e) {
        std::cerr << "Error loading data: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Dataset: " << n << " points, " << d << " dimensions\n\n";

    // Prepare Google format for RejectionSamplingLSH
    std::cout << "Converting to Google format for RejectionSamplingLSH...\n";
    auto data_google = toGoogleFormat(data, n, d);
    std::cout << "Done!\n\n";

    // Open output CSV file
    std::ofstream outfile("../experiments/benchmark_results.csv");
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file!\n";
        return 1;
    }
    outfile << "algorithm,k,runtime_seconds,cost\n";

    // Benchmark each k value
    for (int k : k_values) {
        std::cout << "========================================\n";
        std::cout << "k = " << k << "\n";
        std::cout << "========================================\n\n";

        // 1. RS-k-means++ with FastLSH (our optimized DHHash implementation)
        {
            std::cout << "1. RS-k-means++ (FastLSH)\n";
            std::cout << "   Running..." << std::flush;

            auto start = std::chrono::high_resolution_clock::now();

            rs_kmeans::RSkMeans rskm;
            rskm.preprocess(data, n, d);
            auto result = rskm.cluster(k, 20, "FastLSH", "", random_seed);

            auto end = std::chrono::high_resolution_clock::now();
            double runtime = std::chrono::duration<double>(end - start).count();
            double cost = computeCost(data, n, d, result.first, k);

            std::cout << " Done!\n";
            std::cout << "   Runtime: " << std::fixed << std::setprecision(6) << runtime << " seconds\n";
            std::cout << "   Cost: " << std::fixed << std::setprecision(2) << cost << "\n\n";

            outfile << "RS-k-means++ (FastLSH)," << k << "," << runtime << "," << cost << "\n";
            outfile.flush();
        }

        // 2. RejectionSamplingLSH (Google 2020 multi-tree implementation)
        {
            std::cout << "2. RejectionSamplingLSH (Google 2020)\n";
            std::cout << "   Running..." << std::flush;

            auto start = std::chrono::high_resolution_clock::now();

            fast_k_means::RejectionSamplingLSH rslsh;
            rslsh.RunAlgorithm(data_google, k, number_of_trees, scaling_factor,
                              number_greedy_rounds, boosting_prob_factor);

            auto end = std::chrono::high_resolution_clock::now();
            double runtime = std::chrono::duration<double>(end - start).count();
            double cost = computeCostGoogle(data_google, rslsh.centers);

            std::cout << " Done!\n";
            std::cout << "   Runtime: " << std::fixed << std::setprecision(6) << runtime << " seconds\n";
            std::cout << "   Cost: " << std::fixed << std::setprecision(2) << cost << "\n\n";

            outfile << "RejectionSamplingLSH," << k << "," << runtime << "," << cost << "\n";
            outfile.flush();
        }

#ifdef HAS_FAISS
        // 3. AFK-MC² (MCMC-based sampling)
        // Skip for large k values (too slow)
        if (k <= 500) {
            std::cout << "3. AFK-MC² (MCMC)\n";
            std::cout << "   Running..." << std::flush;

            auto start = std::chrono::high_resolution_clock::now();

            rs_kmeans::AFKMC2 afk;
            afk.preprocess(data, n, d);
            int m = std::max(200, k * 2);  // Markov chain length
            auto result = afk.cluster(k, m, "Flat", "", random_seed);

            auto end = std::chrono::high_resolution_clock::now();
            double runtime = std::chrono::duration<double>(end - start).count();
            double cost = computeCost(data, n, d, result.first, k);

            std::cout << " Done!\n";
            std::cout << "   Runtime: " << std::fixed << std::setprecision(6) << runtime << " seconds\n";
            std::cout << "   Cost: " << std::fixed << std::setprecision(2) << cost << "\n\n";

            outfile << "AFK-MC²," << k << "," << runtime << "," << cost << "\n";
            outfile.flush();
        } else {
            std::cout << "3. AFK-MC² (MCMC)\n";
            std::cout << "   Skipped (too slow for k > 500)\n\n";
        }
#endif

        // 4. PRONE (Standard Gaussian projection)
        {
            std::cout << "4. PRONE (Standard)\n";
            std::cout << "   Running..." << std::flush;

            auto start = std::chrono::high_resolution_clock::now();

            rs_kmeans::PRONE prone(rs_kmeans::ProjectionType::STANDARD);
            prone.preprocess(data, n, d);
            auto result = prone.cluster(k, random_seed);

            auto end = std::chrono::high_resolution_clock::now();
            double runtime = std::chrono::duration<double>(end - start).count();
            double cost = computeCost(data, n, d, result.first, k);

            std::cout << " Done!\n";
            std::cout << "   Runtime: " << std::fixed << std::setprecision(6) << runtime << " seconds\n";
            std::cout << "   Cost: " << std::fixed << std::setprecision(2) << cost << "\n\n";

            outfile << "PRONE (Standard)," << k << "," << runtime << "," << cost << "\n";
            outfile.flush();
        }

        // 5. PRONE (Variance-weighted projection)
        {
            std::cout << "5. PRONE (Variance-weighted)\n";
            std::cout << "   Running..." << std::flush;

            auto start = std::chrono::high_resolution_clock::now();

            rs_kmeans::PRONE prone(rs_kmeans::ProjectionType::VARIANCE_WEIGHTED);
            prone.preprocess(data, n, d);
            auto result = prone.cluster(k, random_seed);

            auto end = std::chrono::high_resolution_clock::now();
            double runtime = std::chrono::duration<double>(end - start).count();
            double cost = computeCost(data, n, d, result.first, k);

            std::cout << " Done!\n";
            std::cout << "   Runtime: " << std::fixed << std::setprecision(6) << runtime << " seconds\n";
            std::cout << "   Cost: " << std::fixed << std::setprecision(2) << cost << "\n\n";

            outfile << "PRONE (Variance)," << k << "," << runtime << "," << cost << "\n";
            outfile.flush();
        }

        // 6. PRONE (Covariance-weighted projection)
        {
            std::cout << "6. PRONE (Covariance-weighted)\n";
            std::cout << "   Running..." << std::flush;

            auto start = std::chrono::high_resolution_clock::now();

            rs_kmeans::PRONE prone(rs_kmeans::ProjectionType::COVARIANCE);
            prone.preprocess(data, n, d);
            auto result = prone.cluster(k, random_seed);

            auto end = std::chrono::high_resolution_clock::now();
            double runtime = std::chrono::duration<double>(end - start).count();
            double cost = computeCost(data, n, d, result.first, k);

            std::cout << " Done!\n";
            std::cout << "   Runtime: " << std::fixed << std::setprecision(6) << runtime << " seconds\n";
            std::cout << "   Cost: " << std::fixed << std::setprecision(2) << cost << "\n\n";

            outfile << "PRONE (Covariance)," << k << "," << runtime << "," << cost << "\n";
            outfile.flush();
        }
    }

    outfile.close();

    std::cout << "========================================\n";
    std::cout << "BENCHMARK COMPLETE!\n";
    std::cout << "Results written to: ../experiments/benchmark_results.csv\n";
    std::cout << "Run: python3 ../experiments/plot_benchmark.py\n";
    std::cout << "========================================\n";

    return 0;
}
