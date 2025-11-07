/**
 * Basic usage example for kmeans-seeding C++ library
 *
 * This example demonstrates how to use the RS-k-means++ algorithm
 * which implements rejection sampling-based k-means++ initialization.
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "kmeans_seeding/rs_kmeans.hpp"
#include "kmeans_seeding/kmeanspp_seeding.h"

// Generate random dataset
std::vector<float> generateRandomData(int n, int d, int seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> data(n * d);
    for (int i = 0; i < n * d; i++) {
        data[i] = dist(gen);
    }
    return data;
}

// Convert flat data to vector<vector<double>> format for Google's k-means++
std::vector<std::vector<double>> toGoogleFormat(const std::vector<float>& data, int n, int d) {
    std::vector<std::vector<double>> result(n, std::vector<double>(d));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            result[i][j] = static_cast<double>(data[i * d + j]);
        }
    }
    return result;
}

// Compute k-means cost given centers (flat format)
double computeCost(const std::vector<float>& data, int n, int d,
                   const std::vector<float>& centers, int k) {
    double cost = 0.0;
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

int main() {
    // Configuration
    const int n = 10000;  // number of points
    const int d = 100;    // dimensionality
    const int k = 50;     // number of clusters

    std::cout << "K-means++ Initialization Benchmark\n";
    std::cout << "===================================\n";
    std::cout << "Dataset: " << n << " points, " << d << " dimensions\n";
    std::cout << "Clusters: " << k << "\n\n";

    // Generate data
    std::cout << "Generating random data...\n";
    auto data = generateRandomData(n, d);

    // 1. Standard k-means++ (Google's implementation)
    {
        std::cout << "\n1. Standard k-means++\n";
        std::cout << "   Running...";
        std::cout.flush();

        auto data_google = toGoogleFormat(data, n, d);
        auto start = std::chrono::high_resolution_clock::now();

        fast_k_means::KMeansPPSeeding kmpp;
        kmpp.RunAlgorithm(data_google, k, 0);  // 0 = no greedy rounds

        auto end = std::chrono::high_resolution_clock::now();
        double runtime = std::chrono::duration<double>(end - start).count();

        // Convert centers back to flat format for cost computation
        std::vector<float> centers_flat(k * d);
        for (int i = 0; i < k; i++) {
            int center_idx = kmpp.centers_[i];
            for (int j = 0; j < d; j++) {
                centers_flat[i * d + j] = static_cast<float>(data_google[center_idx][j]);
            }
        }

        double cost = computeCost(data, n, d, centers_flat, k);

        std::cout << " Done!\n";
        std::cout << "   Runtime: " << runtime << " seconds\n";
        std::cout << "   Cost: " << cost << "\n";
    }

    // 2. RS-k-means++ with FastLSH
    {
        std::cout << "\n2. RS-k-means++ (FastLSH)\n";
        std::cout << "   Running...";
        std::cout.flush();

        auto start = std::chrono::high_resolution_clock::now();

        rs_kmeans::RSkMeans rskm;
        rskm.preprocess(data, n, d);
        auto result = rskm.cluster(k, -1, "FastLSH", "", 42);

        auto end = std::chrono::high_resolution_clock::now();
        double runtime = std::chrono::duration<double>(end - start).count();
        double cost = computeCost(data, n, d, result.first, k);

        std::cout << " Done!\n";
        std::cout << "   Runtime: " << runtime << " seconds\n";
        std::cout << "   Cost: " << cost << "\n";
    }

#ifdef HAS_FAISS
    // 3. RS-k-means++ with FAISS IVFFlat
    {
        std::cout << "\n3. RS-k-means++ (FAISS IVFFlat)\n";
        std::cout << "   Running...";
        std::cout.flush();

        auto start = std::chrono::high_resolution_clock::now();

        rs_kmeans::RSkMeans rskm;
        rskm.preprocess(data, n, d);
        auto result = rskm.cluster(k, -1, "IVFFlat", "", 42);

        auto end = std::chrono::high_resolution_clock::now();
        double runtime = std::chrono::duration<double>(end - start).count();
        double cost = computeCost(data, n, d, result.first, k);

        std::cout << " Done!\n";
        std::cout << "   Runtime: " << runtime << " seconds\n";
        std::cout << "   Cost: " << cost << "\n";
    }
#else
    std::cout << "\nNote: FAISS not available. RS-k-means++ (FAISS) skipped.\n";
    std::cout << "      Install FAISS to use FAISS-based algorithms.\n";
#endif

    std::cout << "\n===================================\n";
    std::cout << "Benchmark complete!\n";

    return 0;
}
