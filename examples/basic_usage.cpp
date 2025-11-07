/**
 * Basic usage example for kmeans-seeding C++ library
 *
 * This example demonstrates how to use the different k-means++ initialization algorithms:
 * - Standard k-means++
 * - RS-k-means++ with FastLSH
 * - AFK-MC²
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "kmeans_seeding/kmeanspp.hpp"
#include "kmeans_seeding/rs_kmeans.hpp"
#include "kmeans_seeding/afkmc2.hpp"

using namespace kmeans_seeding;

// Generate random dataset
std::vector<std::vector<double>> generateRandomData(int n, int d, int seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<std::vector<double>> data(n, std::vector<double>(d));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            data[i][j] = dist(gen);
        }
    }
    return data;
}

// Compute k-means cost
double computeCost(const std::vector<std::vector<double>>& data,
                   const std::vector<std::vector<double>>& centers) {
    double cost = 0.0;
    for (const auto& point : data) {
        double min_dist = std::numeric_limits<double>::max();
        for (const auto& center : centers) {
            double dist = 0.0;
            for (size_t j = 0; j < point.size(); j++) {
                double diff = point[j] - center[j];
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

    // 1. Standard k-means++
    {
        std::cout << "\n1. Standard k-means++\n";
        std::cout << "   Running...";
        std::cout.flush();

        auto start = std::chrono::high_resolution_clock::now();
        KMeansPP kmpp;
        auto centers = kmpp.initialize(data, k, 42);
        auto end = std::chrono::high_resolution_clock::now();

        double runtime = std::chrono::duration<double>(end - start).count();
        double cost = computeCost(data, centers);

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
        RSKMeans rskm;
        rskm.set_index_type("FastLSH");
        auto centers = rskm.initialize(data, k, 42);
        auto end = std::chrono::high_resolution_clock::now();

        double runtime = std::chrono::duration<double>(end - start).count();
        double cost = computeCost(data, centers);

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
        RSKMeans rskm;
        rskm.set_index_type("IVFFlat");
        auto centers = rskm.initialize(data, k, 42);
        auto end = std::chrono::high_resolution_clock::now();

        double runtime = std::chrono::duration<double>(end - start).count();
        double cost = computeCost(data, centers);

        std::cout << " Done!\n";
        std::cout << "   Runtime: " << runtime << " seconds\n";
        std::cout << "   Cost: " << cost << "\n";
    }

    // 4. AFK-MC²
    {
        std::cout << "\n4. AFK-MC²\n";
        std::cout << "   Running...";
        std::cout.flush();

        auto start = std::chrono::high_resolution_clock::now();
        AFKMC2 afk;
        int m = k * 10;  // 10x oversampling
        auto centers = afk.initialize(data, k, m, 42);
        auto end = std::chrono::high_resolution_clock::now();

        double runtime = std::chrono::duration<double>(end - start).count();
        double cost = computeCost(data, centers);

        std::cout << " Done!\n";
        std::cout << "   Runtime: " << runtime << " seconds\n";
        std::cout << "   Cost: " << cost << "\n";
    }
#else
    std::cout << "\nNote: FAISS not available. RS-k-means++ (FAISS) and AFK-MC² skipped.\n";
    std::cout << "      Install FAISS to use these algorithms.\n";
#endif

    std::cout << "\n===================================\n";
    std::cout << "Benchmark complete!\n";

    return 0;
}
