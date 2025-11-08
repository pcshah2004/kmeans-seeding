/**
 * Benchmark with Lightweight Coreset
 *
 * Benchmarks k-means++ initialization algorithms including Lightweight coreset:
 * 1. RS-k-means++ (FastLSH)
 * 2. RejectionSamplingLSH (Google 2020)
 * 3. AFK-MC² (MCMC)
 * 4. PRONE (boosted α=0.001)
 * 5. PRONE (boosted α=0.01)
 * 6. PRONE (boosted α=0.1)
 * 7. Lightweight Coreset (KDD 2018)
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
#include "kmeans_seeding/lightweight.hpp"
#include "prone_boosted.hpp"

#ifdef HAS_FAISS
#include "kmeans_seeding/afkmc2.hpp"
#endif

// Load numpy array from .npy file
std::vector<float> load_npy(const std::string& filename, int& n, int& d) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    char magic[6];
    file.read(magic, 6);

    uint8_t major_version, minor_version;
    file.read(reinterpret_cast<char*>(&major_version), 1);
    file.read(reinterpret_cast<char*>(&minor_version), 1);

    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);

    std::string header(header_len, ' ');
    file.read(&header[0], header_len);

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

    std::vector<float> data(n * d);
    file.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));

    std::cout << "Loaded " << n << " x " << d << " array from " << filename << std::endl;
    return data;
}

// Convert flat data to vector<vector<double>> format
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
    std::vector<int> k_values = {10, 50, 100, 200, 500, 1000};
    const int random_seed = 42;

    if (argc > 1) {
        data_file = argv[1];
    }

    // RejectionSamplingLSH parameters
    const int number_of_trees = 10;
    const double scaling_factor = 4.0;
    const int number_greedy_rounds = 1;
    const double boosting_prob_factor = 1.0;

    std::cout << "========================================\n";
    std::cout << "K-MEANS++ BENCHMARK WITH LIGHTWEIGHT\n";
    std::cout << "========================================\n";
    std::cout << "Methods:\n";
    std::cout << "  1. RS-k-means++ (FastLSH)\n";
    std::cout << "  2. RejectionSamplingLSH\n";
#ifdef HAS_FAISS
    std::cout << "  3. AFK-MC²\n";
#endif
    std::cout << "  4-6. PRONE (boosted)\n";
    std::cout << "  7. Lightweight Coreset\n";
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

    // Prepare Google format
    std::cout << "Converting to Google format...\n";
    auto data_google = toGoogleFormat(data, n, d);
    std::cout << "Done!\n\n";

    // Open output CSV file
    std::ofstream outfile("benchmark_results_prone_boosted.csv");
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

        // 1. RS-k-means++ (FastLSH)
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

        // 2. RejectionSamplingLSH
        {
            std::cout << "2. RejectionSamplingLSH\n";
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
        // 3. AFK-MC²
        if (k <= 500) {
            std::cout << "3. AFK-MC²\n";
            std::cout << "   Running..." << std::flush;

            auto start = std::chrono::high_resolution_clock::now();

            rs_kmeans::AFKMC2 afk;
            afk.preprocess(data, n, d);
            int m = std::max(200, k * 2);
            auto result = afk.cluster(k, m, "Flat", "", random_seed);

            auto end = std::chrono::high_resolution_clock::now();
            double runtime = std::chrono::duration<double>(end - start).count();
            double cost = computeCost(data, n, d, result.first, k);

            std::cout << " Done!\n";
            std::cout << "   Runtime: " << std::fixed << std::setprecision(6) << runtime << " seconds\n";
            std::cout << "   Cost: " << std::fixed << std::setprecision(2) << cost << "\n\n";

            outfile << "AFK-MC²," << k << "," << runtime << "," << cost << "\n";
            outfile.flush();
        }
#endif

        // 4-6. PRONE (boosted) variants
        std::vector<double> alphas = {0.001, 0.01, 0.1};
        for (double alpha : alphas) {
            std::cout << "PRONE (boosted α=" << alpha << ")\n";
            std::cout << "   Running..." << std::flush;

            auto start = std::chrono::high_resolution_clock::now();

            auto result = prone_boosted::run_prone_boosted(data, n, d, k, alpha, random_seed);

            auto end = std::chrono::high_resolution_clock::now();
            double runtime = std::chrono::duration<double>(end - start).count();
            double cost = computeCost(data, n, d, result.first, k);

            std::cout << " Done!\n";
            std::cout << "   Runtime: " << std::fixed << std::setprecision(6) << runtime << " seconds\n";
            std::cout << "   Cost: " << std::fixed << std::setprecision(2) << cost << "\n\n";

            outfile << "PRONE (boosted α=" << alpha << ")," << k << "," << runtime << "," << cost << "\n";
            outfile.flush();
        }

        // 7. Lightweight Coreset
        {
            std::cout << "7. Lightweight Coreset\n";
            std::cout << "   Running..." << std::flush;

            auto start = std::chrono::high_resolution_clock::now();

            rs_kmeans::Lightweight lwcs;
            lwcs.preprocess(data, n, d);
            // Use default coreset size (automatic based on k, d, epsilon)
            auto result = lwcs.cluster(k, -1, 0.1, random_seed);

            auto end = std::chrono::high_resolution_clock::now();
            double runtime = std::chrono::duration<double>(end - start).count();
            double cost = computeCost(data, n, d, result.first, k);

            std::cout << " Done!\n";
            std::cout << "   Runtime: " << std::fixed << std::setprecision(6) << runtime << " seconds\n";
            std::cout << "   Cost: " << std::fixed << std::setprecision(2) << cost << "\n\n";

            outfile << "Lightweight," << k << "," << runtime << "," << cost << "\n";
            outfile.flush();
        }
    }

    outfile.close();

    std::cout << "========================================\n";
    std::cout << "BENCHMARK COMPLETE!\n";
    std::cout << "Results written to: benchmark_results_prone_boosted.csv\n";
    std::cout << "========================================\n";

    return 0;
}
