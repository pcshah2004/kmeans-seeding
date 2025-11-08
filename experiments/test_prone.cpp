#include <iostream>
#include <vector>
#include <random>
#include "../cpp/include/kmeans_seeding/prone.hpp"

int main() {
    // Create simple test data: 100 points in 10 dimensions
    int n = 100, d = 10, k = 5;
    std::vector<float> data(n * d);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < n * d; i++) {
        data[i] = dist(gen);
    }
    
    // Test PRONE with standard projection
    rs_kmeans::PRONE prone(rs_kmeans::ProjectionType::STANDARD);
    prone.preprocess(data, n, d);
    
    auto [centers, assignments] = prone.cluster(k, 42);
    
    std::cout << "PRONE test successful!\n";
    std::cout << "Generated " << centers.size() / d << " centers\n";
    std::cout << "Generated " << assignments.size() << " assignments\n";
    std::cout << "First few assignments: ";
    for (int i = 0; i < std::min(10, (int)assignments.size()); i++) {
        std::cout << assignments[i] << " ";
    }
    std::cout << "\n";
    
    return 0;
}
