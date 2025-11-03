# Top-level CMakeLists.txt for kmeans-seeding
# This delegates to the cpp/ subdirectory

cmake_minimum_required(VERSION 3.15)
project(kmeans_seeding_toplevel VERSION 0.1.0)

# Add the cpp subdirectory
add_subdirectory(cpp)
