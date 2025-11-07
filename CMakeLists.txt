# Top-level CMakeLists.txt for kmeans-seeding
# This delegates to the cpp/ subdirectory

cmake_minimum_required(VERSION 3.15)
project(kmeans_seeding_toplevel VERSION 0.1.0)

# Set C++ standard globally
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies globally so they're available to all subdirectories
# On macOS with Homebrew, help CMake find OpenMP
if(APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
    message(WARNING "OpenMP not found. Building without OpenMP support.")
endif()

# FAISS (optional but recommended)
find_package(faiss QUIET)
if(faiss_FOUND)
    message(STATUS "FAISS found: ${faiss_VERSION}")
    set(HAS_FAISS TRUE)
else()
    message(WARNING "FAISS not found. Some features will be disabled.")
    message(WARNING "Install FAISS: conda install -c pytorch faiss-cpu")
    set(HAS_FAISS FALSE)
endif()

# Add the cpp subdirectory
add_subdirectory(cpp)

# Add examples (optional)
option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
