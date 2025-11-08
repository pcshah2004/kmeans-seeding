cmake_minimum_required(VERSION 3.15...3.27)
project(kmeans_seeding VERSION 0.1.0 LANGUAGES CXX)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# Aggressive release optimizations
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")         # Aggressive math optimizations
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops")      # Unroll loops
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ftree-vectorize")    # Auto-vectorization
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-signed-zeros")   # Assume no signed zeros
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-trapping-math")  # No FP exceptions
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fassociative-math")  # Allow reassociation
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -freciprocal-math")   # Use reciprocal approximations

# Platform-specific SIMD flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx2 -mfma")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=native")
endif()

# Link-time optimization
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/kmeans_seeding)

# Add HAS_FAISS definition if FAISS was found by parent
if(faiss_FOUND)
    add_compile_definitions(HAS_FAISS)
    set(FAISS_LIBRARIES faiss)
else()
    set(FAISS_LIBRARIES "")
endif()

# Source files
set(CORE_SOURCES
    src/kmeanspp_seeding.cc
    src/lsh.cc
    src/rejection_sampling_lsh.cc
    src/random_handler.cc
    src/preprocess_input_points.cc
    src/tree_embedding.cc
    src/compute_cost.cc
    src/multi_tree_clustering.cc
    src/single_tree_clustering.cc
)

# Algorithm sources - some require FAISS
set(ALGORITHM_SOURCES
    src/fast_lsh.cpp
    src/prone.cpp
    src/simd_utils.cpp
    src/lightweight.cpp
)

# Add FAISS-dependent algorithms only if FAISS is available
if(faiss_FOUND)
    list(APPEND ALGORITHM_SOURCES
        src/rs_kmeans.cpp
        src/afkmc2.cpp
    )
endif()

# Core library (without Python bindings)
add_library(kmeans_seeding_core STATIC
    ${CORE_SOURCES}
    ${ALGORITHM_SOURCES}
)

target_include_directories(kmeans_seeding_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(kmeans_seeding_core PUBLIC
    ${FAISS_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(kmeans_seeding_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Pure C++ library - Python bindings removed for simplicity

# Installation rules
install(TARGETS kmeans_seeding_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/kmeans_seeding
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Testing (optional)
option(BUILD_TESTS "Build C++ tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Summary
message(STATUS "")
message(STATUS "========================================")
message(STATUS "kmeans-seeding Configuration Summary")
message(STATUS "========================================")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenMP:         ${OpenMP_CXX_FOUND}")
message(STATUS "  FAISS:          ${faiss_FOUND}")
message(STATUS "  Python:         ${BUILD_PYTHON}")
if(BUILD_PYTHON AND pybind11_FOUND)
message(STATUS "  pybind11:       ${pybind11_VERSION}")
endif()
message(STATUS "  Tests:          ${BUILD_TESTS}")
message(STATUS "========================================")
message(STATUS "")
