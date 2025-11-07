# CMakeLists.txt for examples

cmake_minimum_required(VERSION 3.15)

# Find the kmeans_seeding library
find_package(OpenMP)
find_package(FAISS QUIET)

# Basic usage example
add_executable(basic_usage basic_usage.cpp)
target_link_libraries(basic_usage PRIVATE kmeans_seeding_core)

if(OpenMP_CXX_FOUND)
    target_link_libraries(basic_usage PRIVATE OpenMP::OpenMP_CXX)
endif()

if(FAISS_FOUND)
    target_compile_definitions(basic_usage PRIVATE HAS_FAISS)
    target_link_libraries(basic_usage PRIVATE ${FAISS_LIBRARIES})
endif()

# Install examples
install(TARGETS basic_usage
    RUNTIME DESTINATION bin/examples
)
