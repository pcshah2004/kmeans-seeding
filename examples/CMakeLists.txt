# CMakeLists.txt for examples

cmake_minimum_required(VERSION 3.15)

# Basic usage example
add_executable(basic_usage basic_usage.cpp)

# Link to the kmeans_seeding_core library
target_link_libraries(basic_usage PRIVATE kmeans_seeding_core)

# Include directories from kmeans_seeding_core
target_include_directories(basic_usage PRIVATE
    ${CMAKE_SOURCE_DIR}/cpp/include
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(basic_usage PRIVATE OpenMP::OpenMP_CXX)
endif()

# Add HAS_FAISS definition and link FAISS if available
if(faiss_FOUND)
    target_compile_definitions(basic_usage PRIVATE HAS_FAISS)
    target_link_libraries(basic_usage PRIVATE faiss)
endif()

# Install examples
install(TARGETS basic_usage
    RUNTIME DESTINATION bin/examples
)
