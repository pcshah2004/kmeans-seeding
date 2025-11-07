# CMakeLists.txt for examples

cmake_minimum_required(VERSION 3.15)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Basic usage example
add_executable(basic_usage basic_usage.cpp)

# Link to the kmeans_seeding_core library
target_link_libraries(basic_usage PRIVATE kmeans_seeding_core)

# Include directories from kmeans_seeding_core
target_include_directories(basic_usage PRIVATE
    ${CMAKE_SOURCE_DIR}/cpp/include
)

# Link OpenMP if available
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(basic_usage PRIVATE OpenMP::OpenMP_CXX)
endif()

# Link FAISS if available
if(faiss_FOUND OR FAISS_FOUND)
    target_compile_definitions(basic_usage PRIVATE HAS_FAISS)
    if(TARGET faiss)
        target_link_libraries(basic_usage PRIVATE faiss)
    elseif(FAISS_LIBRARIES)
        target_link_libraries(basic_usage PRIVATE ${FAISS_LIBRARIES})
    endif()
endif()

# Install examples
install(TARGETS basic_usage
    RUNTIME DESTINATION bin/examples
)
