cmake_minimum_required(VERSION 3.14)
project(cls_tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

# Download Catch2 if not found
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.0
)
FetchContent_MakeAvailable(Catch2)

# Add your source files (excluding main.cpp)
file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/src/src/ml.cpp"
    "${CMAKE_SOURCE_DIR}/src/src/network.cpp"
    "${CMAKE_SOURCE_DIR}/src/src/builder.cpp"
    "${CMAKE_SOURCE_DIR}/src/src/npz_reader.cpp"
)

# Create test executable
add_executable(${PROJECT_NAME}
    "${CMAKE_SOURCE_DIR}/src/tests/test_suite.cpp"
    ${SOURCE_FILES}
)

# Add include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src/inc
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    Catch2::Catch2WithMain
    Threads::Threads
    ZLIB::ZLIB
    lapack
    blas
    gomp
    z
)

# Set compile options (consistent with Makefile)
target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall -Wextra -Wno-unused-parameter
    -Wno-deprecated-declarations -Wno-unused-function
)

# If on macOS, add OpenMP flags
if(APPLE)
    find_path(LIBOMP_INCLUDE_DIR omp.h
        PATHS /usr/local/opt/libomp/include
        NO_DEFAULT_PATH
    )
    find_library(LIBOMP_LIBRARY NAMES omp libomp
        PATHS /usr/local/opt/libomp/lib
        NO_DEFAULT_PATH
    )
    if(LIBOMP_INCLUDE_DIR AND LIBOMP_LIBRARY)
        target_compile_options(${PROJECT_NAME} PRIVATE
            -Xpreprocessor -fopenmp
            -I${LIBOMP_INCLUDE_DIR}
        )
        target_link_libraries(${PROJECT_NAME} PRIVATE
            ${LIBOMP_LIBRARY}
        )
    endif()
endif()
