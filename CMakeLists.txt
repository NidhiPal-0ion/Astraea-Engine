cmake_minimum_required(VERSION 3.20)
project(AstraeaEngine VERSION 0.1.0 LANGUAGES CXX)

# Prevent imported targets (GTest/Benchmark) from overriding compiler system paths
set(CMAKE_NO_SYSTEM_FROM_IMPORTED TRUE)

# Enforce strict C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization and performance flags (GNU/Clang and MSVC support)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -O3
        -march=native
        -fno-rtti
        -Wall
        -Wextra
        -Wpedantic
        -Werror
    )
    # Enable LTO/IPO
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported LANGUAGES CXX)
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    endif()
elseif(MSVC)
    add_compile_options(
        /O2
        /GL
        /GR-
        /W4
        /WX
    )
    # Enable MSVC Link Time Code Generation (LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported LANGUAGES CXX)
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    endif()
endif()

# Core header-only engine library
add_library(astraea_engine INTERFACE)
target_include_directories(astraea_engine INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Enable test suite
enable_testing()

# Enable FetchContent for fallback dependencies
include(FetchContent)

# Dynamically locate GTest, fallback to FetchContent
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(STATUS "GTest not found locally. Fetching from GitHub...")
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    # Prevent overriding compiler/linker settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    
    if(NOT TARGET GTest::gtest)
        add_library(GTest::gtest ALIAS gtest)
    endif()
    if(NOT TARGET GTest::gtest_main)
        add_library(GTest::gtest_main ALIAS gtest_main)
    endif()
endif()

# Force CMake to treat GTest as a regular include to prevent breaking compiler system headers
if(TARGET GTest::gtest)
    set_target_properties(GTest::gtest PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "")
endif()
if(TARGET GTest::gtest_main)
    set_target_properties(GTest::gtest_main PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "")
endif()

# Add Unit Tests
add_executable(astraea_tests
    tests/unit_tests.cpp
)
add_executable(order_book_tests
    tests/order_book_tests.cpp
)
target_link_libraries(astraea_tests PRIVATE
    astraea_engine
    GTest::gtest
    GTest::gtest_main
)
target_link_libraries(order_book_tests PRIVATE
    astraea_engine
    GTest::gtest
    GTest::gtest_main
)
gtest_discover_tests(astraea_tests)
gtest_discover_tests(order_book_tests)

# Dynamically locate Google Benchmark, fallback to FetchContent
find_package(benchmark QUIET)
if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found locally. Fetching from GitHub...")
    FetchContent_Declare(
        google_benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_EXCEPTIONS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(google_benchmark)
    
    if(NOT TARGET benchmark::benchmark)
        add_library(benchmark::benchmark ALIAS benchmark)
    endif()
endif()

# Force CMake to treat Google Benchmark as a regular include
if(TARGET benchmark::benchmark)
    set_target_properties(benchmark::benchmark PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "")
endif()

# Add Benchmarks
add_executable(astraea_benchmarks
    benchmarks/benchmarks.cpp
)
add_executable(order_book_benchmarks
    benchmarks/order_book_benchmarks.cpp
)
target_link_libraries(astraea_benchmarks PRIVATE
    astraea_engine
    benchmark::benchmark
)
target_link_libraries(order_book_benchmarks PRIVATE
    astraea_engine
    benchmark::benchmark
)

# Add Astraea TensorRT Compiler Target
find_package(CUDAToolkit QUIET)
find_library(TRT_LIBRARY nvinfer)
find_library(TRT_ONNX_PARSER_LIBRARY nvonnxparser)
find_path(TRT_INCLUDE_DIR NvInfer.h)

if(CUDAToolkit_FOUND AND TRT_LIBRARY AND TRT_ONNX_PARSER_LIBRARY AND TRT_INCLUDE_DIR)
    add_executable(astraea_compiler
        src/analytics/main_compiler.cpp
        src/analytics/TensorRTCompiler.cpp
    )
    target_include_directories(astraea_compiler PRIVATE ${TRT_INCLUDE_DIR} include)
    target_link_libraries(astraea_compiler PRIVATE
        astraea_engine
        CUDA::cudart
        ${TRT_LIBRARY}
        ${TRT_ONNX_PARSER_LIBRARY}
    )

    find_package(Threads REQUIRED)
    add_executable(astraea_runtime
        src/core/EngineRuntime.cpp
        benchmarks/runtime_benchmarks.cpp
    )
    target_include_directories(astraea_runtime PRIVATE ${TRT_INCLUDE_DIR} include)
    target_link_libraries(astraea_runtime PRIVATE
        astraea_engine
        Threads::Threads
        CUDA::cudart
        ${TRT_LIBRARY}
        ${TRT_ONNX_PARSER_LIBRARY}
    )
else()
    message(WARNING "CUDA and/or TensorRT not found. The 'astraea_compiler' and 'astraea_runtime' targets will not be built.")
endif()
