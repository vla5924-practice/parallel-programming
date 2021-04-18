cmake_minimum_required(VERSION 3.14)

project(ParallelProgramming LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/archive)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(ENABLE_01 "Enables 01" OFF)
option(ENABLE_02 "Enables 02" OFF)
option(ENABLE_03 "Enables 03" OFF)
option(ENABLE_04 "Enables 04" OFF)
option(ENABLE_05 "Enables 05" OFF)
option(ENABLE_06 "Enables 06" OFF)
option(ENABLE_07 "Enables 07" OFF)

if(WIN32)
    set(CMAKE_CXX_FLAGS_DEBUG "/MTd /Z7 /Od")
    option(gtest_force_shared_crt "" TRUE)
endif()

add_subdirectory(thirdparty)

enable_testing()
include(GoogleTest)

if(WIN32)
    include_directories("${CMAKE_SOURCE_DIR}/thirdparty/gtest-mpi/windows")
else()
    include_directories("${CMAKE_SOURCE_DIR}/thirdparty/gtest-mpi/linux")
endif()

if(ENABLE_01)
    add_subdirectory(01_matrix_sum)
endif()

if(ENABLE_02)
    add_subdirectory(02_broadcast)
endif()

if(ENABLE_03)
    add_subdirectory(03_shell_sort_batcher_merge)
endif()

if(ENABLE_04)
    add_subdirectory(04_simpson_method_seq)
endif()

if(ENABLE_05)
    add_subdirectory(05_simpson_method_omp)
endif()

if(ENABLE_06)
    add_subdirectory(06_simpson_method_tbb)
endif()

if(ENABLE_07)
    add_subdirectory(07_simpson_method_std)
endif()
