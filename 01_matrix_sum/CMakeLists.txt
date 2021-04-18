cmake_minimum_required(VERSION 3.14)

set(TARGET_NAME "matrix_sum")

find_package(MPI)

file(GLOB_RECURSE TARGET_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE TARGET_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${TARGET_NAME} ${TARGET_SRC} ${TARGET_HEADERS})

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(MPI_FOUND)
    target_include_directories(${TARGET_NAME} PUBLIC ${MPI_INCLUDE_PATH})
endif()

target_link_libraries(${TARGET_NAME} PUBLIC gtest gtest_main)

gtest_discover_tests(${TARGET_NAME})
