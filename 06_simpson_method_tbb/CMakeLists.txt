cmake_minimum_required(VERSION 3.14)

set(TARGET_NAME "simpson_method_tbb")

if(WIN32)
    include(${CMAKE_SOURCE_DIR}/cmake/TBBGet.cmake)
    tbb_get(TBB_ROOT tbb_root RELEASE_TAG "v2020.3" CONFIG_DIR TBB_DIR)

    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(SYSTEM_BIT "intel64")
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(SYSTEM_BIT "ia32")
    endif()

    file(GLOB TBB_PRE_LIB  "${tbb_root}/bin/${SYSTEM_BIT}/vc14/*.dll")
    file(COPY ${TBB_PRE_LIB} DESTINATION "${CMAKE_BINARY_DIR}/bin")

    find_package(TBB)
else()
    include(${CMAKE_SOURCE_DIR}/cmake/FindTBB.cmake)
endif()

if(TBB_FOUND)
    message(STATUS "Found TBB: TRUE (found version \"${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR}\") ")
    include_directories("${tbb_root}/include")
else()
    message(FATAL_ERRROR "Found TBB: FALSE")
endif()

file(GLOB_RECURSE TARGET_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE TARGET_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${TARGET_NAME} ${TARGET_SRC} ${TARGET_HEADERS})

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${TARGET_NAME} PUBLIC gtest gtest_main)

gtest_discover_tests(${TARGET_NAME})
