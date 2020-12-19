// Copyright 2020 Vlasov Maksim
#include <mpi.h>
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "shell_sort_batcher_merge.h"

TEST(Parallel_Shell_Sort_Batcher_Merge_MPI, Size_10) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> arr(10);
    if (rank == 0)
        arr = createRandomVector(10);
    auto check_arr = BatcherMerge::parallelSort(arr, shellSort);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        auto exp_arr = shellSort(arr);
        ASSERT_EQ(exp_arr, check_arr);
    }
}

TEST(Parallel_Shell_Sort_Batcher_Merge_MPI, Size_15) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> arr(15);
    if (rank == 0)
        arr = createRandomVector(15);
    auto check_arr = BatcherMerge::parallelSort(arr, shellSort);
    if (rank == 0) {
        auto exp_arr = shellSort(arr);
        ASSERT_EQ(exp_arr, check_arr);
    }
}

TEST(Parallel_Shell_Sort_Batcher_Merge_MPI, Size_100) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> arr(100);
    if (rank == 0)
        arr = createRandomVector(100);
    auto check_arr = BatcherMerge::parallelSort(arr, shellSort);
    if (rank == 0) {
        auto exp_arr = shellSort(arr);
        ASSERT_EQ(exp_arr, check_arr);
    }
}

TEST(Parallel_Shell_Sort_Batcher_Merge_MPI, Size_500) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> arr(500);
    if (rank == 0)
        arr = createRandomVector(500);
    auto check_arr = BatcherMerge::parallelSort(arr, shellSort);
    if (rank == 0) {
        auto exp_arr = shellSort(arr);
        ASSERT_EQ(exp_arr, check_arr);
    }
}

TEST(Parallel_Shell_Sort_Batcher_Merge_MPI, Size_1000) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> arr(1000, 0);
    if (rank == 0)
        arr = createRandomVector(1000);
    auto check_arr = BatcherMerge::parallelSort(arr, shellSort);
    if (rank == 0) {
        auto exp_arr = shellSort(arr);
        ASSERT_EQ(exp_arr, check_arr);
    }
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
