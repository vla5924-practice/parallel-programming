// Copyright 2020 Vlasov Maksim
#include <mpi.h>
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "broadcast.h"

namespace pi_calc {
    constexpr double PI25DT = 3.141592653589793238462643;

    constexpr double EPSILON = 1e-7;

    void log(const char* const label, double result, double time) {
        std::cout << label
            << ": pi=" << result
            << ", error=" << fabs(result - PI25DT)
            << ", time=" << time << std::endl;
    }

    inline double integral(double a) {
        return 4. / (1. + a * a);
    }

    double sequential(int iter) {
        double t1, t2;
        t1 = MPI_Wtime();
        double h = 1. / static_cast<double>(iter);
        double sum = 0.;
        for (int i = 1; i <= iter; i++) {
            double x = h * (static_cast<double>(i) - 0.5);
            sum += integral(x);
        }
        double result = h * sum;
        t2 = MPI_Wtime();
        log(__FUNCTION__, result, t2 - t1);
        return result;
    }

    double my_bcast(int iter) {
        int rank, size;
        double t1, t2;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            t1 = MPI_Wtime();
        broadcast(&iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
        double result = 0.;
        double h = 1. / static_cast<double>(iter);
        double sum = 0.;
        for (int i = rank + 1; i <= iter; i += size) {
            double x = h * (static_cast<double>(i) - 0.5);
            sum += integral(x);
        }
        double part = h * sum;
        MPI_Reduce(&part, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            t2 = MPI_Wtime();
            log(__FUNCTION__, result, t2 - t1);
        }
        return result;
    }

    double mpi_bcast(int iter) {
        int rank, size;
        double t1, t2;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            t1 = MPI_Wtime();
        MPI_Bcast(&iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
        double result = 0.;
        double h = 1. / static_cast<double>(iter);
        double sum = 0.;
        for (int i = rank + 1; i <= iter; i += size) {
            double x = h * (static_cast<double>(i) - 0.5);
            sum += integral(x);
        }
        double part = h * sum;
        MPI_Reduce(&part, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            t2 = MPI_Wtime();
            log(__FUNCTION__, result, t2 - t1);
        }
        return result;
    }
}  // namespace pi_calc

TEST(Bcast_Impl_MPI, Vector5_Int) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int size = 5;
    std::vector<int> v(size);
    if (rank == 0)
        v = { 0, 1, 2, 3, 4 };
    broadcast(v.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    ASSERT_EQ(v, std::vector<int>({ 0, 1, 2, 3, 4 }));
}

TEST(Bcast_Impl_MPI, Vector10_Double) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int size = 10;
    std::vector<double> v(size);
    if (rank == 0)
        v = { 0., 1., 2., 3., 4., 5., 6., 7., 8., 9. };
    broadcast(v.data(), size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    ASSERT_EQ(v, std::vector<double>({ 0., 1., 2., 3., 4., 5., 6., 7., 8., 9. }));
}

TEST(Bcast_Impl_MPI, Pi_calculating_problem__Comp_with_sequential) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double parallel = pi_calc::my_bcast(10000000);
    if (rank == 0) {
        ASSERT_LE(fabs(parallel - pi_calc::sequential(10000000)), pi_calc::EPSILON);
    }
}

TEST(Bcast_Impl_MPI, Pi_calculating_problem__Comp_with_vendor) {
    ASSERT_LE(fabs(pi_calc::my_bcast(10000000) - pi_calc::mpi_bcast(10000000)), pi_calc::EPSILON);
}

TEST(Bcast_Impl_MPI, Performance_Vector500) {
    int rank;
    double t1, t2;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int size = 500;
    const int value = 32;
    std::vector<int> v(size);
    if (rank == 0) {
        std::fill(v.begin(), v.end(), value);
        t1 = MPI_Wtime();
    }
    for (int i = 0; i < size; i++)
        broadcast(v.data() + i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        t2 = MPI_Wtime();
        std::cout << "my_bcast: " << (t2 - t1) << std::endl;
    }
    for (int i = 0; i < size; i++)
        if (v[i] != value) {
            FAIL();
        }
    if (rank == 0) {
        t1 = MPI_Wtime();
    }
    for (int i = 0; i < size; i++)
        MPI_Bcast(v.data() + i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        t2 = MPI_Wtime();
        std::cout << "MPI_bcast: " << (t2 - t1) << std::endl;
        SUCCEED();
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
