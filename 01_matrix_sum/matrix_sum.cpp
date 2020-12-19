// Copyright 2020 Vlasov Maksim
#include <mpi.h>
#include <vector>
#include <random>
#include <numeric>
#include "matrix_sum.h"

std::vector<int> createRandomVector(int elements_count) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::vector<int> result(elements_count);
    for (int& elem : result)
        elem = static_cast<int>(generator() % 100u);
    return result;
}

int calculateSumSequental(const std::vector<int> &vector) {
    return std::accumulate(vector.begin(), vector.end(), 0);
}

int calculateSumParallel(const std::vector<int> &vector, int elements_count) {
    int process_count, process_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    int delta = elements_count / process_count;
    int remain = elements_count - delta * process_count;
    if (process_rank == 0) {
        for (int process_num = 1; process_num < process_count; process_num++)
            MPI_Send(vector.data() + remain + process_num * delta, delta, MPI_INT,
                     process_num, 0, MPI_COMM_WORLD);
    }
    std::vector<int> part_vector(process_rank == 0 ? delta + remain : delta);
    if (process_rank == 0) {
        part_vector = { vector.begin(), vector.begin() + delta + remain };
    } else {
        MPI_Status status;
        MPI_Recv(part_vector.data(), delta, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }
    int sum = 0;
    int part_sum = calculateSumSequental(part_vector);
    MPI_Reduce(&part_sum, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    return sum;
}
