// Copyright 2020 Vlasov Maksim
#include <mpi.h>
#include <cmath>
#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
#include "shell_sort_batcher_merge.h"


Vector createRandomVector(int elements_count) {
    std::random_device rd;
    std::mt19937 generator(rd());
    Vector result(elements_count);
    for (int& elem : result)
        elem = static_cast<int>(generator() % 100u);
    return result;
}

Vector shellSort(Vector arr) {
    auto size = arr.size();
    for (auto step = size / 2; step > 0; step /= 2) {
        for (auto i = step; i < size; i++) {
            for (auto j = i; j >= step && arr[j] < arr[j - step]; j -= step) {
                std::swap(arr[j], arr[j - step]);
            }
        }
    }
    return arr;
}

namespace BatcherMerge {
    using Comparator = std::pair<int, int>;
    std::vector<Comparator> comparators;

    Vector join(const Vector& first, const Vector& second) {
        Vector temp(0);
        temp.reserve(first.size() + second.size());
        temp.insert(temp.end(), first.begin(), first.end());
        temp.insert(temp.end(), second.begin(), second.end());
        return temp;
    }

    void mergeNetwork(const Vector& ranks_up, const Vector& ranks_down) {
        size_t size = ranks_up.size() + ranks_down.size();
        if (size == 1)
            return;
        if (size == 2) {
            comparators.emplace_back(ranks_up.front(), ranks_down.front());
            return;
        }

        Vector ranks_up_odd, ranks_up_even;
        for (size_t i = 0; i < ranks_up.size(); i++) {
            if (i % 2 == 0)
                ranks_up_odd.push_back(ranks_up[i]);
            else
                ranks_up_even.push_back(ranks_up[i]);
        }
        Vector ranks_down_odd, ranks_down_even;
        for (size_t i = 0; i < ranks_down.size(); i++) {
            if (i % 2 == 0)
                ranks_down_odd.push_back(ranks_down[i]);
            else
                ranks_down_even.push_back(ranks_down[i]);
        }

        mergeNetwork(ranks_up_odd, ranks_down_odd);
        mergeNetwork(ranks_up_even, ranks_down_even);

        Vector temp_comp = join(ranks_up, ranks_down);
        for (size_t i = 1; i < temp_comp.size() - 1; i += 2)
            comparators.emplace_back(temp_comp[i], temp_comp[i + 1]);
    }

    void buildNetwork(const Vector& ranks) {
        size_t size = ranks.size();
        if (size < 2)
            return;

        size_t ranks_up_size = size / 2;
        Vector ranks_up{ ranks.begin(), ranks.begin() + ranks_up_size };
        Vector ranks_down{ ranks.begin() + ranks_up_size, ranks.end() };

        buildNetwork(ranks_up);
        buildNetwork(ranks_down);
        mergeNetwork(ranks_up, ranks_down);
    }

    void check(Vector* arr) {
        size_t i = 0;
        for (i = arr->size() - 1; i > 0; i--)
            if (arr->at(i - 1) > arr->at(i))
                break;
        while (i > 0 && arr->at(i - 1) > arr->at(i)) {
            std::swap(arr->at(i - 1), arr->at(i));
            i--;
        }
    }

    Vector parallelSort(Vector arr, std::function<Vector(Vector)> sort_func) {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int arr_size = static_cast<int>(arr.size());
        if (arr_size < 2)
            return arr;
        if (arr_size <= size)
            return sort_func(arr);

        // int extra_size = arr_size % size;
        int extra_size = static_cast<int>(std::pow(2, std::ceil(std::log2(arr_size + arr_size % size)))) - arr_size;
        arr_size += extra_size;
        arr.resize(arr_size, std::numeric_limits<int>::max());
        int part_size = arr_size / size;

        Vector ranks(size);
        std::iota(ranks.begin(), ranks.end(), 0);
        buildNetwork(ranks);

        Vector part(part_size), part_curr(part_size), part_temp(part_size);
        MPI_Scatter(arr.data(), part_size, MPI_INT, part.data(), part_size, MPI_INT, 0, MPI_COMM_WORLD);
        part = sort_func(part);

        for (const auto& comp : comparators) {
            if (rank == comp.first) {
                MPI_Send(part.data(), part_size, MPI_INT, comp.second, 0, MPI_COMM_WORLD);
                MPI_Recv(part_curr.data(), part_size, MPI_INT, comp.second, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int i = 0, i_curr = 0, i_temp = 0; i_temp < part_size; i_temp++) {
                    int value = part[i];
                    int value_curr = part_curr[i_curr];
                    if (value < value_curr) {
                        part_temp[i_temp] = value;
                        i++;
                    } else {
                        part_temp[i_temp] = value_curr;
                        i_curr++;
                    }
                }
                std::swap(part, part_temp);
            } else if (rank == comp.second) {
                MPI_Recv(part_curr.data(), part_size, MPI_INT, comp.first, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(part.data(), part_size, MPI_INT, comp.first, 0, MPI_COMM_WORLD);
                size_t i_start = part_size - 1;
                for (int i = i_start, i_curr = i_start, i_temp = part_size; i_temp > 0; i_temp--) {
                    int value = part[i];
                    int value_curr = part_curr[i_curr];
                    if (value > value_curr) {
                        part_temp[i_temp - 1] = value;
                        i--;
                    } else {
                        part_temp[i_temp - 1] = value_curr;
                        i_curr--;
                    }
                }
                std::swap(part, part_temp);
            }
        }
        MPI_Gather(part.data(), part_size, MPI_INT, arr.data(), part_size, MPI_INT, 0, MPI_COMM_WORLD);
        arr_size -= extra_size;
        arr.resize(arr_size);
        if (rank == 0)
            check(&arr);
        return arr;
    }
}  // namespace BatcherMerge
