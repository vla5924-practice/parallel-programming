// Copyright 2021 Vlasov Maksim

#include "simpson_method.h"

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>

static void sumUp(std::vector<double>* accum, const std::vector<double>& add) {
    assert(accum->size() == add.size());
    for (size_t i = 0; i < accum->size(); i++)
        accum->at(i) += add[i];
}

double SimpsonMethod::sequential(const std::function<double(const std::vector<double>&)>& func,
                                 const std::vector<double>& seg_begin, const std::vector<double>& seg_end,
                                 int steps_count) {
    if (steps_count <= 0)
        throw std::runtime_error("Steps count must be positive");
    if (seg_begin.empty() || seg_end.empty())
        throw std::runtime_error("No segments");
    if (seg_begin.size() != seg_end.size())
        throw std::runtime_error("Invalid segments");
    size_t dim = seg_begin.size();
    std::vector<double> steps(dim), segments(dim);
    for (size_t i = 0; i < dim; i++) {
        steps[i] = (seg_end[i] - seg_begin[i]) / steps_count;
        segments[i] = seg_end[i] - seg_begin[i];
    }
    std::pair<double, double> sum = std::make_pair(0.0, 0.0);
    std::vector<double> args = seg_begin;
    for (int i = 0; i < steps_count; i++) {
        sumUp(&args, steps);
        if (i % 2 == 0)
            sum.first += func(args);
        else
            sum.second += func(args);
    }
    double seg_prod = std::accumulate(segments.begin(), segments.end(), 1.0, [](double p, double s) { return p * s; });
    return (func(seg_begin) + 4 * sum.first + 2 * sum.second - func(seg_end)) * seg_prod / (3.0 * steps_count);
}

double SimpsonMethod::parallel(const std::function<double(const std::vector<double>&)>& func,
                               std::vector<double> seg_begin, std::vector<double> seg_end, int steps_count) {
    if (steps_count <= 0)
        throw std::runtime_error("Steps count must be positive");
    if (seg_begin.empty() || seg_end.empty())
        throw std::runtime_error("No segments");
    if (seg_begin.size() != seg_end.size())
        throw std::runtime_error("Invalid segments");
    size_t dim = seg_begin.size();
    std::vector<double> steps(dim), segments(dim);
    for (size_t i = 0; i < dim; i++) {
        steps[i] = (seg_end[i] - seg_begin[i]) / steps_count;
        segments[i] = seg_end[i] - seg_begin[i];
    }
    double sum_first = 0, sum_second = 0;
    std::vector<double> args(dim);
    int t_count = 0, t_steps = 0;
#pragma omp parallel firstprivate(args) reduction(+ : sum_first, sum_second)
    {
        int t_id = omp_get_thread_num();
        t_count = omp_get_num_threads();
        t_steps = steps_count / t_count;
        for (size_t i = 0; i < dim; i++)
            args[i] = seg_begin[i] + steps[i] * t_id * t_steps;
        int t_start = t_id * t_steps;
        int t_end = t_start + t_steps;
        for (int i = t_start; i < t_end; i++) {
            sumUp(&args, steps);
            if (i % 2 == 0)
                sum_first += func(args);
            else
                sum_second += func(args);
        }
    }
    if (steps_count % t_count != 0) {
        int passed_steps_count = t_count * t_steps;
        for (size_t i = 0; i < dim; i++)
            args[i] = seg_begin[i] + steps[i] * passed_steps_count;
        for (int i = passed_steps_count; i < steps_count; i++) {
            sumUp(&args, steps);
            if (i % 2 == 0)
                sum_first += func(args);
            else
                sum_second += func(args);
        }
    }
    double seg_prod = std::accumulate(segments.begin(), segments.end(), 1.0, [](double p, double s) { return p * s; });
    return (func(seg_begin) + 4 * sum_first + 2 * sum_second - func(seg_end)) * seg_prod / (3.0 * steps_count);
}
