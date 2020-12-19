// Copyright 2020 Vlasov Maksim
#pragma once
#include <functional>
#include <vector>

using Vector = std::vector<int>;

Vector createRandomVector(int size);

Vector shellSort(Vector arr);

namespace BatcherMerge {
    Vector parallelSort(Vector arr, std::function<Vector(Vector)> sort_func);
}  // namespace BatcherMerge
