// Copyright 2021 Vlasov Maksim

#pragma once

#include <functional>
#include <vector>

namespace SimpsonMethod {

double sequential(const std::function<double(const std::vector<double>&)>& func, const std::vector<double>& seg_begin,
                  const std::vector<double>& seg_end, int steps_count);

double parallel(const std::function<double(const std::vector<double>&)>& func, std::vector<double> seg_begin,
                std::vector<double> seg_end, int steps_count, int num_threads = 1);

} // namespace SimpsonMethod
