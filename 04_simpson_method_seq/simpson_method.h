// Copyright 2021 Vlasov Maksim

#pragma once

#include <functional>
#include <vector>

namespace SimpsonMethod {

double integrate(const std::function<double(const std::vector<double>&)>& func, const std::vector<double>& seg_begin,
                 const std::vector<double>& seg_end, int steps_count);

} // namespace SimpsonMethod
