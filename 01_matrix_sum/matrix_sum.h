// Copyright 2020 Vlasov Maksim
#pragma once

#include <vector>

std::vector<int> createRandomVector(int elements_count);
int calculateSumSequental(const std::vector<int> &vector);
int calculateSumParallel(const std::vector<int> &vector, int elements_count);
