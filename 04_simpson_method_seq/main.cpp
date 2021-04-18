// Copyright 2021 Vlasov Maksim

#include <gtest/gtest.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "simpson_method.h"

#define MULTIDIM_FUNC(FNAME, FVARCOUNT, FCOMP)                                                                         \
    double FNAME(const std::vector<double>& x) {                                                                       \
        assert(x.size() == (FVARCOUNT));                                                                               \
        return (FCOMP);                                                                                                \
    }

MULTIDIM_FUNC(generic, 1, 0);
MULTIDIM_FUNC(parabola, 1, -x[0] * x[0] + 4);
MULTIDIM_FUNC(body, 2, x[0] * x[0] + x[1] * x[1]);
MULTIDIM_FUNC(super, 3, std::sin(x[0] + 3) - std::log(x[1]) + x[2] * x[2]);

TEST(Sequential_SimpsonMethodTest, can_integrate_2d_function) {
    std::vector<double> seg_begin = {0};
    std::vector<double> seg_end = {2};
    double square = SimpsonMethod::integrate(parabola, seg_begin, seg_end, 100);
    ASSERT_NEAR(16.0 / 3.0, square, 1e-6);
}

TEST(Sequential_SimpsonMethodTest, can_integrate_3d_function) {
    std::vector<double> seg_begin = {0, 0};
    std::vector<double> seg_end = {1, 1};
    double volume = SimpsonMethod::integrate(body, seg_begin, seg_end, 100);
    ASSERT_NEAR(2.0 / 3.0, volume, 1e-6);
}

// Calculated by WolframAlpha with the following query:
// integrate (sin(x + 3) - ln(y) + z^2), x=[-2, 1], y=[1, 3], z=[0, 2]
TEST(Sequential_SimpsonMethodTest, can_integrate_super_function) {
    std::vector<double> seg_begin = {-2, 1, 0};
    std::vector<double> seg_end = {1, 3, 2};
    double integral = SimpsonMethod::integrate(super, seg_begin, seg_end, 100);
    ASSERT_NEAR(13.0007625, integral, 1e-6);
}

TEST(Sequential_SimpsonMethodTest, cannot_accept_empty_segment_vectors) {
    ASSERT_ANY_THROW(SimpsonMethod::integrate(generic, {}, {}, 100));
    ASSERT_ANY_THROW(SimpsonMethod::integrate(generic, {0}, {}, 100));
    ASSERT_ANY_THROW(SimpsonMethod::integrate(generic, {}, {0}, 100));
}

TEST(Sequential_SimpsonMethodTest, cannot_accept_invalid_segment_vectors) {
    ASSERT_ANY_THROW(SimpsonMethod::integrate(generic, {1, 2}, {1, 2, 3}, 100));
    ASSERT_ANY_THROW(SimpsonMethod::integrate(generic, {1, 2, 3}, {1, 2}, 100));
}

TEST(Sequential_SimpsonMethodTest, cannot_accept_invalid_steps_count) {
    ASSERT_ANY_THROW(SimpsonMethod::integrate(generic, {0}, {0}, 0));
    ASSERT_ANY_THROW(SimpsonMethod::integrate(generic, {0}, {0}, -1));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
