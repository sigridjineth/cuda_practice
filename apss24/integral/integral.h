#pragma once

#include <cstdlib>

double integral_naive(size_t num_intervals);
double integral(size_t num_intervals);
void integral_init(size_t num_intervals);
void integral_cleanup();
