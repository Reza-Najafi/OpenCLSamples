#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "CL/cl.hpp"
void convolution_cpu(int *input, int *output, int size);
void convolution_gpu(cl::CommandQueue& queue, std::vector<cl::Kernel>& kernels, cl::Buffer d_input, std::vector<cl::Buffer>& aux_buffs, int input_size, int depth = 0);
void compare_arrays(int * a, int * b, int size);