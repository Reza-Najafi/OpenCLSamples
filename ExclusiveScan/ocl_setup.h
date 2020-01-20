#pragma once
#include "CL\cl.h"
#include "CL\cl.hpp"
std::pair<cl::Context, cl::Device> get_device_and_context( int devIdx, cl_device_type clDeviceType);