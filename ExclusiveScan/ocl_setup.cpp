#include "stdafx.h"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include "ocl_setup.h"

#define TOSTR(X) std::to_string(static_cast<int>(X))
#define STR(X) std::string(X)

#define LOG(X) std::clog << "[INF] "  << " {" << __func__ <<"} " << " " << X << std::endl;
#define ERR(X) std::clog << "[ERR] "  << " {" << __func__ <<"} " << " " << X << std::endl;
#define WRN(X) std::clog << "[WRN] "  << " {" << __func__ <<"} " << " " << X << std::endl;

std::pair<cl::Context, cl::Device> get_device_and_context(int devIdx, cl_device_type clDeviceType)
{
	cl_context context;
	cl_device_id device;
	cl_int error = CL_DEVICE_NOT_FOUND;

	int status;

	cl_uint numPlatforms = 0;
	cl_platform_id platform = nullptr;
	std::vector<cl_platform_id> platforms;
	status = clGetPlatformIDs(0, nullptr, &numPlatforms);
	if (status != CL_SUCCESS)
		throw std::runtime_error("clGetPlatformIDs returned error: " + TOSTR(status));


		if (0 < numPlatforms)
		{
			platforms.resize(numPlatforms);
			status = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
			if (status != CL_SUCCESS)
				throw std::runtime_error("clGetPlatformIDs returned error: " + TOSTR(status));

			for (unsigned i = 0; i < numPlatforms; ++i)
			{
				char vendor[100];
				status = clGetPlatformInfo(platforms[i],
					CL_PLATFORM_VENDOR,
					sizeof(vendor),
					vendor,
					nullptr);

				const std::string AMD_NAME = "Advanced Micro Devices, Inc.";

				if (status != CL_SUCCESS) {
					LOG("clGetPlatformInfo returned error: " + TOSTR(status))
						continue;
				}
				if (AMD_NAME.compare(0, AMD_NAME.length(), vendor) == 0)
				{
					LOG("Platform found " + STR(vendor))
					platform = platforms[i];
					break;
				}

			}
		}
		
		if (!platform)
			throw std::runtime_error("Couldn't find AMD OpenCL platform");


		// enumerate devices
		char driverVersion[100] = "\0";

	cl_context_properties contextProps[3] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platform,
		0
	};

	// Retrieve device
	cl_uint numDevices = 0;
	clGetDeviceIDs(platform, clDeviceType, 0, nullptr, &numDevices);
	if (numDevices == 0)
		throw std::runtime_error("No GPU OpenCL device found on the AMD's platform");

		std::vector<cl_device_id> devices;
	devices.resize(numDevices);

	status = clGetDeviceIDs(platform, clDeviceType, numDevices, devices.data(), &numDevices);
	if (status != CL_SUCCESS)
		throw std::runtime_error("clGetDeviceIDs returned error: " + TOSTR(status));

		clGetDeviceInfo(devices[0], CL_DRIVER_VERSION, sizeof(driverVersion), driverVersion, nullptr);

	LOG("Driver version: " + STR(driverVersion));
	LOG("Devices found " + TOSTR(numDevices));
	for (unsigned int n = 0; n < numDevices; n++)
	{
		char deviceName[100] = "\0";
		clGetDeviceInfo(devices[n], CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
		LOG("GPU device selected: " + STR(deviceName))
	}

	if (devIdx >= 0 && devIdx <  (int)numDevices)
	{
		device = devices[devIdx];
		clRetainDevice(device);
		context = clCreateContext(contextProps, 1, &device, nullptr, nullptr, &error);

		if (error == CL_SUCCESS) {
			char deviceName[100] = "\0";
			clGetDeviceInfo(devices[devIdx], CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
			LOG("Using GPU device " + STR(deviceName));
		}
		else {
			throw std::runtime_error("clCreateContext failed: " + TOSTR(error));
		}
	}
	else {
		throw std::runtime_error("Device id " + TOSTR(devIdx) + " is out of range of available devices " + TOSTR(numDevices));
	}

	for (unsigned int idx = 0; idx < numDevices; idx++) {
		clReleaseDevice(devices[idx]);
	}

	return std::make_pair(cl::Context(context), cl::Device(device));
}