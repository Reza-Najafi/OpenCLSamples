#include <string>
#include <memory>
#include <cstdlib>
#include <iostream>

#include "CL/cl.hpp"
#include "stdafx.h"
#include "exclusive_scan.h"
#include "ocl_setup.h"

#define BLOCK_SIZE 256
//
// Assumption is the length of input is a multiple of BLOCK_SIZE
// 
int main(int argc, char** argv)
{
	int input_len_pow2 = 10;

	if (argc > 1)
		input_len_pow2 = atoi(argv[1]);

	int input_len = 1 << input_len_pow2;

	printf("Running input length %d (pow2, %d)\n", input_len, input_len_pow2);
	const int input_byte_size = sizeof(int) * input_len;

	clock_t cpu_start, cpu_end, gpu_start, gpu_end,  xfer_d2h_start, xfer_d2h_end;

	std::vector<int> h_input(input_len);
	std::vector<int> h_cpu_output(input_len);
	std::vector<int> h_gpu_output(input_len);

	time_t t;
	srand((unsigned)time(&t));

	for (int i = 0; i < input_len; i++)
		h_input[i] = rand() % 10;


	cpu_start = clock();
	convolution_cpu(h_input.data(), h_cpu_output.data(), input_len);
	cpu_end = clock();

	try
	{
		int platform_idx = 0, device_idx = 0;
		auto ret = get_device_and_context(device_idx, CL_DEVICE_TYPE_GPU);
		cl::Context context = ret.first;
		std::vector<cl::Device> device = { ret.second };

		// Creating a command queue
		cl::CommandQueue queue = cl::CommandQueue(context, device[0]);

		// Reading the OpenCL program source file
		FILE* f = fopen("../../ExclusiveScan/kernels.cl", "r");
		if (!f)
			throw std::runtime_error("Couldn't open the kernel file (kernels.cl) ");
		else
			printf("kernels suorce loaded\n");
		// Finding the kernel file size
		fseek(f, 0, SEEK_END);
		size_t size = ftell(f);

		std::vector<char> source_code(size);

		rewind(f);
		fread(source_code.data(), sizeof(char), size, f);
		cl::Program::Sources source(1, std::make_pair(source_code.data(), source_code.size()));

		// Making program from the source code
		cl::Program program = cl::Program(context, source);

		std::string build_options = "-DBLOCK_SIZE=" + std::to_string(BLOCK_SIZE);
		// Building the program for the devices
		if (program.build(device, build_options.c_str()) != CL_SUCCESS)
			throw std::runtime_error("Failed building the program");

		printf("OCL Program built\n");

		// creating kernels from the program
		cl::Kernel parallel_exclusive_scan(program, "parallel_exclusive_scan");
		cl::Kernel serial_exclusive_scan(program, "serial_exclusive_scan");
		cl::Kernel post_process(program, "post_process");
		int interm_size = input_len;
		int interm_byte_size = sizeof(int) * interm_size;
		std::vector<cl::Buffer> d_interm_buffs;

		// Creating intermediate buffers used to perform scan of the last elemets of each work group in each recursive round of the exclusive_scan_recursive_gpu function call
		while (interm_size > BLOCK_SIZE)
		{
			d_interm_buffs.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, interm_byte_size));
			interm_size /= BLOCK_SIZE;
			interm_byte_size /= BLOCK_SIZE;
		}
		printf("Total %d intermediate buffers created\n", d_interm_buffs.size());


		int zero = 0;
		for (auto&& d_aux : d_interm_buffs)
			queue.enqueueFillBuffer(d_aux, &zero, 0, interm_size, 0, 0);

		cl::Buffer d_input = cl::Buffer(context,  CL_MEM_COPY_HOST_PTR, input_byte_size, h_input.data());
		// Calling clFinish to make sure input buffer is created and initialized
		queue.finish();

		// Running on the device
		std::vector<cl::Kernel> kernels = { parallel_exclusive_scan , serial_exclusive_scan, post_process };
		gpu_start = clock();
		convolution_gpu(queue, kernels, d_input, d_interm_buffs, input_len);
		queue.finish();
		gpu_end = clock();

		xfer_d2h_start = clock();
		queue.enqueueReadBuffer(d_input, CL_TRUE, 0, input_byte_size, h_gpu_output.data());
		xfer_d2h_end = clock();
		auto xfer_d2h = xfer_d2h_end - xfer_d2h_start;
		printf("CPU time %lld, GPU %lld, XferD2H %lld\n", cpu_end - cpu_start, gpu_end - gpu_start, xfer_d2h);

	}
	catch (std::exception& e)
	{
		printf("exception thrown >>> %s\n", e.what());
		return 0;
	}

	compare_arrays(h_gpu_output.data(), h_cpu_output.data(), input_len);
	printf("Done executing the code\n");
	return 0;
}

void convolution_gpu(cl::CommandQueue& queue, std::vector<cl::Kernel>& kernels, cl::Buffer d_input, std::vector<cl::Buffer>& interm_buffs, int input_len, int depth)
{
	if (depth >= interm_buffs.size())
	{
		auto kernel = kernels[1]; // serial scan
		kernel.setArg(0, d_input);
		kernel.setArg(1, input_len);
		if (queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1)) != CL_SUCCESS)
			throw std::runtime_error("Couldn't run the serial_exclusive_scan kernel");
		return;
	}
	else
	{
		auto kernel = kernels[0];// exclusive scan efficinet single block;
		kernel.setArg(0, d_input);
		kernel.setArg(1, interm_buffs[depth]);
		kernel.setArg(2, input_len);
		if (queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_len), cl::NDRange(BLOCK_SIZE)) != CL_SUCCESS)
			throw std::runtime_error("Couldn't run the parallel_exclusive_scan kernel");
		convolution_gpu(queue, kernels, interm_buffs[depth], interm_buffs, input_len / BLOCK_SIZE, depth + 1);
	}

	auto kernel = kernels[2]; // post_process
	kernel.setArg(0, d_input);
	kernel.setArg(1, interm_buffs[depth]);
	kernel.setArg(2, input_len);
	if (queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_len), cl::NDRange(BLOCK_SIZE)) != CL_SUCCESS)
		throw std::runtime_error("Couldn't run the post_process kernel");
}

void convolution_cpu(int* input, int* output, int size)
{
	output[0] = 0;

	for (int i = 1; i < size; i++)
		output[i] = output[i - 1] + input[i - 1];
}


//compare arrays
void compare_arrays(int* a, int* b, int size)
{
	const int MAX_ERROR_DISPLAY = 16;
	int error_count = 0;
	for (int i = 0; i < size; i++)
	{
		if (a[i] != b[i])
		{
			error_count++;
			if (error_count < MAX_ERROR_DISPLAY)
				printf("MissMatch %d - %d | %d \n", i, a[i], b[i]);
		}
	}
	if (error_count == 0)
		printf("CPU and GPU results are same \n");
	else
		printf("CPU and GPU results are different in %d number of elements\n", error_count);
}

