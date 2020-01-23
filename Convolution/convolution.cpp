#include <string>
#include <memory>
#include <cstdlib>


#include "CL/cl.hpp"
#include "stdafx.h"
#include "convolution.h"
#include "ocl_setup.h"
   

#define BLOCK_Y 16
#define BLOCK_X 16
//compare arrays
template<typename T>
void compare_arrays(T* a, T* b, int size)
{
	const int MAX_ERROR_DISPLAY = 16;
	int error_count = 0;
	for (int i = 0; i < size; i++)
	{
		if (a[i] != b[i])
		{
			error_count++;
			if (error_count < MAX_ERROR_DISPLAY)
				std::cout << "MissMatch " << i << " - " << a[i] << " | " << b[i] << "\n";
		}
	}
	if (error_count == 0)
		printf("CPU and GPU results are same \n");
	else
		printf("CPU and GPU results are different in %d number of elements\n", error_count);
}


int main(int argc, char** argv)
{
	int x_len = 6;
	int y_len = 4;
	bool print_out = 0;
	size_t k_rad = 1;
	size_t k_len = k_rad * 2 + 1;
	int kernel_id = 0;
	if (argc > 1)
		y_len = atoi(argv[1]);

	if (argc > 2)
		x_len = atoi(argv[2]);
	
	if (argc > 3)
		k_rad = atoi(argv[3]);

	if (argc > 4)
		kernel_id = atoi(argv[4]);

	if (argc > 5)
		print_out = atoi(argv[5]);

	printf("Running input length %d x %d \n", y_len, x_len);
	const int input_byte_size = sizeof(float) * x_len * y_len;
	const int kernel_size = sizeof(float) * k_len * k_len;


	clock_t cpu_start, cpu_end, gpu_start, gpu_end,  xfer_d2h_start, xfer_d2h_end;

	Image<float> h_input(y_len, x_len );
	Image<float> h_cpu_output(y_len, x_len);
	Image<float> h_kernel(3, 3);
	Image<float> h_gpu_output(y_len, x_len);

	time_t t;
	srand((unsigned)time(&t));
	
	for (int y = 0; y < y_len; y++)
		for (int x = 0; x < x_len; x++)
			h_input[y][x] = rand() % 10;

	for (int y = 0; y < k_len; y++)
		for (int x = 0; x < k_len; x++)
			h_kernel[y][x] = (1.0f/9.0f);

	if (print_out)
	{
		h_input.print();
		h_kernel.print();
	}


	cpu_start = clock();
	convolution_cpu(h_input, h_cpu_output, h_kernel);
	if (print_out)
		h_cpu_output.print();
	cpu_end = clock();
#if 1
	try
	{
		int platform_idx = 0, device_idx = 0;
		auto ret = get_device_and_context(device_idx, CL_DEVICE_TYPE_GPU);
		cl::Context context = ret.first;
		std::vector<cl::Device> device = { ret.second };

		// Creating a command queue
		cl::CommandQueue queue = cl::CommandQueue(context, device[0]);

		// Reading the OpenCL program source file
		FILE* f = fopen("../../Convolution/kernels_conv.cl", "r");
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

		std::string build_options = "-DY_BLK_SIZE="+ std::to_string(BLOCK_Y+k_rad*2) + " -DX_BLK_SIZE=" + std::to_string(BLOCK_X+k_rad*2) + " -DKERN_SIZE=" + std::to_string((k_rad * 2 + 1)*(k_rad * 2+1));
		// Building the program for the devices
		if (program.build(device, build_options.c_str()) != CL_SUCCESS)
		{
			std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device[0]);
			std::cout << buildlog << std::endl;
			throw std::runtime_error("Failed building the program");
		}
			

		printf("OCL Program built\n");

		// creating kernels from the program
		std::string kernel_name = (kernel_id == 0 )?"conv":
			(kernel_id == 1) ? "conv_shm":
			(kernel_id == 2) ? "conv_shm_unrolled":"";
		std::cout << "Using " << kernel_name << std::endl;
		cl::Kernel conv(program, kernel_name.c_str());

		cl::Buffer d_input = cl::Buffer(context,  CL_MEM_COPY_HOST_PTR, input_byte_size, h_input._buff.data());
		cl::Buffer d_kernel = cl::Buffer(context, CL_MEM_COPY_HOST_PTR, kernel_size, h_kernel._buff.data());
		cl::Buffer d_output = cl::Buffer(context,CL_MEM_WRITE_ONLY, input_byte_size);

		// Calling clFinish to make sure input buffer is created and initialized
		queue.finish();

		// Running on the device
		std::vector<cl::Kernel> kernels = { conv };
		gpu_start = clock();
		convolution_gpu(queue, kernels[0], d_input, d_output, d_kernel, y_len, x_len, k_rad);
		queue.finish();
		gpu_end = clock();

		xfer_d2h_start = clock();
		queue.enqueueReadBuffer(d_output, CL_TRUE, 0, input_byte_size, h_gpu_output._buff.data());
		xfer_d2h_end = clock();
		printf("done GPU \n");
		auto xfer_d2h = xfer_d2h_end - xfer_d2h_start;
		printf("CPU time %lld, GPU %lld, XferD2H %lld\n", cpu_end - cpu_start, gpu_end - gpu_start, xfer_d2h);

	}
	catch (std::exception& e)
	{
		printf("exception thrown >>> %s\n", e.what());
		return 0;
	}

	compare_arrays(h_gpu_output._buff.data(), h_cpu_output._buff.data(), y_len*x_len);
#endif
	printf("Done executing the code\n");
	return 0;
}

void convolution_gpu(cl::CommandQueue& queue,cl::Kernel kernel, cl::Buffer d_input, cl::Buffer d_output, cl::Buffer d_kernel, int y_len, int x_len, int k_rad)
{
	kernel.setArg(0, d_input);
	kernel.setArg(1, d_output);
	kernel.setArg(2, d_kernel);
	kernel.setArg(3, y_len);
	kernel.setArg(4, x_len);
	kernel.setArg(5, k_rad);
	if (queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(y_len, x_len), cl::NDRange(BLOCK_Y, BLOCK_X)) != CL_SUCCESS)
		throw std::runtime_error("Couldn't run the kernel");
}


void convolution_cpu(Image<float>& input, Image<float>& output, Image<float>& kernel)
{
	if (kernel._x_len != kernel._y_len || kernel._x_len % 2 == 0)
		throw std::runtime_error("invalid kernel size");

	int k_rad = kernel._x_len/2;
	printf("Kernel radius %d\n", k_rad);
	for (int y = 0; y < input._y_len; ++y)
	{
		for (int x = 0; x < input._x_len; ++x)
		{
			float sum = 0;
			for (int j = -k_rad; j <= k_rad; ++j)
				for (int i = -k_rad; i <= k_rad; ++i)
					sum += (float)input(y+j, x+i)* kernel(k_rad+j, k_rad+i);
			output[y][x] = sum;
		}
	}
}



