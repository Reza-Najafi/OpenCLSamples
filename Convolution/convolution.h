#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "CL/cl.hpp"
#include <iomanip>
#include <iostream>

template<typename T>
struct Image
{
	std::vector<T> _buff;
	const size_t _x_len;
	const size_t _y_len;
	Image() = delete;
	T _zero = 0;
	void print()
	{
		std::cout << std::setw(4);
		for (int y = 0; y < _y_len; y++)
		{
			std::cout<<"\n";
			for (int x = 0; x < _x_len; x++)
				 std::cout << +(*this)[y][x] << " ";
				
		}
		std::cout << "\n";
	}
	Image( size_t y_len, size_t x_len) :_x_len(x_len), _y_len(y_len)
	{
		_buff.resize(x_len*y_len);
	}
	inline bool is_valid_cord(int y, int x)
	{
		return (y >= 0 && x >= 0 && y < _y_len && x < _x_len);
	}
	inline T operator()(int y, int x)
	{
		if (is_valid_cord(y, x))
			return _buff[y*_x_len + x];
		return _zero;
	}
	class Proxy
	{
	public:
		inline Proxy(T* arr, size_t len) :_arr(arr), _len(len) {}
		inline T& operator[](size_t ind)
		{
			if (ind >= _len)
				throw std::out_of_range("");
			return _arr[ind];
		}
	private:
		const size_t _len;
		T* _arr;
	};
	inline Proxy operator[](size_t ind)
	{
		if (ind >= _y_len)
			throw std::out_of_range("");
		return Proxy(_buff.data() + ind*_x_len, _x_len);
	}
	inline bool operator==(const Image& other)
	{
		for (int y = 0; y < _y_len; y++)
			for (int x = 0; x < _x_len; x++)
				if (other[y][x] != (*this)[y][x])
					return false;
		return true;
	}
};

void convolution_cpu(Image<float>& input, Image<float>& output, Image<float>& kernel);
void convolution_gpu(cl::CommandQueue& queue, cl::Kernel kernel, cl::Buffer d_input, cl::Buffer d_output, cl::Buffer d_kernel, int y_len, int x_len, int k_len);