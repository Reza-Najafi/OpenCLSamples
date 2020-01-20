__kernel void conv(__global float* input, __global float* output, __global float* kern, int y_len , int x_len, int k_rad)
{
	int y = get_global_id(0);
	int x = get_global_id(1);

	if(y >= y_len || x >= x_len)
		return;

	float sum = 0;
	int k_len = 2*k_rad +1;
	
	for(int j = -k_rad; j <= k_rad; ++j)
		for(int i = -k_rad; i <= k_rad; ++i)
			if(j+y >= 0 && i+x >= 0 && j+y < y_len && i+x < x_len)
				sum += kern[(j+k_rad)*k_len+(i+k_rad)]*input[(j+y)*x_len+(i+x)];

	output[y*x_len+x] = sum;

}

__kernel void conv_shm(__global float* input, __global float* output, __global float* kern, int y_len , int x_len, int k_rad)
{
	int y = get_global_id(0);
	int x = get_global_id(1);
	int y_lcl = get_local_id(0);
	int x_lcl = get_local_id(1);
	int Y_BLK = get_local_size(0);
	int X_BLK = get_local_size(1);
	int y_off = get_group_id(0)*Y_BLK;
	int x_off = get_group_id(1)*X_BLK;

	local float lds[Y_BLK_SIZE][X_BLK_SIZE];

	if(y >= y_len || x >= x_len)
		return;

	for(int j = y_lcl ; j < Y_BLK_SIZE; j += Y_BLK)
		for(int i = x_lcl; i < X_BLK_SIZE; i += X_BLK)
		{
			int y_idx = y_off-k_rad+j;
			int x_idx = x_off-k_rad+i;

			if(y_idx >= 0 && x_idx >= 0 && y_idx < y_len && x_idx < x_len)
				lds[j][i] = input[y_idx*x_len+x_idx];
			else
				lds[j][i] = 0;
		}

	barrier(CLK_LOCAL_MEM_FENCE);

	float sum = 0;
	int k_len = 2*k_rad +1;
	
	for(int j = -k_rad; j <= k_rad; ++j)
		for(int i = -k_rad; i <= k_rad; ++i)
			sum += kern[(j+k_rad)*k_len+(i+k_rad)]*lds[j+y_lcl+k_rad][i+x_lcl+k_rad];

	output[y*x_len+x] = sum;
}


__kernel void conv_shm_unrolled(__global float* input, __global float* output, __global float* kern, int y_len , int x_len, int k_rad)
{
	int y = get_global_id(0);
	int x = get_global_id(1);
	int y_lcl = get_local_id(0);
	int x_lcl = get_local_id(1);
	int Y_BLK = get_local_size(0);
	int X_BLK = get_local_size(1);
	int y_off = get_group_id(0)*Y_BLK;
	int x_off = get_group_id(1)*X_BLK;

	local float lds[Y_BLK_SIZE][X_BLK_SIZE];

	if(y >= y_len || x >= x_len)
		return;

	for(int j = y_lcl ; j < Y_BLK_SIZE; j += Y_BLK)
		for(int i = x_lcl; i < X_BLK_SIZE; i += X_BLK)
		{
			int y_idx = y_off-k_rad+j;
			int x_idx = x_off-k_rad+i;

			if(y_idx >= 0 && x_idx >= 0 && y_idx < y_len && x_idx < x_len)
				lds[j][i] = input[y_idx*x_len+x_idx];
			else
				lds[j][i] = 0;
		}

	barrier(CLK_LOCAL_MEM_FENCE);

	float sum = 0;	
	sum += kern[0]*lds[y_lcl][x_lcl];
	sum += kern[1]*lds[y_lcl][x_lcl+1];
	sum += kern[2]*lds[y_lcl][x_lcl+2];

	sum += kern[3]*lds[y_lcl+1][x_lcl];
	sum += kern[4]*lds[y_lcl+1][x_lcl+1];
	sum += kern[5]*lds[y_lcl+1][x_lcl+2];

	sum += kern[6]*lds[y_lcl+2][x_lcl];
	sum += kern[7]*lds[y_lcl+2][x_lcl+1];
	sum += kern[8]*lds[y_lcl+2][x_lcl+2];

	output[y*x_len+x] = sum;
}
