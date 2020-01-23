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

	local float kern_lds[KERN_SIZE];

	for(int i = y_lcl; i < KERN_SIZE; i += Y_BLK_SIZE)
		kern_lds[i] = kern[i];

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
			sum += kern_lds[(j+k_rad)*k_len+(i+k_rad)]*lds[j+y_lcl+k_rad][i+x_lcl+k_rad];

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
	local float kern_lds[KERN_SIZE];


	if(y >= y_len || x >= x_len)
		return;

	for(int i = y_lcl; i < KERN_SIZE; i += Y_BLK_SIZE*X_BLK_SIZE)
		kern_lds[i] = kern[i];

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
#if KERN_SIZE == 3 
	sum += kern_lds[0]*lds[y_lcl][x_lcl];
	sum += kern_lds[1]*lds[y_lcl][x_lcl+1];
	sum += kern_lds[2]*lds[y_lcl][x_lcl+2];

	sum += kern_lds[3]*lds[y_lcl+1][x_lcl];
	sum += kern_lds[4]*lds[y_lcl+1][x_lcl+1];
	sum += kern_lds[5]*lds[y_lcl+1][x_lcl+2];

	sum += kern_lds[6]*lds[y_lcl+2][x_lcl];
	sum += kern_lds[7]*lds[y_lcl+2][x_lcl+1];
	sum += kern_lds[8]*lds[y_lcl+2][x_lcl+2];
#endif
#if KERN_SIZE == 5
	sum += kern_lds[0]*lds[y_lcl][x_lcl];
	sum += kern_lds[1]*lds[y_lcl][x_lcl+1];
	sum += kern_lds[2]*lds[y_lcl][x_lcl+2];
	sum += kern_lds[3]*lds[y_lcl][x_lcl+3];
	sum += kern_lds[4]*lds[y_lcl][x_lcl+4];

	sum += kern_lds[5]*lds[y_lcl+1][x_lcl];
	sum += kern_lds[6]*lds[y_lcl+1][x_lcl+1];
	sum += kern_lds[7]*lds[y_lcl+1][x_lcl+2];
	sum += kern_lds[8]*lds[y_lcl+1][x_lcl+3];
	sum += kern_lds[9]*lds[y_lcl+1][x_lcl+4];

	sum += kern_lds[10]*lds[y_lcl+2][x_lcl];
	sum += kern_lds[11]*lds[y_lcl+2][x_lcl+1];
	sum += kern_lds[12]*lds[y_lcl+2][x_lcl+2];
	sum += kern_lds[13]*lds[y_lcl+2][x_lcl+3];
	sum += kern_lds[14]*lds[y_lcl+2][x_lcl+4];

	sum += kern_lds[15]*lds[y_lcl+3][x_lcl];
	sum += kern_lds[16]*lds[y_lcl+3][x_lcl+1];
	sum += kern_lds[17]*lds[y_lcl+3][x_lcl+2];
	sum += kern_lds[18]*lds[y_lcl+3][x_lcl+3];
	sum += kern_lds[19]*lds[y_lcl+3][x_lcl+4];
	
	sum += kern_lds[20]*lds[y_lcl+4][x_lcl];
	sum += kern_lds[21]*lds[y_lcl+4][x_lcl+1];
	sum += kern_lds[22]*lds[y_lcl+4][x_lcl+2];
	sum += kern_lds[23]*lds[y_lcl+4][x_lcl+3];
	sum += kern_lds[24]*lds[y_lcl+4][x_lcl+4];
#endif
#if KERN_SIZE == 7
	sum += kern_lds[0]*lds[y_lcl][x_lcl];
	sum += kern_lds[1]*lds[y_lcl][x_lcl+1];
	sum += kern_lds[2]*lds[y_lcl][x_lcl+2];
	sum += kern_lds[3]*lds[y_lcl][x_lcl+3];
	sum += kern_lds[4]*lds[y_lcl][x_lcl+4];
	sum += kern_lds[5]*lds[y_lcl][x_lcl+5];
	sum += kern_lds[6]*lds[y_lcl][x_lcl+6];

	sum += kern_lds[7]*lds[y_lcl+1][x_lcl];
	sum += kern_lds[8]*lds[y_lcl+1][x_lcl+1];
	sum += kern_lds[9]*lds[y_lcl+1][x_lcl+2];
	sum += kern_lds[10]*lds[y_lcl+1][x_lcl+3];
	sum += kern_lds[11]*lds[y_lcl+1][x_lcl+4];
	sum += kern_lds[12]*lds[y_lcl+1][x_lcl+5];
	sum += kern_lds[13]*lds[y_lcl+1][x_lcl+6];

	sum += kern_lds[14]*lds[y_lcl+2][x_lcl];
	sum += kern_lds[15]*lds[y_lcl+2][x_lcl+1];
	sum += kern_lds[16]*lds[y_lcl+2][x_lcl+2];
	sum += kern_lds[17]*lds[y_lcl+2][x_lcl+3];
	sum += kern_lds[18]*lds[y_lcl+2][x_lcl+4];
	sum += kern_lds[19]*lds[y_lcl+2][x_lcl+5];
	sum += kern_lds[20]*lds[y_lcl+2][x_lcl+6];

	sum += kern_lds[21]*lds[y_lcl+3][x_lcl];
	sum += kern_lds[22]*lds[y_lcl+3][x_lcl+1];
	sum += kern_lds[23]*lds[y_lcl+3][x_lcl+2];
	sum += kern_lds[24]*lds[y_lcl+3][x_lcl+3];
	sum += kern_lds[25]*lds[y_lcl+3][x_lcl+4];
	sum += kern_lds[26]*lds[y_lcl+3][x_lcl+5];
	sum += kern_lds[27]*lds[y_lcl+3][x_lcl+6];
		
	sum += kern_lds[28]*lds[y_lcl+4][x_lcl];
	sum += kern_lds[29]*lds[y_lcl+4][x_lcl+1];
	sum += kern_lds[30]*lds[y_lcl+4][x_lcl+2];
	sum += kern_lds[31]*lds[y_lcl+4][x_lcl+3];
	sum += kern_lds[32]*lds[y_lcl+4][x_lcl+4];
	sum += kern_lds[33]*lds[y_lcl+4][x_lcl+5];
	sum += kern_lds[34]*lds[y_lcl+4][x_lcl+6];

	sum += kern_lds[35]*lds[y_lcl+5][x_lcl];
	sum += kern_lds[36]*lds[y_lcl+5][x_lcl+1];
	sum += kern_lds[37]*lds[y_lcl+5][x_lcl+2];
	sum += kern_lds[38]*lds[y_lcl+5][x_lcl+3];
	sum += kern_lds[39]*lds[y_lcl+5][x_lcl+4];
	sum += kern_lds[40]*lds[y_lcl+5][x_lcl+5];
	sum += kern_lds[41]*lds[y_lcl+5][x_lcl+6];

	sum += kern_lds[42]*lds[y_lcl+6][x_lcl];
	sum += kern_lds[43]*lds[y_lcl+6][x_lcl+1];
	sum += kern_lds[44]*lds[y_lcl+6][x_lcl+2];
	sum += kern_lds[45]*lds[y_lcl+6][x_lcl+3];
	sum += kern_lds[46]*lds[y_lcl+6][x_lcl+4];
	sum += kern_lds[47]*lds[y_lcl+6][x_lcl+5];
	sum += kern_lds[48]*lds[y_lcl+6][x_lcl+6];

#endif
	output[y*x_len+x] = sum;
}

