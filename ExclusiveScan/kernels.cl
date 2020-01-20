
__kernel void serial_exclusive_scan(__global int* interm_buff, int size)
{
	int glb_id = get_global_id(0);
	if(glb_id != 0)
		return;

	int sum = interm_buff[0];	
	int temp = 0;

	for(int i = 1; i < size; ++i)
	{
		sum += temp;
		temp = interm_buff[i];
		interm_buff[i] = sum;
	}
	interm_buff[0] = 0;
	
}

__kernel void post_process(__global int* input, __global int* interm_buff, int size)
{
	int glb_id = get_global_id(0);
	int blockid = get_group_id(0);

	if (glb_id >= size)
		return;

	input[glb_id] += interm_buff[blockid];
}

__kernel void parallel_exclusive_scan(__global int *input, __global int* interm_buff, int size)
{

	int glb_id = get_global_id(0);
	int lcl_id = get_local_id(0);
	int blockid = get_group_id(0);
	int blockoff = blockid*BLOCK_SIZE;

	if (glb_id >= size)
		return;
	

	__local int input_lds[BLOCK_SIZE];

	// Copy the data that is going to be worked on from the global memory to the local data share on the CU
	input_lds[lcl_id] =  input[glb_id];

	int interm_buff_value = 0;
	
	// Store last element of the block in the interm_buff buffer, this is used for post scan phase
	if(lcl_id == 0 )
		interm_buff_value = input[blockoff+BLOCK_SIZE - 1];

	barrier(CLK_LOCAL_MEM_FENCE);

	
	// Reduction step
	// first step indexes : 2n-1 , n = 1,2, ...
	// second step indexes: 4n-1 , n = 1,2, ...
	// (step*2)*(tid+1) - 1 
	for (int step = 1; step < BLOCK_SIZE; step *= 2)
	{
		int index =  (lcl_id + 1) * 2 * step - 1;
		if(index < BLOCK_SIZE)
			input_lds[index] += input_lds[index - step];
		
			
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(lcl_id == 0 )
		input_lds[BLOCK_SIZE - 1] = 0;
	
	// Down-sweep step
	for(int step = BLOCK_SIZE/2; step > 0; step /= 2)
	{
		int index =  (lcl_id + 1) * 2 * step - 1;
		if(index < BLOCK_SIZE)
		{
			// set index-step to index
			int temp = input_lds[index - step];
			input_lds[index - step] = input_lds[index];
			// add index and index - step, store at index
			input_lds[index] += temp;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}	

	// Augment last element of the block in the interm_buff buffer with last element of the scan results
	if(lcl_id == 0 )
		interm_buff[blockid] = interm_buff_value + input_lds[BLOCK_SIZE - 1];

	// Copy the processed data in the local data share back to the global memory
	input[glb_id] = input_lds[lcl_id];
	
}
