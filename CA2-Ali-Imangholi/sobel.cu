


#include <stdio.h>
#include <stdlib.h> 
#include <time.h>

#define ROWS 227
#define COLS 227
#define UNDECLARE -999

//		d_outp[y*ROWS + x] = UNDECLARE;

__global__ void sobel_gpu(int *d_outp, int *d_inp, int *d_mask) 
{
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  int y = threadIdx.y + blockIdx.y * blockDim.y; 
  unsigned int idx = y*ROWS + x;
  
  if(idx < COLS * ROWS)
  {
	if( y!=0 || y!=(ROWS-1) || x!=0 || x!=(COLS-1))
	{	d_outp[y*ROWS + x] = d_inp[y*ROWS + x - 4] * d_mask[0] + 
							d_inp[y*ROWS + x - 3] * d_mask[1] +
							d_inp[y*ROWS + x - 2] * d_mask[2] +
							d_inp[y*ROWS + x - 1] * d_mask[3] +
							d_inp[y*ROWS + x    ] * d_mask[4] +
							d_inp[y*ROWS + x + 1] * d_mask[5] +
							d_inp[y*ROWS + x + 2] * d_mask[6] +
							d_inp[y*ROWS + x + 3] * d_mask[7] +
							d_inp[y*ROWS + x + 4] * d_mask[8];
	}
  }
}

void sobel_cpu(int *h_outp, int *h_inp, int *h_mask) 
{
	for(int i=1 ; i<ROWS-1 ; i++)
	{
		for(int j=1 ; j<COLS-1 ; j++)
		{
		    h_outp[i*ROWS + j] = h_inp[i*ROWS + j - 4] * h_mask[0]  + 
			                     h_inp[i*ROWS + j - 3] * h_mask[1]  +
				                 h_inp[i*ROWS + j - 2] * h_mask[2]  +
				                 h_inp[i*ROWS + j - 1] * h_mask[3]  +
				                 h_inp[i*ROWS + j    ] * h_mask[4]  +
				                 h_inp[i*ROWS + j + 1] * h_mask[5]  +
				                 h_inp[i*ROWS + j + 2] * h_mask[6]  +
				                 h_inp[i*ROWS + j + 3] * h_mask[7]  +
				                 h_inp[i*ROWS + j + 4] * h_mask[8] ;
		}
		
	}
}


int main(void)
{
	int *h_inp;
	int *h_outp;
	int *h_mask;
	
	int picSize = ROWS * COLS * sizeof(int);
	int maskSize = 3 * 3 * sizeof(int);
    
	h_inp = (int*)malloc(picSize);
    h_outp = (int*)malloc(picSize);
    h_mask = (int*)malloc(maskSize);	
	
	srand(time(NULL));
	for(int i=0 ; i<ROWS ; i++)
	{
		for(int j=0 ; j<COLS ; j++)
		{
		    h_inp[i*ROWS + j] = ((rand() % 10)+1);
		}
	}
	h_mask[0] = -1;
	h_mask[1] = 0;
	h_mask[2] = 1;
	h_mask[3] = -2;
	h_mask[4] = 0;
	h_mask[5] = 2;
	h_mask[6] = -1;
	h_mask[7] = 0;
	h_mask[8] = 1;
	
	
	int *d_inp;
    int *d_outp;
    int *d_mask;
	cudaMalloc((void**)&d_inp, picSize);
	cudaMalloc((void**)&d_outp, picSize);
	cudaMalloc((void**)&d_mask, maskSize);
	
	cudaMemcpy(d_inp, h_inp, picSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_outp, h_outp, picSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mask, h_mask, maskSize, cudaMemcpyHostToDevice);
	
	
	for(int i=0 ; i<ROWS ; i++)
	{
		for(int j=0 ; j<COLS ; j++)
		{
		    h_outp[i*ROWS + j] = 0;
		}
	}
	
  
	/*
	clock_t start_serial, end_serial;
	
	start_serial = clock();
	
	sobel_cpu(h_outp, h_inp, h_mask);
		
	end_serial = clock();
	
	printf("CPU(serial) time: %f s.\n",(end_serial-start_serial)/(float)CLOCKS_PER_SEC);
	
	for(int i=0 ; i<ROWS ; i++)
	{
		for(int j=0 ; j<COLS ; j++)
		{
			if( i==0 || i==(ROWS-1) || j==0 || j==(COLS-1))
			{
				h_outp[i*ROWS + j] = UNDECLARE;
			}
		}
	}
	*/	
		

	
	dim3 dimBlock(32, 32);
  dim3 dimGrid(  ((COLS+dimBlock.x-1)/dimBlock.x),  ((ROWS+dimBlock.y-1)/dimBlock.y)  );
 
 /* for part3 */
 // dim3 dimGrid(2,13);


	sobel_gpu<<<dimGrid, dimBlock>>>(d_outp, d_inp, d_mask);
	
	clock_t start_gpu, end_gpu;

	start_gpu = clock();

	cudaMemcpy(h_outp,d_outp,picSize,cudaMemcpyDeviceToHost);
	
    end_gpu = clock();	
	
	printf("GPU(parallel) time: %f s.\n",(end_gpu-start_gpu)/(float)CLOCKS_PER_SEC);
  printf("dimGrid.x %d dimBlock.x %d \n", dimGrid.x, dimBlock.x);	
  printf("dimGrid.y %d dimBlock.y %d \n", dimGrid.y, dimBlock.y);
  printf("dimGrid.z %d dimBlock.z %d \n", dimGrid.z, dimBlock.z);


	for(int i=0 ; i<ROWS ; i++)
	{
		for(int j=0 ; j<COLS ; j++)
		{
			if( i==0 || i==(ROWS-1) || j==0 || j==(COLS-1))
			{
				h_outp[i*ROWS + j] = UNDECLARE;
			}
		}
	}

	
	
	free( h_inp );
    free( h_outp );
    free( h_mask );
	
	cudaFree(d_inp);
	cudaFree(d_outp);
	cudaFree(d_mask);
	
	return 0;
}