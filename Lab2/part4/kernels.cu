#include <math.h>
#include <float.h>
#include <cuda.h>

__global__ void gpu_Heat (float *h, float *g, int N) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
    	int j = blockIdx.y * blockDim.y + threadIdx.y;

    	if( i > 0 && i < (N-1) && j > 0 && j < (N-1) ) { // we don't want to compute the edges
    		g[N * i + j] = 0.25f *(
			h[N * (i + 1) + j] +
			h[N * (i - 1) + j] +
			h[N * i + j - 1] +
			h[N * i + j + 1]);
	}
}


__global__ void gpu_Diff(float *u, float *utmp, float* diffs, int N) {
    int i = (blockIdx.y * blockDim.y) + threadIdx.y;
    int j = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i > 0 && i < N-1 && j > 0 && j < N-1){
        utmp[i*N+j]= 0.25 * (u[ i*N     + (j-1) ]+  // left
                u[ i*N     + (j+1) ]+  // right
                u[ (i-1)*N + j     ]+  // top
                u[ (i+1)*N + j     ]); // bottom
        diffs[i*N+j] = utmp[i*N+j] - u[i*N+j];
        diffs[i*N+j] *= diffs[i*N+j];
    }
}

__global__ void gpu_Heat_reduction(float *idata, float *odata, int N) {
	extern __shared__ float sdata[];
	unsigned int s;

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	unsigned int gridSize = blockDim.x * 2 * gridDim.x;
	sdata[tid] = 0;
	while (i < N) {
		sdata[tid] += idata[i] + idata[i + blockDim.x];
		i += gridSize;
	}
	__syncthreads();

	for (s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if (tid < 32) {
		volatile float *smem = sdata;

		smem[tid] += smem[tid + 32];
		smem[tid] += smem[tid + 16];
		smem[tid] += smem[tid + 8];
		smem[tid] += smem[tid + 4];
		smem[tid] += smem[tid + 2];
		smem[tid] += smem[tid + 1];
	}

	if (tid == 0)
		odata[blockIdx.x] = sdata[0];
}
