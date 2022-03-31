#include <Test/ParallelPrimitives/RadixSortConfigs.h>
#define LDS_BARRIER __syncthreads()

using namespace Oro;
typedef unsigned char u8;
typedef unsigned int u32;

#define THE_FIRST_THREAD threadIdx.x == 0 && blockIdx.x == 0

extern "C"
__global__ void CountKernelReference( int* gSrc, int* gDst, int gN, int gNItemsPerWI, const int START_BIT, const int N_WGS_EXECUTED )
{
	
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;

	const int offset = blockIdx.x * blockDim.x * gNItemsPerWI;

	int table[BIN_SIZE] = {0};

	for( int i = 0; i < gNItemsPerWI; i++)
	{
		int idx = offset + threadIdx.x * gNItemsPerWI + i;

		if( idx >= gN )
			continue;
		int tableIdx = ( gSrc[idx] >> START_BIT ) & RADIX_MASK;
		table[tableIdx] ++;
	}
	
	const int wgIdx = blockIdx.x;

	for(int i=0; i<BIN_SIZE; i++)
	{
		if( table[i] != 0 )
		{
			atomicAdd( &gDst[i * N_WGS_EXECUTED + wgIdx], table[i] );
		}
	}
}

extern "C" 
__global__ void SortKernelReference( int* gSrc, int* gDst, int* gHistogram, int gN, int gNItemsPerWI, const int START_BIT, const int N_WGS_EXECUTED )
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int offset = blockIdx.x * blockDim.x * gNItemsPerWI;
	const int wgIdx = blockIdx.x;

	__shared__ int lds[WG_SIZE];
	__shared__ int localHistogram[BIN_SIZE];
	__shared__ int localOffsets[BIN_SIZE];

	for(int i=threadIdx.x; i<BIN_SIZE; i+=WG_SIZE)
	{
		localOffsets[i] = gHistogram[i * N_WGS_EXECUTED + wgIdx ];
	}
	LDS_BARRIER;

	for( int i = 0; i < gNItemsPerWI; i++ )
	{
		int idx = offset + threadIdx.x * gNItemsPerWI + i;

		int key;
		int tableIdx;
		if( idx < gN )
		{
			key = gSrc[idx];
			tableIdx = ( key >> START_BIT ) & RADIX_MASK;
		}

		LDS_BARRIER;
		if( idx < gN )
		{
			int dstIdx = atomicAdd( &localOffsets[tableIdx], 1 );
			gDst[dstIdx] = key;
		}
	}
}


//=====
extern "C" __global__ void CountKernel( int* gSrc, int* gDst, int gN, int gNItemsPerWI, const int START_BIT, const int N_WGS_EXECUTED )
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int offset = blockIdx.x * blockDim.x * gNItemsPerWI;

/*
	int value = 1;
	__shared__ int scanned[WG_SIZE];

	{
		unsigned int mask = 0xffffffff;
		int width = 32;
		for( int i = 1; i <= width; i *= 2 )
		{
			int n = __shfl_up_sync( mask, value, i, width );

			if( threadIdx.x >= i ) value += n;
		}

		{//make it exclusive
			value = __shfl_up_sync( mask, value, 1, width );
			if( threadIdx.x == 0 ) value = 0;
		}
		scanned[threadIdx.x] = value;

		LDS_BARRIER;

		if( gIdx == 0 )
		{
			for(int i=0; i<WG_SIZE; i++)
				printf("%d,", scanned[i]);
			int a = 0;
			a ++;
		}
	}
*/

	__shared__ union
	{
		u8 m_wiCounter[WG_SIZE][N_PACKED][PACK_FACTOR];//can share a counter among WIs to reduce lds
		int m_wiPackedCounter[WG_SIZE][N_PACKED];
	} lds;

	int table[N_PACKED_PER_WI][PACK_FACTOR] = { 0 };

	for( int iter = 0; iter < gNItemsPerWI; iter+=255 )
	{
		LDS_BARRIER;
		for(int i=0; i<N_PACKED; i++)
			lds.m_wiPackedCounter[threadIdx.x][i] = 0;

		for(int i=0; i<min(255, gNItemsPerWI-iter); i++)
		{
			int idx = offset + (iter+i) * WG_SIZE + threadIdx.x;
			if( idx < gN )
			{
				int tableIdx = ( gSrc[idx] >> START_BIT ) & RADIX_MASK;
				int s = tableIdx / PACK_FACTOR;
				int t = tableIdx % PACK_FACTOR;
				lds.m_wiCounter[threadIdx.x][s][t]++;
			}
		}

		LDS_BARRIER;
		// put them back to vgpr
		for(int i=0; i<N_PACKED_PER_WI; i++)
		{
			for( int k = 0; k < PACK_FACTOR; k++ )
			{
				int sum = 0;
				int ii = threadIdx.x * N_PACKED_PER_WI + i;
				for( int j = 0; j < WG_SIZE; j++ )
				{
					sum += lds.m_wiCounter[j][ii][k];
				}
				table[i][k] += sum;
			}
		}
	}

	const int wgIdx = blockIdx.x;

	for( int i = 0; i < N_PACKED_PER_WI; i++ )
	{
		int ii = threadIdx.x * N_PACKED_PER_WI + i;
		for( int k = 0; k < PACK_FACTOR; k++ )
		{
			int binIdx = ii * PACK_FACTOR + k;
			gDst[binIdx * N_WGS_EXECUTED + wgIdx] = table[i][k];
		}
	}
}

__device__ int ldsScan( int* lds, int width ) 
{
	int idx = threadIdx.x;
	for( int i = 1; i < width; i*=2 )
	{
		if( idx >= i ) lds[idx] += lds[idx - i];
		LDS_BARRIER;
	}
/*
	if( idx >= 1 ) lds[idx] += lds[idx - 1];
	LDS_BARRIER
	if( idx >= 2 ) lds[idx] += lds[idx - 2];
	LDS_BARRIER
	if( idx >= 4 ) lds[idx] += lds[idx - 4];
	LDS_BARRIER
	if( idx >= 8 ) lds[idx] += lds[idx - 8];
	LDS_BARRIER
	if( idx >= 16 ) lds[idx] += lds[idx - 16];
	LDS_BARRIER
*/
	//	if( idx >= 32 ) lds[idx] += lds[idx - 32];

	LDS_BARRIER;
	int sum = lds[width-1];
	LDS_BARRIER;

	int t = (idx==0)?0:lds[idx-1];
	lds[idx] = t;

	LDS_BARRIER;
	return sum;
}


__device__ void localSort4bit( u32* ldsKeys, u32 ldsTemp[WG_SIZE][16], u32 ldsTemp1[16], const int START_BIT )
{
	int key = ldsKeys[threadIdx.x];
	int in4bit = ( key >> START_BIT ) & 0xf;
	for(int i=0; i<16; i++)
	{
		ldsTemp[threadIdx.x][i] = 0;
	}
	ldsTemp[threadIdx.x][in4bit] = 1;

	LDS_BARRIER;

	if( threadIdx.x < 16 )
	{
		int sum = 0;
		for(int i=0; i<WG_SIZE; i++)
		{
			int t = ldsTemp[i][threadIdx.x];
			ldsTemp[i][threadIdx.x] = sum;
			sum += t;
		}
		ldsTemp1[threadIdx.x] = sum;
	}
	LDS_BARRIER;
	if( threadIdx.x == 0 )
	{
		int sum = 0;
		for(int i=0; i<16; i++)
		{
			int t = ldsTemp1[i];
			ldsTemp1[i] = sum;
			sum += t;
		}
	}
	LDS_BARRIER;

	int offset = ldsTemp1[in4bit];
	int rank = ldsTemp[threadIdx.x][in4bit];

	ldsKeys[ offset + rank ] = key;

	LDS_BARRIER;
}

__device__ void localSort8bit( u32* ldsKeys, u32 ldsTemp[WG_SIZE][16], u32 ldsTemp1[16], const int START_BIT )
{ 
	localSort4bit( ldsKeys, ldsTemp, ldsTemp1, START_BIT );
	if( N_RADIX  > 4 )
		localSort4bit( ldsKeys, ldsTemp, ldsTemp1, START_BIT+4 );
}


extern "C" __global__ void SortKernel( int* gSrc, int* gDst, int* gHistogram, int gN, int gNItemsPerWI, const int START_BIT, const int N_WGS_EXECUTED )
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int offset = blockIdx.x * blockDim.x * gNItemsPerWI;
	const int wgIdx = blockIdx.x;

	__shared__ int localOffsets[BIN_SIZE];
	__shared__ u32 ldsKeys[WG_SIZE];
	__shared__ u32 ldsTemp[WG_SIZE][16];
	__shared__ u32 ldsTemp1[16];
	__shared__ int ldsHistogram[BIN_SIZE];
	__shared__ int ldsScanned[BIN_SIZE];

	for( int i = threadIdx.x; i < BIN_SIZE; i += WG_SIZE )
	{
		localOffsets[i] = gHistogram[i * N_WGS_EXECUTED + wgIdx];
	}
	LDS_BARRIER;

	for( int ii = 0; ii < gNItemsPerWI; ii++ )
	{
		int idx = offset + threadIdx.x * gNItemsPerWI + ii;

		int key;
		int tableIdx;
		ldsKeys[threadIdx.x] = 0xffffffff;
		if( idx < gN )
		{
			key = gSrc[idx];
			ldsKeys[threadIdx.x] = key;
		}

		localSort8bit( ldsKeys, ldsTemp, ldsTemp1, START_BIT );

		for(int i = threadIdx.x; i<BIN_SIZE; i+=WG_SIZE)
			ldsHistogram[i] = 0;
		LDS_BARRIER;
		{
			int tableIdx = ( ldsKeys[threadIdx.x] >> START_BIT ) & RADIX_MASK;
			atomicAdd( &ldsHistogram[tableIdx], 1 );
		}
		LDS_BARRIER;
		if( threadIdx.x == 0 )//todo. parallel scan
		{
			int sum = 0;
			for(int i=0; i<BIN_SIZE; i++)
			{
				int t = ldsHistogram[i];
				ldsScanned[i] = sum;
				sum += t;
			}
		}
		LDS_BARRIER;
		if( idx < gN )
		{
			int tableIdx = ( ldsKeys[threadIdx.x] >> START_BIT ) & RADIX_MASK;
			int dstIdx = localOffsets[tableIdx] + threadIdx.x - ldsScanned[tableIdx];
			gDst[dstIdx] = ldsKeys[threadIdx.x];
		}
		LDS_BARRIER;
		for( int i = threadIdx.x; i < BIN_SIZE; i += WG_SIZE )
		{
			localOffsets[i] += ldsHistogram[i];//todo. ldsHistogram can be private
		}
	}
}

__device__ void localSort4bit( int* keys, u32* ldsKeys, u32 ldsTemp[WG_SIZE][N_BINS_4BIT], u32 ldsTemp1[N_BINS_4BIT], const int START_BIT )
{
	int key = keys[0];
	int in4bit = ( key >> START_BIT ) & 0xf;
	for( int i = 0; i < N_BINS_4BIT; i++ )
	{
		ldsTemp[threadIdx.x][i] = 0;
	}
	ldsTemp[threadIdx.x][in4bit] = 1;

	LDS_BARRIER;

	if( threadIdx.x < N_BINS_4BIT )
	{
		int sum = 0;
		for( int i = 0; i < WG_SIZE; i++ )
		{
			int t = ldsTemp[i][threadIdx.x];
			ldsTemp[i][threadIdx.x] = sum;
			sum += t;
		}
		ldsTemp1[threadIdx.x] = sum;
	}
	LDS_BARRIER;
	if( threadIdx.x == 0 )
	{
		int sum = 0;
		for( int i = 0; i < N_BINS_4BIT; i++ )
		{
			int t = ldsTemp1[i];
			ldsTemp1[i] = sum;
			sum += t;
		}
	}
	LDS_BARRIER;

	int offset = ldsTemp1[in4bit];
	int rank = ldsTemp[threadIdx.x][in4bit];

	ldsKeys[offset + rank] = key;

	LDS_BARRIER;

	keys[0] = ldsKeys[threadIdx.x];
}

__device__ void localSort8bit( int* keys, u32* ldsKeys, u32 ldsTemp[WG_SIZE][N_BINS_4BIT], u32 ldsTemp1[N_BINS_4BIT], const int START_BIT )
{
	localSort4bit( keys, ldsKeys, ldsTemp, ldsTemp1, START_BIT );
	if( N_RADIX > 4 ) localSort4bit( keys, ldsKeys, ldsTemp, ldsTemp1, START_BIT + 4 );
}

extern "C" __global__ void SortKernel1( int* gSrc, int* gDst, int* gHistogram, int gN, int gNItemsPerWI, const int START_BIT, const int N_WGS_EXECUTED )
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int offset = blockIdx.x * blockDim.x * gNItemsPerWI;
	const int wgIdx = blockIdx.x;

	__shared__ u32 localOffsets[BIN_SIZE];
	__shared__ u32 ldsKeys[WG_SIZE];//todo. can be aliased
	__shared__ u32 ldsTemp[WG_SIZE][N_BINS_4BIT];
	__shared__ u32 ldsTemp1[N_BINS_4BIT];

	__shared__ int ldsHistogram[BIN_SIZE];//todo. can be aliased

	int histogram[N_BINS_PER_WI] = {0};
	int keys = 0;

	for( int i = threadIdx.x; i < BIN_SIZE; i += WG_SIZE )
	{
		localOffsets[i] = gHistogram[i * N_WGS_EXECUTED + wgIdx];
	}
	LDS_BARRIER;

	for( int ii = 0; ii < gNItemsPerWI; ii++ )
	{
		int idx = offset + threadIdx.x * gNItemsPerWI + ii;

		int tableIdx;
		keys = 0xffffffff;
		if( idx < gN )
		{
			keys = gSrc[idx];
		}

		localSort8bit( &keys, ldsKeys, ldsTemp, ldsTemp1, START_BIT );

		for( int i = threadIdx.x; i < BIN_SIZE; i += WG_SIZE )
			ldsHistogram[i] = 0;
		LDS_BARRIER;
		{
			int tableIdx = ( keys >> START_BIT ) & RADIX_MASK;
			atomicAdd( &ldsHistogram[tableIdx], 1 );
		}
		LDS_BARRIER;
		for(int i=0; i<N_BINS_PER_WI; i++)
		{
			histogram[i] = ldsHistogram[threadIdx.x * N_BINS_PER_WI + i];
		}

		if( threadIdx.x == 0 ) // todo. parallel scan
		{
			int sum = 0;
			for( int i = 0; i < BIN_SIZE; i++ )
			{
				int t = ldsHistogram[i];
				ldsHistogram[i] = sum;
				sum += t;
			}
		}
		LDS_BARRIER;
		if( idx < gN )
		{
			int tableIdx = ( keys >> START_BIT ) & RADIX_MASK;
			int dstIdx = localOffsets[tableIdx] + threadIdx.x - ldsHistogram[tableIdx];
			gDst[dstIdx] = keys;
		}
		LDS_BARRIER;

		for( int i = 0; i < N_BINS_PER_WI; i++ )
		{
			int idx = threadIdx.x * N_BINS_PER_WI + i;
			localOffsets[idx] += histogram[i];
		}

	}
}

//========================

__device__ void localSort4bitMulti( int* keys, u32* ldsKeys, u32 ldsTemp[WG_SIZE][N_BINS_4BIT], u32 ldsTemp1[N_BINS_4BIT], const int START_BIT )
{
	for( int i = 0; i < N_BINS_4BIT; i++ )
	{
		ldsTemp[threadIdx.x][i] = 0;
	}
	for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
	{
		int in4bit = ( keys[i] >> START_BIT ) & 0xf;
		ldsTemp[threadIdx.x][in4bit] += 1;
	}

	LDS_BARRIER;

	if( threadIdx.x < N_BINS_4BIT )//16 scans, pack 4 scans into 1 to make 4 parallel scans
	{
		int sum = 0;
		for( int i = 0; i < WG_SIZE; i++ )
		{
			int t = ldsTemp[i][threadIdx.x];
			ldsTemp[i][threadIdx.x] = sum;
			sum += t;
		}
		ldsTemp1[threadIdx.x] = sum;
	}
	LDS_BARRIER;
	if( threadIdx.x == 0 )//todo parallel scan
	{
		int sum = 0;
		for( int i = 0; i < N_BINS_4BIT; i++ )
		{
			int t = ldsTemp1[i];
			ldsTemp1[i] = sum;
			sum += t;
		}
	}
	LDS_BARRIER;

	for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
	{
		int in4bit = ( keys[i] >> START_BIT ) & 0xf;
		int offset = ldsTemp1[in4bit];
		int rank = ldsTemp[threadIdx.x][in4bit]++;

		ldsKeys[offset + rank] = keys[i];
	}
	LDS_BARRIER;

	for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
	{
		keys[i] = ldsKeys[threadIdx.x * SORT_N_ITEMS_PER_WI + i];
	}
}

__device__ void localSort8bitMulti( int* keys, u32* ldsKeys, u32 ldsTemp[WG_SIZE][N_BINS_4BIT], u32 ldsTemp1[N_BINS_4BIT], const int START_BIT )
{
	localSort4bitMulti( keys, ldsKeys, ldsTemp, ldsTemp1, START_BIT );
	if( N_RADIX > 4 ) localSort4bitMulti( keys, ldsKeys, ldsTemp, ldsTemp1, START_BIT + 4 );
}

extern "C" __global__ void SortKernel2( int* gSrc, int* gDst, int* gHistogram, int gN, int gNItemsPerWI, const int START_BIT, const int N_WGS_EXECUTED )
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockIdx.x * blockDim.x * gNItemsPerWI;
	const int wgIdx = blockIdx.x;

	__shared__ u32 localOffsets[BIN_SIZE];

	__shared__ u32 ldsKeys[WG_SIZE * SORT_N_ITEMS_PER_WI]; // todo. can be aliased
	__shared__ u32 ldsTemp[WG_SIZE][N_BINS_4BIT];
	__shared__ u32 ldsTemp1[N_BINS_4BIT];

	__shared__ int ldsHistogram[BIN_SIZE]; // todo. can be aliased

	int histogram[N_BINS_PER_WI] = { 0 };
	int keys[SORT_N_ITEMS_PER_WI] = { 0 };

	for( int i = threadIdx.x; i < BIN_SIZE; i += WG_SIZE )
	{
		localOffsets[i] = gHistogram[i * N_WGS_EXECUTED + wgIdx];
	}
	LDS_BARRIER;

	for( int ii = 0; ii < gNItemsPerWI; ii += SORT_N_ITEMS_PER_WI )
	{
		for(int i=0; i<SORT_N_ITEMS_PER_WI; i++)
		{
			int idx = offset + threadIdx.x * SORT_N_ITEMS_PER_WI + i;
			keys[i] = (idx<gN)? gSrc[idx] : 0xffffffff;
		}

		//local sort keys[];
		localSort8bitMulti( keys, ldsKeys, ldsTemp, ldsTemp1, START_BIT );
#if 0
		if( THE_FIRST_THREAD )
		{
			for( int i = 0; i < WG_SIZE * SORT_N_ITEMS_PER_WI ; i++)
				printf("%d,", ldsKeys[i]);
			printf("\n");
		}
		break;
#endif	
		for( int i = threadIdx.x; i < BIN_SIZE; i += WG_SIZE )
			ldsHistogram[i] = 0;
		LDS_BARRIER;
		for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
		{
			int tableIdx = ( keys[i] >> START_BIT ) & RADIX_MASK;
			atomicAdd( &ldsHistogram[tableIdx], 1 );
		}
		LDS_BARRIER;
		for( int i = 0; i < N_BINS_PER_WI; i++ )
		{
			histogram[i] = ldsHistogram[threadIdx.x * N_BINS_PER_WI + i];
		}

		if( threadIdx.x == 0 ) // todo. parallel scan
		{
			int sum = 0;
			for( int i = 0; i < BIN_SIZE; i++ )
			{
				int t = ldsHistogram[i];
				ldsHistogram[i] = sum;
				sum += t;
			}
		}
		LDS_BARRIER;
		for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
		{
			int idx = offset + threadIdx.x * SORT_N_ITEMS_PER_WI + i;
			if( idx < gN )
			{
				int tableIdx = ( keys[i] >> START_BIT ) & RADIX_MASK;
				int dstIdx = localOffsets[tableIdx] + (threadIdx.x*SORT_N_ITEMS_PER_WI+i) - ldsHistogram[tableIdx];
				gDst[dstIdx] = keys[i];
			}
		}
		LDS_BARRIER;

		for( int i = 0; i < N_BINS_PER_WI; i++ )
		{
			int idx = threadIdx.x * N_BINS_PER_WI + i;
			localOffsets[idx] += histogram[i];
		}
		//===
		offset += WG_SIZE * SORT_N_ITEMS_PER_WI;
	}
}