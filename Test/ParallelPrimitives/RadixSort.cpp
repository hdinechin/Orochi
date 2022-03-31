#include <Test/ParallelPrimitives/RadixSort.h>
#include <Test/ParallelPrimitives/RadixSortConfigs.h>
#include <Test/OrochiUtils.h>

namespace Oro
{

struct RadixSortImpl
{
	static
	void printKernelInfo( oroFunction func ) 
	{ 
		int a, b, c;
		oroFuncGetAttribute( &a, ORO_FUNC_ATTRIBUTE_NUM_REGS, func );
		oroFuncGetAttribute( &b, ORO_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func );
		oroFuncGetAttribute( &c, ORO_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, func );
		printf( "vgpr : shared = %d : %d : %d\n", a, b, c );
	}

};

RadixSort::RadixSort() 
{ 
	m_nWGsToExecute = 4;
}

RadixSort::~RadixSort()
{ 
}

void RadixSort::sort( int* src, int* dst, int n, int startBit, int endBit )
{
	const bool reference = false;
	int a = N_PACKED_PER_WI;
	//allocate temps
	//clear temps
	//count kernel
	//scan
	//sort
	int* temps;
	OrochiUtils::malloc( temps, BIN_SIZE * m_nWGsToExecute );
	OrochiUtils::memset( temps, 0, BIN_SIZE * m_nWGsToExecute * sizeof(int) );

	const int nWIs = WG_SIZE * m_nWGsToExecute;
	int nItemsPerWI = (n + (nWIs-1))/nWIs;
	printf("nWGs: %d\n",m_nWGsToExecute);
	printf("nNItemsPerWI: %d\n", nItemsPerWI);
	{
		oroFunction func = OrochiUtils::getFunctionFromFile( "../Test/ParallelPrimitives/RadixSortKernels.h", 
			reference?"CountKernelReference":"CountKernel", 0 );
		RadixSortImpl::printKernelInfo( func );
		
		const void* args[] = { &src, &temps, &n, &nItemsPerWI, &startBit, &m_nWGsToExecute };
		OrochiUtils::launch1D( func, WG_SIZE *m_nWGsToExecute, args, WG_SIZE );
		OrochiUtils::waitForCompletion();
	}

	{//exclusive scan
		int* h = new int[BIN_SIZE * m_nWGsToExecute];
		OrochiUtils::copyDtoH( h, temps, BIN_SIZE * m_nWGsToExecute );
		OrochiUtils::waitForCompletion();
#if 0
		for( int j = 0; j < m_nWGsToExecute; j++)
		{
			for( int i = 0; i < BIN_SIZE; i++ )
			{
				printf( "%d, ", h[j * BIN_SIZE + i] );
			}
			printf("\n");
		}
#endif
		int sum = 0;
		for( int i = 0; i < BIN_SIZE * m_nWGsToExecute; i++)
		{
			int t = h[i];
			h[i] = sum;
			sum += t;
		}

		OrochiUtils::copyHtoD( temps, h, BIN_SIZE * m_nWGsToExecute );
		OrochiUtils::waitForCompletion();
		delete[] h;
	}
	if(reference)
	{
		oroFunction func = OrochiUtils::getFunctionFromFile( "../Test/ParallelPrimitives/RadixSortKernels.h", 
			reference?"SortKernelReference":"SortKernel1", 0 );
		const void* args[] = { &src, &dst, &temps, &n, &nItemsPerWI, &startBit, &m_nWGsToExecute };
		OrochiUtils::launch1D( func, WG_SIZE * m_nWGsToExecute, args, WG_SIZE );
		OrochiUtils::waitForCompletion();
	}
	else
	{
		oroFunction func = OrochiUtils::getFunctionFromFile( "../Test/ParallelPrimitives/RadixSortKernels.h", reference ? "SortKernelReference" : "SortKernel2", 0 );
		RadixSortImpl::printKernelInfo( func );
		const void* args[] = { &src, &dst, &temps, &n, &nItemsPerWI, &startBit, &m_nWGsToExecute };
		OrochiUtils::launch1D( func, WG_SIZE * m_nWGsToExecute, args, WG_SIZE );
		OrochiUtils::waitForCompletion();	
	}

	int* h = new int[n];
	OrochiUtils::copyDtoH( h, dst, n );
	OrochiUtils::waitForCompletion();

	printf("xx\n");
}

};
