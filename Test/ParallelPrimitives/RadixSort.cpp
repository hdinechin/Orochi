#include <Test/OrochiUtils.h>
#include <Test/ParallelPrimitives/RadixSort.h>
#include <Test/ParallelPrimitives/RadixSortConfigs.h>
#include <numeric>

namespace
{

/// @brief Exclusive scan algorithm on CPU for testing.
/// It copies the count result from the Device to Host before computation, and then copies the offsets back from Host to Device afterward.
/// @param countsGpu The count result in GPU memory. Otuput: The offset.
/// @param offsetsGpu The offsets.
/// @param nWGsToExecute Number of WGs to execute
void exclusiveScanCpu( int* countsGpu, int* offsetsGpu, const int nWGsToExecute ) noexcept
{
	std::vector<int> counts( BIN_SIZE * nWGsToExecute );
	OrochiUtils::copyDtoH( counts.data(), countsGpu, BIN_SIZE * nWGsToExecute );
	OrochiUtils::waitForCompletion();

#if 0
	for( int j = 0; j < nWGsToExecute; j++ )
	{
		for( int i = 0; i < BIN_SIZE; i++ )
		{
			printf( "%d, ", counts[j * BIN_SIZE + i] );
		}
		printf( "\n" );
	}
#endif

	std::vector<int> offsets( BIN_SIZE * nWGsToExecute );
	std::exclusive_scan( std::cbegin( counts ), std::cend( counts ), std::begin( offsets ), 0 );

	OrochiUtils::copyHtoD( offsetsGpu, offsets.data(), BIN_SIZE * nWGsToExecute );
	OrochiUtils::waitForCompletion();
}

} // namespace

namespace Oro
{

struct RadixSortImpl
{
	static void printKernelInfo( oroFunction func )
	{
		int a, b, c;
		oroFuncGetAttribute( &a, ORO_FUNC_ATTRIBUTE_NUM_REGS, func );
		oroFuncGetAttribute( &b, ORO_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func );
		oroFuncGetAttribute( &c, ORO_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, func );
		printf( "vgpr : shared = %d : %d : %d\n", a, b, c );
	}
};

RadixSort::RadixSort() : m_nWGsToExecute{ 4 } {}

RadixSort::~RadixSort() = default;

void RadixSort::sort( int* src, int* dst, int n, int startBit, int endBit )
{
	constexpr bool reference = false;
	int a = N_PACKED_PER_WI;
	// allocate temps
	// clear temps
	// count kernel
	// scan
	// sort
	int* temps;
	OrochiUtils::malloc( temps, BIN_SIZE * m_nWGsToExecute );
	OrochiUtils::memset( temps, 0, BIN_SIZE * m_nWGsToExecute * sizeof( int ) );

	const int nWIs = WG_SIZE * m_nWGsToExecute;
	int nItemsPerWI = ( n + ( nWIs - 1 ) ) / nWIs;
	printf( "nWGs: %d\n", m_nWGsToExecute );
	printf( "nNItemsPerWI: %d\n", nItemsPerWI );

	constexpr auto kernalPath{ "../Test/ParallelPrimitives/RadixSortKernels.h" };

	{
		// Count
		constexpr auto funcName{ reference ? "CountKernelReference" : "CountKernel" };
		oroFunction func = OrochiUtils::getFunctionFromFile( kernalPath, funcName, nullptr );
		RadixSortImpl::printKernelInfo( func );

		const void* args[] = { &src, &temps, &n, &nItemsPerWI, &startBit, &m_nWGsToExecute };
		OrochiUtils::launch1D( func, WG_SIZE * m_nWGsToExecute, args, WG_SIZE );
		OrochiUtils::waitForCompletion();
	}

	{
		// Exclusive scan
		exclusiveScanCpu( temps, temps, m_nWGsToExecute );
	}

	{
		// Sort
		constexpr auto funcName{ reference ? "SortKernelReference" : "SortKernel2" };
		oroFunction func = OrochiUtils::getFunctionFromFile( kernalPath, funcName, nullptr );
		RadixSortImpl::printKernelInfo( func );
		const void* args[] = { &src, &dst, &temps, &n, &nItemsPerWI, &startBit, &m_nWGsToExecute };
		OrochiUtils::launch1D( func, WG_SIZE * m_nWGsToExecute, args, WG_SIZE );
		OrochiUtils::waitForCompletion();
	}

	printf( "sort completed\n" );
}

}; // namespace Oro
