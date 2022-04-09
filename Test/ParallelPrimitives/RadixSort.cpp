#include <Test/OrochiUtils.h>
#include <Test/ParallelPrimitives/RadixSort.h>
#include <Test/ParallelPrimitives/RadixSortConfigs.h>
#include <cassert>
#include <chrono>
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
	std::vector<int> counts( Oro::BIN_SIZE * nWGsToExecute );
	OrochiUtils::copyDtoH( counts.data(), countsGpu, Oro::BIN_SIZE * nWGsToExecute );
	OrochiUtils::waitForCompletion();

	constexpr auto ENABLE_PRINT{ false };

	if constexpr( ENABLE_PRINT )
	{
		for( int j = 0; j < nWGsToExecute; j++ )
		{
			for( int i = 0; i < Oro::BIN_SIZE; i++ )
			{
				printf( "%d, ", counts[j * Oro::BIN_SIZE + i] );
			}
			printf( "\n" );
		}
	}

	std::vector<int> offsets( Oro::BIN_SIZE * nWGsToExecute );
	std::exclusive_scan( std::cbegin( counts ), std::cend( counts ), std::begin( offsets ), 0 );

	OrochiUtils::copyHtoD( offsetsGpu, offsets.data(), Oro::BIN_SIZE * nWGsToExecute );
	OrochiUtils::waitForCompletion();
}

template<class Func>
inline void measureTime( Func&& callable ) noexcept
{
	const auto t1 = std::chrono::high_resolution_clock::now();

	std::invoke( std::forward<Func>( callable ) );

	const auto t2 = std::chrono::high_resolution_clock::now();
	const auto ms_int = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 );

	printf( "Scan Execution time: %lld us \n", ms_int.count() );
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
		printf( "vgpr : shared : const = %d : %d : %d\n", a, b, c );
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
	int* temps = nullptr;
	OrochiUtils::malloc( temps, BIN_SIZE * m_nWGsToExecute );
	OrochiUtils::memset( temps, 0, BIN_SIZE * m_nWGsToExecute * sizeof( int ) );

	int* partialSum = nullptr;
	OrochiUtils::malloc( partialSum, m_nWGsToExecute );
	OrochiUtils::memset( partialSum, 0, m_nWGsToExecute * sizeof( int ) );

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

	constexpr auto enableGPUParallelScan = true;

	if constexpr( enableGPUParallelScan )
	{

		const auto useSingleWG = false;

		if( useSingleWG )
		{
			// Parallel Exclusive Scan using GPU with single WG.
			constexpr auto funcNameScan{ "ParallelExclusiveScan" };
			oroFunction func = OrochiUtils::getFunctionFromFile( kernalPath, funcNameScan, nullptr );
			RadixSortImpl::printKernelInfo( func );

			measureTime(
				[&]()
				{
					const void* args[] = { &temps, &temps, &m_nWGsToExecute };
					OrochiUtils::launch1D( func, WG_SIZE * m_nWGsToExecute, args, WG_SIZE );
					OrochiUtils::waitForCompletion();
				} );
		}
		else
		{
			// Parallel Exclusive Scan using GPU.
			constexpr auto funcNameScan{ "ParallelExclusiveScanAdv" };
			oroFunction funcScan = OrochiUtils::getFunctionFromFile( kernalPath, funcNameScan, nullptr );
			RadixSortImpl::printKernelInfo( funcScan );

			constexpr auto funcNameOffset{ "ApplyGlobalOffset" };
			oroFunction funcOffset = OrochiUtils::getFunctionFromFile( kernalPath, funcNameOffset, nullptr );
			RadixSortImpl::printKernelInfo( funcOffset );

			measureTime(
				[&]()
				{
					{

						const void* args[] = { &temps, &temps, &partialSum };
						OrochiUtils::launch1D( funcScan, WG_SIZE * m_nWGsToExecute, args, WG_SIZE );
					}

					{

						const void* args[] = { &temps, &partialSum };
						OrochiUtils::launch1D( funcOffset, WG_SIZE * m_nWGsToExecute, args, WG_SIZE );
					}

					OrochiUtils::waitForCompletion();
				} );
		}
	}
	else
	{
		// Exclusive scan using CPU

		measureTime( [&]() { exclusiveScanCpu( temps, temps, m_nWGsToExecute ); } );
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
