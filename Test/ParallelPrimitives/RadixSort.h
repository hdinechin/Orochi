#pragma once

#include <Orochi/Orochi.h>
#include <string>
#include <unordered_map>

namespace Oro
{

class RadixSort
{
  public:
	typedef unsigned int u32;

	enum Flag
	{
		FLAG_LOG = 1 << 0,
	};

	RadixSort();

	void configure( oroDevice device, u32& tempBufferSizeOut );

	void setFlag( Flag flag );

	void sort( u32* src, u32* dst, int n, int startBit, int endBit, u32* tempBuffer );

  private:
	void sort1pass( u32* src, u32* dst, int n, int startBit, int endBit, int* tmps );

	void compileKernels();

  private:
	int m_nWGsToExecute;
	Flag m_flags;

	enum class Kernel
	{
		COUNT,
		COUNT_REF,
		SCAN_SINGLE_WG,
		SCAN_PARALLEL,
		APPLY_OFFSET,
		SORT,
		SORT_REF
	};

	std::unordered_map<Kernel, oroFunction> oroFunctions;

	/// @brief  The enum class which indicates the selected algorithm of prefix scan.
	enum class ScanAlgo
	{
		SCAN_CPU,
		SCAN_GPU_SINGLE_WG,
		SCAN_GPU_PARALLEL,
	};

	constexpr static auto selectedScanAlgo{ ScanAlgo::SCAN_GPU_PARALLEL };

	int* m_partialSum{ nullptr };
};

}; // namespace Oro
