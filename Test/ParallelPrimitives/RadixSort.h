#pragma once

#include <Orochi/Orochi.h>

namespace Oro
{

class RadixSort
{
  public:
	typedef unsigned int u32;

	enum Flag
	{
		FLAG_LOG = 1<<0,
	};

	  RadixSort();

	  ~RadixSort();

	  void configure( oroDevice device, u32& tempBufferSizeOut );

	  void setFlag( Flag flag );

	  void sort( u32* src, u32* dst, int n, int startBit, int endBit, u32* tempBuffer );

  private:
	  void sort1pass( u32* src, u32* dst, int n, int startBit, int endBit, int* tmps );

  private:
	  int m_nWGsToExecute;
	  Flag m_flags;

};

};
