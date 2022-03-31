#pragma once

namespace Oro
{

class RadixSort
{
  public:
	  RadixSort();

	  ~RadixSort();

	  void sort( int* src, int* dst, int n, int startBit, int endBit );

  private:
	  int m_nWGsToExecute;

};

};
