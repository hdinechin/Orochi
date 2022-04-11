//
// Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#include <Orochi/Orochi.h>
#include <Test/Common.h>

#include <Test/OrochiUtils.h>
#include <Test/ParallelPrimitives/RadixSort.h>
#include <Test/ParallelPrimitives/RadixSortConfigs.h>
#include <algorithm>
#include <vector>

#include <chrono>

class Stopwatch
{
  public:
	void start() { m_start = std::chrono::system_clock::now(); }
	void stop() { m_end = std::chrono::system_clock::now(); }
	float getMs() 
	{
		return  std::chrono::duration_cast<std::chrono::milliseconds>( m_end - m_start ).count();
	}

	private:
	  std::chrono::time_point<std::chrono::system_clock> m_start, m_end;

};

#define u64 unsigned long long
#define u32 unsigned int

class SortTest
{
  public:
	SortTest( oroDevice dev, oroCtx ctx ) : m_device( dev ), m_ctx( ctx ) 
	{ 
		m_sort = new Oro::RadixSort(); 
		m_sort->configure( m_device );
//		m_sort->setFlag( Oro::RadixSort::FLAG_LOG );
	}

	void test( int testSize, const int testBits = 32, const int nRuns = 1 )
	{
		using namespace std;
		srand( 123 );
		vector<u32> src( testSize );
		for( int i = 0; i < testSize; i++ )
		{
			src[i] = getRandom( 0u, (u32)(( 1ull << (u64)testBits ) - 1) );
		}

		u32* srcGpu;
		u32* dstGpu;
		OrochiUtils::malloc( srcGpu, testSize );
		OrochiUtils::malloc( dstGpu, testSize );

		Stopwatch sw;
		for( int i = 0; i < nRuns ; i++)
		{
			OrochiUtils::copyHtoD( srcGpu, src.data(), testSize );
			OrochiUtils::waitForCompletion();
			sw.start();
			m_sort->sort( srcGpu, dstGpu, testSize, 0, testBits );
			OrochiUtils::waitForCompletion();
			sw.stop();
			printf("%3.2fms\n", sw.getMs());
		}

		vector<u32> dst( testSize );
		OrochiUtils::copyDtoH( dst.data(), dstGpu, testSize );

		std::sort( src.begin(), src.end() );
		for( int i = 0; i < testSize; i++ )
		{
			if( dst[i] != src[i] ) 
			{
				printf( "fail\n" );
				__debugbreak();
				break;
			}
		}

		OrochiUtils::free( srcGpu );
		OrochiUtils::free( dstGpu );
		printf("passed: %3.2fK keys\n", testSize/1000.f);
	}

	template<typename T>
	inline T getRandom( const T minV, const T maxV )
	{
		double r = std::min( (double)RAND_MAX - 1, (double)rand() ) / RAND_MAX;
		T range = maxV - minV;
		return ( T )( minV + r * range );
	}

  private:
	oroDevice m_device;
	oroCtx m_ctx;
	Oro::RadixSort* m_sort;
};

enum TestType
{
	TEST_SIMPLE,
	TEST_PERF,
	TEST_BITS,
};

int main(int argc, char** argv )
{
	TestType testType = TEST_BITS;
	oroApi api = getApiType( argc, argv );

	int a = oroInitialize( api, 0 );
	if( a != 0 )
	{
		printf("initialization failed\n");
		return 0;
	}
	printf( ">> executing on %s\n", ( api == ORO_API_HIP )? "hip":"cuda" );

	printf(">> testing initialization\n");
	oroError e;
	e = oroInit( 0 );
	oroDevice device;
	e = oroDeviceGet( &device, 0 );
	oroCtx ctx;
	e = oroCtxCreate( &ctx, 0, device );


	printf(">> testing device props\n");
	{
		oroDeviceProp props;
		oroGetDeviceProperties( &props, device );
		printf( "executing on %s (%s), %d SIMDs\n", props.name, props.gcnArchName, props.multiProcessorCount );
	}

	SortTest sort( device, ctx );
	const int testBits = 16;
	switch( testType )
	{
	case TEST_SIMPLE:
		sort.test( 64 * 100, testBits );
		break;
	case TEST_PERF:
		sort.test( 64 * 100 * 10, testBits );
		sort.test( 64 * 100 * 100, testBits );
		sort.test( 64 * 100 * 1000, testBits );
		break;
	case TEST_BITS:
	{
		int testSize = 64*1000;
		sort.test( testSize, 8 );
		sort.test( testSize, 16 );
		sort.test( testSize, 24 );
		sort.test( testSize, 32 );
	}
		break;
	};

	printf(">> done\n");
	return 0;
}
