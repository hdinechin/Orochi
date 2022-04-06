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
#include <chrono>
#include <optional>
#include <random>
#include <vector>

class SortTest
{
  public:
	SortTest( oroDevice dev, oroCtx ctx ) : m_device( dev ), m_ctx( ctx ), m_sort{ std::make_optional<Oro::RadixSort>() } {}

	void test( int testSize )
	{
		std::vector<int> src( testSize );
		std::vector<int> historgram( BIN_SIZE, 0 );

		const auto seed{ std::chrono::system_clock::now().time_since_epoch().count() };
		std::default_random_engine generator( seed );
		std::uniform_int_distribution<int> distribution( 0, ( 1 << N_RADIX ) - 1 );

		// generate random integers within the given range
		for( int i = 0; i < testSize; i++ )
		{
			src[i] = distribution( generator );
			historgram[src[i]]++;
		}

		int* srcGpu;
		int* dstGpu;
		OrochiUtils::malloc( srcGpu, testSize );
		OrochiUtils::malloc( dstGpu, testSize );
		OrochiUtils::copyHtoD( srcGpu, src.data(), testSize );

		m_sort->sort( srcGpu, dstGpu, testSize, 0, 8 );

		std::vector<int> dst( testSize );
		OrochiUtils::copyDtoH( dst.data(), dstGpu, testSize );
		OrochiUtils::waitForCompletion();

		std::sort( src.begin(), src.end() );

		if( !std::equal( cbegin( src ), cend( src ), cbegin( dst ) ) )
		{
			printf( "Test failed\n" );
		}
		else
		{
			printf( "Test passed\n" );
		}
	}

  private:
	oroDevice m_device;
	oroCtx m_ctx;
	std::optional<Oro::RadixSort> m_sort;
};

int main( int argc, char** argv )
{
	oroApi api = getApiType( argc, argv );

	if( int a = oroInitialize( api, 0 ); a != 0 )
	{
		printf( "initialization failed\n" );
		return EXIT_FAILURE;
	}
	printf( ">> executing on %s\n", ( api == ORO_API_HIP ) ? "hip" : "cuda" );

	printf( ">> testing initialization\n" );
	oroError e;
	e = oroInit( 0 );
	oroDevice device;
	e = oroDeviceGet( &device, 0 );
	oroCtx ctx;
	e = oroCtxCreate( &ctx, 0, device );
	OROASSERT( e == oroSuccess, 0 );

	printf( ">> testing device props\n" );
	{
		oroDeviceProp props;
		oroGetDeviceProperties( &props, device );
		printf( "executing on %s (%s)\n", props.name, props.gcnArchName );
	}

	SortTest sort( device, ctx );

	constexpr auto testSize{ 64 * 10 };
	sort.test( testSize );

	printf( ">> done\n" );
	return EXIT_SUCCESS;
}
