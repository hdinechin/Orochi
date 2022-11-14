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
#include <Orochi/OrochiUtils.h>
#include <Test/Common.h>
#include "half.h"


int main(int argc, char** argv )
{
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
		printf("executing on %s (%s)\n", props.name, props.gcnArchName );
		int v;
		oroDriverGetVersion( &v );
		printf("running on driver: %d\n", v);
	}
	printf(">> testing kernel execution\n");
	{
		oroFunction function;
		{
			const char* code = "extern \"C\" __global__ void testKernel( "
				" float* arg1_in "
				",float* arg2_out "
				",half*  arg3_in "
				",half*  arg4_out "
				",half*  arg5_out "
				")"
			"{"
				"int tid = threadIdx.x;"
				"arg2_out[tid] = arg1_in[tid];"
				"arg4_out[tid] = hcos( arg3_in[tid] );"
				"arg5_out[tid] = hsin( arg3_in[tid] );"
			"}";
			const char* funcName = "testKernel";
			orortcProgram prog;
			orortcResult e;
			e = orortcCreateProgram( &prog, code, funcName, 0, 0, 0 );
			std::vector<const char*> opts; 
			opts.push_back( "-I ../" );

			e = orortcCompileProgram( prog, opts.size(), opts.data() );
			if( e != ORORTC_SUCCESS )
			{
				size_t logSize;
				orortcGetProgramLogSize(prog, &logSize);
				if (logSize) 
				{
					std::string log(logSize, '\0');
					orortcGetProgramLog(prog, &log[0]);
					std::cout << log << '\n';
				};
			}
			size_t codeSize;
			e = orortcGetCodeSize(prog, &codeSize);

			std::vector<char> codec(codeSize);
			e = orortcGetCode(prog, codec.data());
			e = orortcDestroyProgram(&prog);
			oroModule module;
			oroError ee = oroModuleLoadData(&module, codec.data());
			ee = oroModuleGetFunction(&function, module, funcName);
		}

		oroStream stream;
		oroStreamCreate( &stream );
		
		oroEvent start, stop;
		oroEventCreateWithFlags( &start, 0 );
		oroEventCreateWithFlags( &stop, 0 );
		oroEventRecord( start, stream );


		const int floatCount = 5;

		float* arg1_host = nullptr;
		oroDeviceptr arg1_device = 0;
		size_t  arg1_size = sizeof(float) * floatCount;

		float* arg2_host = nullptr;
		oroDeviceptr arg2_device = 0;
		size_t  arg2_size = sizeof(float) * floatCount;

		OrochiTests::half* arg3_host = nullptr;
		oroDeviceptr arg3_device = 0;
		size_t  arg3_size = sizeof(OrochiTests::half) * floatCount;
		
		OrochiTests::half* arg4_host = nullptr;
		oroDeviceptr arg4_device = 0;
		size_t  arg4_size = sizeof(OrochiTests::half) * floatCount;

		OrochiTests::half* arg5_host = nullptr;
		oroDeviceptr arg5_device = 0;
		size_t  arg5_size = sizeof(OrochiTests::half) * floatCount;


		arg1_host = (float*)malloc(arg1_size);
		arg2_host = (float*)malloc(arg2_size);
		arg3_host = (OrochiTests::half*)malloc(arg3_size);
		arg4_host = (OrochiTests::half*)malloc(arg4_size);
		arg5_host = (OrochiTests::half*)malloc(arg5_size);


		ERROR_CHECK(oroMalloc(&arg1_device, arg1_size));
		ERROR_CHECK(oroMalloc(&arg2_device, arg2_size));
		ERROR_CHECK(oroMalloc(&arg3_device, arg3_size));
		ERROR_CHECK(oroMalloc(&arg4_device, arg4_size));
		ERROR_CHECK(oroMalloc(&arg5_device, arg5_size));

		for(int i=0; i<floatCount; i++)
		{
			arg1_host[i] = (float)i * 10.0f;
			arg2_host[i] = -1.0f;

			arg3_host[i] = (float)i * 2.0f / (float)floatCount;
			arg4_host[i] = -1.0f;
			arg5_host[i] = -1.0f;
		}


		// Copy host vectors to device
		ERROR_CHECK(oroMemcpyHtoD( arg1_device, arg1_host, arg1_size));
		ERROR_CHECK(oroMemcpyHtoD( arg3_device, arg3_host, arg3_size));


		const void* args[] = { 
			&arg1_device , 
			&arg2_device ,
			&arg3_device ,
			&arg4_device ,
			&arg5_device ,
			};

		// execute kernel
		OrochiUtils::launch1D( function, floatCount, args, floatCount );
		oroEventRecord( stop, stream );
		oroDeviceSynchronize();
		oroStreamDestroy( stream );

		// Copy device to host 
		ERROR_CHECK(oroMemcpyDtoH( arg2_host ,  arg2_device, arg2_size));
		ERROR_CHECK(oroMemcpyDtoH( arg4_host ,  arg4_device, arg4_size));
		ERROR_CHECK(oroMemcpyDtoH( arg5_host ,  arg5_device, arg5_size));


		printf("\n === OUTPUTS === \n" );

		printf( "\narg2 = \n" );
		for(int i=0; i<floatCount; i++)
		{
			printf("[%d]=%f\n" , i , (float)arg2_host[i]  );
		}

		printf( "\narg4 = \n" );
		for(int i=0; i<floatCount; i++)
		{
			printf("[%d]=%f\n" , i , (float)arg4_host[i]  );
		}

		printf( "\narg5 = \n" );
		for(int i=0; i<floatCount; i++)
		{
			printf("[%d]=%f\n" , i , (float)arg5_host[i]  );
		}

		printf("\n\n" );



		ERROR_CHECK(oroFree(arg1_device));
		ERROR_CHECK(oroFree(arg2_device));
		ERROR_CHECK(oroFree(arg3_device));
		ERROR_CHECK(oroFree(arg4_device));
		ERROR_CHECK(oroFree(arg5_device));


		free(arg1_host); arg1_host = nullptr;
		free(arg2_host); arg2_host = nullptr;
		free(arg3_host); arg3_host = nullptr;
		free(arg4_host); arg4_host = nullptr;
		free(arg5_host); arg5_host = nullptr;


		float milliseconds = 0.0f;
		oroEventElapsedTime( &milliseconds, start, stop );
		printf( ">> kernel - %.5f ms\n", milliseconds );
		oroEventDestroy( start );
		oroEventDestroy( stop );
	}
	printf(">> done\n");
	return 0;
}
