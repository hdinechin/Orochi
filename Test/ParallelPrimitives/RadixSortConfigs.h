#pragma once

namespace Oro
{
#define WG_SIZE 64
#define N_RADIX 8
#define BIN_SIZE ( 1 << N_RADIX )
#define RADIX_MASK ( ( 1 << N_RADIX ) - 1 )
#define PACK_FACTOR ( sizeof( int ) / sizeof( char ) )
#define N_PACKED ( BIN_SIZE / PACK_FACTOR )
#define PACK_MAX 255
#define N_PACKED_PER_WI ( N_PACKED / WG_SIZE )
#define N_BINS_PER_WI ( BIN_SIZE / WG_SIZE )
#define N_BINS_4BIT 16
#define N_BINS_PACK_FACTOR ( sizeof( long long ) / sizeof( short ) )
#define N_BINS_PACKED_4BIT ( N_BINS_4BIT / N_BINS_PACK_FACTOR )

// sort configs
#define SORT_N_ITEMS_PER_WI 8

// Scan configs

// This number must match the number of WGs.
#define NUM_COUNTS_PER_BIN 4

}; // namespace Oro
