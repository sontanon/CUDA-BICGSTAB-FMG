#include <assert.h>

int IDX(const int r, const int z, const int NrTotal, const int NzTotal)
{
	// Check for overflow: uncomment if sure.
	assert(r < NrTotal);
	assert(z < NzTotal);
	return r * NzTotal + z;
}

int nnz_calculator(const int NrInterior, const int NzInterior)
{
	return 5 * NrInterior * NzInterior + 6 * NrInterior + 6 * NzInterior + 16;
}
