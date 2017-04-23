#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "tools.h"

void csrGenerator(int *csrRowPtrA, int *csrColIndA, double *csrValA, 
		  const int NrInterior, const int NzInterior,
		  const double *h_r, const double *h_z, 
		  const double *h_rr, const double *h_s,
		  double *h_f, const double h, const double uInf)
{
	int NrTotal = NrInterior + 2;
	int NzTotal = NzInterior + 2;

	int nnzA = nnz_calculator(NrInterior, NzInterior);

	// Number of elements we have filled in.
	int offset = 0;

	// Corner average.
	csrRowPtrA[IDX(0, 0, NrTotal, NzTotal)] = offset;
	csrValA[offset] = 1.0;
	csrValA[offset + 1] = -0.375;
	csrValA[offset + 2] = -0.375;
	csrValA[offset + 3] = -0.25;
	csrColIndA[offset] = IDX(0, 0, NrTotal, NzTotal);
	csrColIndA[offset + 1] = IDX(0, 1, NrTotal, NzTotal);
	csrColIndA[offset + 2] = IDX(1, 0, NrTotal, NzTotal);
	csrColIndA[offset + 3] = IDX(1, 1, NrTotal, NzTotal);
	offset += 4;

	// Fill left boundary.
	// We need to fill NzInterior values.
	for (int j = 1; j < NzInterior + 1; j++) {
		// Row begining is at offset.
		csrRowPtrA[IDX(0, j, NrTotal, NzTotal)] = offset;
		// Values.
		csrValA[offset] = 1.0;
		csrValA[offset + 1] = -1.0;
		// Columns.
		csrColIndA[offset] = IDX(0, j, NrTotal, NzTotal);
		csrColIndA[offset + 1] = IDX(1, j, NrTotal, NzTotal);
		// Having filled two values, we increase offset by two.
		offset += 2;
	}

	// Left top corner.
	csrRowPtrA[IDX(0, NzInterior + 1, NrTotal, NzTotal)] = offset;
	csrValA[offset] = -0.375;
	csrValA[offset + 1] = 1.0;
	csrValA[offset + 2] = -0.25;
	csrValA[offset + 3] = -0.375;
	csrColIndA[offset] = IDX(0, NzInterior, NrTotal, NzTotal);
	csrColIndA[offset + 1] = IDX(0, NzInterior + 1, NrTotal, NzTotal);
	csrColIndA[offset + 2] = IDX(1, NzInterior, NrTotal, NzTotal);
	csrColIndA[offset + 3] = IDX(1, NzInterior + 1, NrTotal, NzTotal);
	offset += 4;

	// Start interior filling.
	for (int i = 1; i < NrInterior + 1; i++) {
		for (int j = 0; j < NzInterior + 2; j++) {
			// Check if we are at bottom.
			if (j == 0) {
				// Row begins at offset.
				csrRowPtrA[IDX(i, 0, NrTotal, NzTotal)] = offset;
				// Values.
				csrValA[offset] = 1.0;
				csrValA[offset + 1] = -1.0;
				// Columns.
				csrColIndA[offset] = IDX(i, 0, NrTotal, NzTotal);
				csrColIndA[offset + 1] = IDX(i, 1, NrTotal, NzTotal);
				// Increase offset.
				offset += 2;
			}

			// Check if we are at top.
			else if (j == NzInterior + 1) {
				// 3rd-order ROBIN.
				// Row begins at offset.
				csrRowPtrA[IDX(i, NzInterior + 1, NrTotal, NzTotal)] = offset;
				// Values.
				csrValA[offset] = (1.0 / 6.0) * h / h_z[IDX(i, NzInterior, NrTotal, NzTotal)];
				csrValA[offset + 1] = -h / h_z[IDX(i, NzInterior, NrTotal, NzTotal)];
				csrValA[offset + 2] = 0.5 * h / h_z[IDX(i, NzInterior, NrTotal, NzTotal)];
				csrValA[offset + 3] = (1.0 / 3.0) * h / h_z[IDX(i, NzInterior, NrTotal, NzTotal)] + h * h / (h_rr[IDX(i, NzInterior, NrTotal, NzTotal)] * h_rr[IDX(i, NzInterior, NrTotal, NzTotal)]);
				// Columns.
				csrColIndA[offset] = IDX(i, NzInterior - 2, NrTotal, NzTotal);
				csrColIndA[offset + 1] = IDX(i, NzInterior - 1, NrTotal, NzTotal);
				csrColIndA[offset + 2] = IDX(i, NzInterior, NrTotal, NzTotal);
				csrColIndA[offset + 3] = IDX(i, NzInterior + 1, NrTotal, NzTotal);
				// Increase offset.
				offset += 4;
				// Also fill F source.
				h_f[IDX(i, NzInterior + 1, NrTotal, NzTotal)] = h * h * uInf / (h_rr[IDX(i, NzInterior, NrTotal, NzTotal)] * h_rr[IDX(i, NzInterior, NrTotal, NzTotal)]);
			}

			// Else fill interior points.
			else {
				// Row begins at offset.
				csrRowPtrA[IDX(i, j, NrTotal, NzTotal)] = offset;
				// Values.
				csrValA[offset] = 1 - 0.5 * h / h_r[IDX(i, j, NrTotal, NzTotal)];
				csrValA[offset + 1] = 1.0;
				csrValA[offset + 2] = h * h * h_s[IDX(i, j, NrTotal, NzTotal)] - 4.0;
				csrValA[offset + 3] = 1.0;
				csrValA[offset + 4] = 1.0 + 0.5 * h / h_r[IDX(i, j, NrTotal, NzTotal)];
				// Columns.
				csrColIndA[offset] = IDX(i - 1, j, NrTotal, NzTotal);
				csrColIndA[offset + 1] = IDX(i, j - 1, NrTotal, NzTotal);
				csrColIndA[offset + 2] = IDX(i, j, NrTotal, NzTotal);
				csrColIndA[offset + 3] = IDX(i, j + 1, NrTotal, NzTotal);
				csrColIndA[offset + 4] = IDX(i + 1, j, NrTotal, NzTotal);
				// Increase offset.
				offset += 5;
			}
		}
	}

	// Right-bottom corner.
	csrRowPtrA[IDX(NrInterior + 1, 0, NrTotal, NzTotal)] = offset;
	csrValA[offset] = -0.375;
	csrValA[offset + 1] = -0.25;
	csrValA[offset + 2] = 1.0;
	csrValA[offset + 3] = -0.375;
	csrColIndA[offset] = IDX(NrInterior, 0, NrTotal, NzTotal);
	csrColIndA[offset + 1] = IDX(NrInterior, 1, NrTotal, NzTotal);
	csrColIndA[offset + 2] = IDX(NrInterior + 1, 0, NrTotal, NzTotal);
	csrColIndA[offset + 3] = IDX(NrInterior + 1, 1, NrTotal, NzTotal);
	offset += 4;

	// Right boundary.
	for (int j = 1; j < NzInterior + 1; j++) {
		// 3rd-order ROBIN.
		// Row begins at offset.
		csrRowPtrA[IDX(NrInterior + 1, j, NrTotal, NzTotal)] = offset;
		// Values.
		csrValA[offset] = (1.0 / 6.0) * h / h_r[IDX(NrInterior, j, NrTotal, NzTotal)];
		csrValA[offset + 1] = -h / h_r[IDX(NrInterior, j, NrTotal, NzTotal)];
		csrValA[offset + 2] = 0.5 * h / h_r[IDX(NrInterior, j, NrTotal, NzTotal)];
		csrValA[offset + 3] = (1.0 / 3.0) * h / h_r[IDX(NrInterior, j, NrTotal, NzTotal)] + h * h / (h_rr[IDX(NrInterior, j, NrTotal, NzTotal)] * h_rr[IDX(NrInterior, j, NrTotal, NzTotal)]);
		// Columns.
		csrColIndA[offset] = IDX(NrInterior - 2, j, NrTotal, NzTotal);
		csrColIndA[offset + 1] = IDX(NrInterior - 1, j, NrTotal, NzTotal);
		csrColIndA[offset + 2] = IDX(NrInterior, j, NrTotal, NzTotal);
		csrColIndA[offset + 3] = IDX(NrInterior + 1, j, NrTotal, NzTotal);
		// Increase offset.
		offset += 4;
		// Also fill F source.
		h_f[IDX(NrInterior + 1, j, NrTotal, NzTotal)] = h * h * uInf / (h_rr[IDX(NrInterior, j, NrTotal, NzTotal)] * h_rr[IDX(NrInterior, j, NrTotal, NzTotal)]);
	}

	// Left top corner.
	csrRowPtrA[IDX(NrInterior + 1, NzInterior + 1, NrTotal, NzTotal)] = offset;
	csrValA[offset] = -0.25;
	csrValA[offset + 1] = -0.375;
	csrValA[offset + 2] = -0.375;
	csrValA[offset + 3] = 1.0;
	csrColIndA[offset] = IDX(NrInterior, NzInterior, NrTotal, NzTotal);
	csrColIndA[offset + 1] = IDX(NrInterior, NzInterior + 1, NrTotal, NzTotal);
	csrColIndA[offset + 2] = IDX(NrInterior + 1, NzInterior, NrTotal, NzTotal);
	csrColIndA[offset + 3] = IDX(NrInterior + 1, NzInterior + 1, NrTotal, NzTotal);
	offset += 4;

	// Finally, fill last element of row_offsets.
	csrRowPtrA[IDX(NrInterior + 1, NzInterior + 1, NrTotal, NzTotal) + 1] = offset;

	// Check if we filled properly.
	assert(offset == nnzA);

	printf("CSR-GENERATOR: Filled CSR matrix with %d nonzero elements.\n", nnzA);
}
