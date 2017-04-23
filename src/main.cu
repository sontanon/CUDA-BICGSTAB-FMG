#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cublas_v2.h"
#include "cusparse.h"
#include "cusolverSp.h"

#include "cudaErrorCheck.h"
#include "tools.h"
#include "csrGenerator.h"

#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

int main(int argc, char *argv[])
{
	// Start by setting grids for both resolutions.
	const int NTotal0 = 65;
	const int NTotal1 = 127;
	const int DIM0 = NTotal0 * NTotal0;
	const int DIM1 = NTotal1 * NTotal1;

	// Interior points.
	const int NrInterior0 = NTotal0 - 2;
	const int NzInterior0 = NTotal0 - 2;
	const int NrInterior1 = NTotal1 - 2;
	const int NzInterior1 = NTotal1 - 2;

	// Step sizes.
	const double h0 = 10.0/((double)NTotal0 - 1.0);
	const double h1 = 0.5 * h0;

	// Value at infinity.
	const double uInf = 1.0;

	// Print parameters.
	printf("MAIN: Coarse grid parameters are:\n");
	printf("      NTotal0 = %d, h0 = %3.6E.\n", NTotal0, h0);
	printf("MAIN: Fine grid parameters are:\n");
	printf("      NTotal1 = %d, h1 = %3.6E.\n", NTotal1, h1);

	// Allocate host and device memory at both resolutions.
	double *h_r0 	= (double *)malloc(sizeof(double) * DIM0);
	double *h_r1 	= (double *)malloc(sizeof(double) * DIM1);
	double *h_z0 	= (double *)malloc(sizeof(double) * DIM0);
	double *h_z1 	= (double *)malloc(sizeof(double) * DIM1);
	double *h_rr0 	= (double *)malloc(sizeof(double) * DIM0);
	double *h_rr1 	= (double *)malloc(sizeof(double) * DIM1);
	double *h_s0 	= (double *)malloc(sizeof(double) * DIM0);
	double *h_s1 	= (double *)malloc(sizeof(double) * DIM1);
	double *h_u0 	= (double *)malloc(sizeof(double) * DIM0);
	double *h_u1 	= (double *)malloc(sizeof(double) * DIM1);
	double *h_f0 	= (double *)malloc(sizeof(double) * DIM0);
	double *h_f1 	= (double *)malloc(sizeof(double) * DIM1);
	double *h_res0 	= (double *)malloc(sizeof(double) * DIM0);
	double *h_res1 	= (double *)malloc(sizeof(double) * DIM1);

	// Assert that allocation was correct.
	assert(h_r0 	!= NULL);
	assert(h_r1 	!= NULL);
	assert(h_z0 	!= NULL);
	assert(h_z1 	!= NULL);
	assert(h_rr0 	!= NULL);
	assert(h_rr1 	!= NULL);
	assert(h_s0 	!= NULL);
	assert(h_s1 	!= NULL);
	assert(h_u0 	!= NULL);
	assert(h_u1 	!= NULL);
	assert(h_res0 	!= NULL);
	assert(h_res1 	!= NULL);
	assert(h_f0 	!= NULL);
	assert(h_f1 	!= NULL);
	printf("MAIN: Allocated host memory.\n");

	// Brill parameters.
	const double a0 = 1.0;
	const double rs02 = 1.0;
	const double zs02 = 1.0;

	// Auxiliary variables for fill-in.
	double temp_r, temp_z, temp_rr, temp_q, temp_s;

	// Main Filling can be optimized by using the major loop.
	for (int i = 0; i < NTotal0; i++) {
		for (int j = 0; j < NTotal0; j++) {
			// Write fine grid.
			temp_r = ((double)i - 0.5) * h0;
			temp_z = ((double)j - 0.5) * h0;
			temp_rr = sqrt(temp_r * temp_r + temp_z * temp_z);
			temp_q = a0 * exp(-(temp_r * temp_r/rs02 + temp_z * temp_z/zs02));
			temp_s = (0.5 + temp_r * temp_r * (-2.5/rs02 - 0.5/zs02
						+ temp_r * temp_r/(rs02 * rs02)
						+ temp_z * temp_z/(zs02 * zs02))) * temp_q;

			h_r0[IDX(i,j,NTotal0,NTotal0)] = temp_r;
			h_z0[IDX(i,j,NTotal0,NTotal0)] = temp_z;
			h_rr0[IDX(i,j,NTotal0,NTotal0)] = temp_rr;
			h_s0[IDX(i,j,NTotal0,NTotal0)] = temp_s;
			h_u0[IDX(i,j,NTotal0,NTotal0)] = 1.0;
			// Could do with memset.
			h_res0[IDX(i,j,NTotal0,NTotal0)] = 0.0;
			h_f0[IDX(i,j,NTotal0,NTotal0)] = 0.0;
		}
	}
	for (int i = 0; i < NTotal1; i++) {
		for (int j = 0; j < NTotal1; j++) {
			// Write fine grid.
			temp_r = ((double)i - 0.5) * h1;
			temp_z = ((double)j - 0.5) * h1;
			temp_rr = sqrt(temp_r * temp_r + temp_z * temp_z);
			temp_q = a0 * exp(-(temp_r * temp_r/rs02 + temp_z * temp_z/zs02));
			temp_s = (0.5 + temp_r * temp_r * (-2.5/rs02 - 0.5/zs02
						+ temp_r * temp_r/(rs02 * rs02)
						+ temp_z * temp_z/(zs02 * zs02))) * temp_q;

			h_r1[IDX(i,j,NTotal1,NTotal1)] = temp_r;
			h_z1[IDX(i,j,NTotal1,NTotal1)] = temp_z;
			h_rr1[IDX(i,j,NTotal1,NTotal1)] = temp_rr;
			h_s1[IDX(i,j,NTotal1,NTotal1)] = temp_s;
			h_u1[IDX(i,j,NTotal1,NTotal1)] = 1.0;
			// Could do with memset.
			h_res1[IDX(i,j,NTotal1,NTotal1)] = 0.0;
			h_f1[IDX(i,j,NTotal1,NTotal1)] = 0.0;
		}
	}

	printf("MAIN: Filled grids on host.\n");

	// Device memory.
	double *d_s0 = NULL;
	double *d_s1 = NULL;
	double *d_u0 = NULL;
	double *d_u1 = NULL;

	size_t pitch0, pitch1;
	cudaSafeCall(cudaMallocPitch((void **)&d_s0, &pitch0, NTotal0 * sizeof(double), NTotal0));
	cudaSafeCall(cudaMallocPitch((void **)&d_s1, &pitch1, NTotal1 * sizeof(double), NTotal1));
	cudaSafeCall(cudaMallocPitch((void **)&d_u0, &pitch0, NTotal0 * sizeof(double), NTotal0));
	cudaSafeCall(cudaMallocPitch((void **)&d_u1, &pitch1, NTotal1 * sizeof(double), NTotal1));
	printf("MAIN: Allocated device memory: pitch0 = %zd, pitch1 = %zd elements.\n", 
			pitch0/sizeof(double), pitch1/sizeof(double));

	cudaSafeCall(cudaMemcpy2D(d_s0, pitch0, h_s0, NTotal0 * sizeof(double), 
				NTotal0 * sizeof(double), NTotal0, cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy2D(d_s1, pitch1, h_s1, NTotal1 * sizeof(double), 
				NTotal1 * sizeof(double), NTotal0, cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy2D(d_u0, pitch0, h_u0, NTotal0 * sizeof(double), 
				NTotal0 * sizeof(double), NTotal0, cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy2D(d_u1, pitch1, h_u1, NTotal1 * sizeof(double), 
				NTotal1 * sizeof(double), NTotal0, cudaMemcpyHostToDevice));
	printf("MAIN: Copied to device memory.\n");

	// Allocate memory for CSR matrix.
	int nnzA0 = nnz_calculator(NrInterior0, NzInterior0);
	int *h_csrRowPtrA0 = (int *)	malloc(sizeof(int) * (DIM0 + 1));
	int *h_csrColIndA0 = (int *)	malloc(sizeof(int) * nnzA0);
	double *h_csrValA0 = (double *)	malloc(sizeof(double) * nnzA0);
	assert(h_csrRowPtrA0 != NULL);
	assert(h_csrColIndA0 != NULL);
	assert(h_csrValA0    != NULL);

	csrGenerator(h_csrRowPtrA0, h_csrColIndA0, h_csrValA0, NrInterior0, NzInterior0, 
			h_r0, h_z0, h_rr0, h_s0, h_f0, h0, uInf);
	printf("MAIN: Filled CSR matrix at coarse level.\n");

	// Device CSR arrays.
	int *d_csrRowPtrA0 = NULL;
	int *d_csrColIndA0 = NULL;
	double *d_csrValA0 = NULL;
	cudaSafeCall(cudaMalloc((void **)&d_csrRowPtrA0, sizeof(int) * (DIM0 + 1)));
	cudaSafeCall(cudaMalloc((void **)&d_csrColIndA0, sizeof(int) * nnzA0));
	cudaSafeCall(cudaMalloc((void **)&d_csrValA0, sizeof(double) * nnzA0));
	cudaSafeCall(cudaMemcpy(d_csrRowPtrA0, h_csrRowPtrA0, sizeof(int) * (DIM0 + 1), 
				cudaMemcpyHostToDevice));
       	cudaSafeCall(cudaMemcpy(d_csrColIndA0, h_csrColIndA0, sizeof(int) * nnzA0, 
				cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(d_csrValA0, h_csrValA0, sizeof(double) * nnzA0, 
				cudaMemcpyHostToDevice));
	printf("MAIN: Transfered CSR matrix to device.\n");
	
	// Declare handles.
	cublasHandle_t cublasHandle = 0;
	cusolverSpHandle_t cusolverHandle = 0;
	cusparseHandle_t cusparseHandle = 0;
	cudaStream_t stream = 0;
	cusparseMatDescr_t descr_A = 0;

	// Initialize handles and set stream.
	cublasSafeCall(cublasCreate(&cublasHandle));
	cusolverSafeCall(cusolverSpCreate(&cusolverHandle));
	cusparseSafeCall(cusparseCreate(&cusparseHandle));
	cudaSafeCall(cudaStreamCreate(&stream));
	cusolverSafeCall(cusolverSpSetStream(cusolverHandle, stream));
	cusparseSafeCall(cusparseCreateMatDescr(&descr_A));
	const cusparseOperation_t trans_A = CUSPARSE_OPERATION_NON_TRANSPOSE;
	cusparseSafeCall(cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL));
	cusparseSafeCall(cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO));
	printf("MAIN: Handles and matrix description set.\n");

	// ILU Factorization.
	double *d_csrValA0_M = NULL;
	cudaSafeCall(cudaMalloc((void **)&d_csrValA0_M, sizeof(double) * nnzA0));
	cudaSafeCall(cudaMemcpy(d_csrValA0_M, d_csrValA0, sizeof(double) * nnzA0, 
				cudaMemcpyDeviceToDevice));

	printf("MAIN: ILU: Starting...\n");
	// Step 0: Set descriptions, analysis, etc.
	cusparseMatDescr_t descr_M = 0;
	cusparseMatDescr_t descr_L = 0;
	cusparseMatDescr_t descr_U = 0;

	csrilu02Info_t info_M = 0;
	csrsv2Info_t info_L = 0;
	csrsv2Info_t info_U = 0;

	int pBufferSize_M;
	int pBufferSize_L;
	int pBufferSize_U;
	int pBufferSize;
	void *pBuffer = NULL;

	int structural_zero;
	int numerical_zero;

	cusparseStatus_t status;

	const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
	const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
	const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
	const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
	const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
	printf("MAIN: ILU: Allocated descriptions and solve policies.\n");

	// Step 1: Create a descriptor for each matrix:
	// - L is base 0.
	// - L is lower triangular.
	// - L has unit diagonal.
	// - U is base 0.
	// - U is upper triangular.
	// - U has non unit diagonal.

	cusparseSafeCall(cusparseCreateMatDescr(&descr_M));
	cusparseSafeCall(cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO));
	cusparseSafeCall(cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL));

	cusparseSafeCall(cusparseCreateMatDescr(&descr_L));
	cusparseSafeCall(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO));
	cusparseSafeCall(cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
	cusparseSafeCall(cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER));
	cusparseSafeCall(cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT));

	cusparseSafeCall(cusparseCreateMatDescr(&descr_U));
	cusparseSafeCall(cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO));
	cusparseSafeCall(cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL));
	cusparseSafeCall(cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER));
	cusparseSafeCall(cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT));

	printf("MAIN: ILU: Set descriptions for matrices.\n");

	// Step 2: Create an empty info structure.
	// We need one info for csrilu02 and two for csrsv2.
	cusparseSafeCall(cusparseCreateCsrilu02Info(&info_M));
	cusparseSafeCall(cusparseCreateCsrsv2Info(&info_L));
	cusparseSafeCall(cusparseCreateCsrsv2Info(&info_U));

	printf("MAIN: ILU: Created info structures.\n");
	
	// Step 3: Query how much memory is used in csrilu02 and csrsv2 and allocate
	// the buffer.
	cusparseSafeCall(cusparseDcsrilu02_bufferSize(cusparseHandle, DIM0, nnzA0,
		descr_A, d_csrValA0_M, d_csrRowPtrA0, d_csrColIndA0,
		info_M, &pBufferSize_M));
	cusparseSafeCall(cusparseDcsrsv2_bufferSize(cusparseHandle, trans_L, DIM0, nnzA0,
		descr_L, d_csrValA0_M, d_csrRowPtrA0, d_csrColIndA0,
		info_L, &pBufferSize_L));
	cusparseSafeCall(cusparseDcsrsv2_bufferSize(cusparseHandle, trans_U, DIM0, nnzA0,
		descr_U, d_csrValA0_M, d_csrRowPtrA0, d_csrColIndA0,
		info_U, &pBufferSize_U));

	pBufferSize = MAX(pBufferSize_M, MAX(pBufferSize_L, pBufferSize_U));

	// pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
	cudaSafeCall(cudaMalloc((void **)&pBuffer, pBufferSize));

	printf("MAIN: ILU: Set pBuffer size.\n");

	// Step 4: Perform analysis of incomplete Cholesky on A.
	//         Perform analysis of triangular solve on L.
	//         Perform analysis of triangular solve on U.
	// The lower (upper) triangular part of M has the same sparsity pattern as
	// L (U), so we can do analysis of csrilu02 and csrsv2 simultaneously.
	cusparseSafeCall(cusparseDcsrilu02_analysis(cusparseHandle, DIM0, nnzA0,
		descr_M, d_csrValA0_M, d_csrRowPtrA0, d_csrColIndA0,
		info_M, policy_M, pBuffer));

        status = cusparseXcsrilu02_zeroPivot(cusparseHandle, info_M, &structural_zero);
	cusparseSafeCall(status);
	if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
		printf("A(%d, %d) is missing.\n", structural_zero, structural_zero);
	}

	cusparseSafeCall(cusparseDcsrsv2_analysis(cusparseHandle, trans_L, DIM0, nnzA0,
		descr_L, d_csrValA0_M, d_csrRowPtrA0, d_csrColIndA0,
		info_L, policy_L, pBuffer));

	cusparseSafeCall(cusparseDcsrsv2_analysis(cusparseHandle, trans_U, DIM0, nnzA0,
		descr_U, d_csrValA0_M, d_csrRowPtrA0, d_csrColIndA0,
		info_U, policy_U, pBuffer));

	printf("MAIN: ILU: Performed analysis.\n");

	// Step 5: A = L * U.
	cusparseSafeCall(cusparseDcsrilu02(cusparseHandle, DIM0, nnzA0, descr_A, d_csrValA0_M, d_csrRowPtrA0, d_csrColIndA0, info_M, policy_M, pBuffer));

	status = cusparseXcsrilu02_zeroPivot(cusparseHandle, info_M, &numerical_zero);
	cusparseSafeCall(status);
	if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
		printf("U(%d, %d) is zero,\n", numerical_zero, numerical_zero);
	}
	printf("MAIN: ILU: Completed ILU factorization.\n");

	// Start solver at lowest level.

	// ILU factorization.

	// BICGSTAB solve.

	// Interpolate to higher resolution.

	// Run Gauss-Seidel until convergence.

	// Clean up.
	// ILU resources.
	if (pBuffer) {
		cudaSafeCall(cudaFree(pBuffer));
		pBuffer = NULL;
	}
	if (info_M) {
		cusparseSafeCall(cusparseDestroyCsrilu02Info(info_M));
		info_M = 0;
	}
	if (info_L) {
		cusparseSafeCall(cusparseDestroyCsrsv2Info(info_L));
		info_L = 0;
	}
	if (info_U) {
		cusparseSafeCall(cusparseDestroyCsrsv2Info(info_U));
		info_U = 0;
	}

	// CSR matrix.
	if (h_csrRowPtrA0) {
		free(h_csrRowPtrA0);
		h_csrRowPtrA0 = NULL;
	}
	if (h_csrColIndA0) {
		free(h_csrColIndA0);
		h_csrColIndA0 = NULL;
	}
	if (h_csrValA0) {
		free(h_csrValA0);
		h_csrValA0 = NULL;
	}
	if (d_csrRowPtrA0) {
		cudaSafeCall(cudaFree(d_csrRowPtrA0));
		d_csrRowPtrA0 = NULL;
	}
	if (d_csrColIndA0) {
		cudaSafeCall(cudaFree(d_csrColIndA0));
		d_csrColIndA0 = NULL;
	}
	if (d_csrValA0) {
		cudaSafeCall(cudaFree(d_csrValA0));
		d_csrValA0 = NULL;
	}
	if (d_csrValA0_M) {
		cudaSafeCall(cudaFree(d_csrValA0_M));
		d_csrValA0_M = NULL;
	}
	// Device memory.
	if (d_s0) {
		cudaSafeCall(cudaFree(d_s0));
		d_s0 = NULL;
	}
	if (d_s1) {
		cudaSafeCall(cudaFree(d_s1));
		d_s1 = NULL;
	}
	if (d_u0) {
		cudaSafeCall(cudaFree(d_u0));
		d_u0 = NULL;
	}
	if (d_u1) {
		cudaSafeCall(cudaFree(d_u1));
		d_u1 = NULL;
	}
	printf("MAIN: Cleared device memory.\n");

	// Matrix descriptions.
	if (descr_A) {
		cusparseSafeCall(cusparseDestroyMatDescr(descr_A));
		descr_A = 0;
	}
	if (descr_L) {
		cusparseSafeCall(cusparseDestroyMatDescr(descr_L));
		descr_L = 0;
	}
	if (descr_U) {
		cusparseSafeCall(cusparseDestroyMatDescr(descr_U));
		descr_U = 0;
	}
	if (descr_M) {
		cusparseSafeCall(cusparseDestroyMatDescr(descr_M));
		descr_M = 0;
	}
	printf("MAIN: Cleared matrix descriptions.\n");

	// Handles and misc.
	if (cusolverHandle) {
		cusolverSafeCall(cusolverSpDestroy(cusolverHandle));
		cusolverHandle = 0;
	}
	if (cusparseHandle) {
		cusparseSafeCall(cusparseDestroy(cusparseHandle));
		cusparseHandle = 0;
	}
	if (cublasHandle) {
		cublasSafeCall(cublasDestroy(cublasHandle));
		cublasHandle = 0;
	}
	if (stream) {
		cudaSafeCall(cudaStreamDestroy(stream));
		stream = 0;
	}
	printf("MAIN: Cleared handles and stream.\n");

	// Host memory.
	if (h_r0) {
		free(h_r0);
		h_r0 = NULL;
	}
	if (h_r1) {
		free(h_r1);
		h_r1 = NULL;
	}
	if (h_z0) {
		free(h_z0);
		h_z0 = NULL;
	}
	if (h_z1) {
		free(h_z1);
		h_z1 = NULL;
	}
	if (h_s0) {
		free(h_s0);
		h_s0 = NULL;
	}
	if (h_rr0) {
		free(h_rr0);
		h_rr0 = NULL;
	}
	if (h_rr1) {
		free(h_rr1);
		h_rr1 = NULL;
	}
	if (h_s1) {
		free(h_s1);
		h_s1 = NULL;
	}
	if (h_u0) {
		free(h_u0);
		h_u0 = NULL;
	}
	if (h_u1) {
		free(h_u1);
		h_u1 = NULL;
	}
	if (h_res0) {
		free(h_res0);
		h_res0 = NULL;
	}
	if (h_res1) {
		free(h_res1);
		h_res1 = NULL;
	}
	if (h_f0) {
		free(h_f0);
		h_f0 = NULL;
	}
	if (h_f1) {
		free(h_f1);
		h_f1 = NULL;
	}
	printf("MAIN: Cleared host memory.\n");
	printf("MAIN: All done, have a nice day!\n");

	return 0;
}
