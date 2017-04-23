/********************************************************************
 ***                      CUDA ERROR CHECKING                     ***
 ********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cublas_v2.h"
#include "cusparse.h"
#include "cusolverSp.h"

#define CUDA_ERROR_CHECK

#define cudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define cudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
#define cublasSafeCall( err ) __cublasSafeCall( err , __FILE__ , __LINE__ )
#define cusparseSafeCall( err ) __cusparseSafeCall( err , __FILE__ , __LINE__ )
#define cusolverSafeCall( err ) __cusolverSafeCall( err , __FILE__ , __LINE__ )

inline void __cudaSafeCall(cudaError_t err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaSafeCall() failed at %s: %i: %s.\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif
	return;
}

inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	cudaError_t err = cudaPeekAtLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() failed at %s: %i: %s.\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}

	// Sync devices and check again.
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() wiith sync failed at %s: %i: %s.\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif
	return;
}

inline void __cublasSafeCall(cublasStatus_t cudaStatus, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	if (cudaStatus != CUBLAS_STATUS_SUCCESS) {
		char *errString = (char *)malloc(60);
		assert(errString != NULL);

		switch (cudaStatus) {
			case CUBLAS_STATUS_NOT_INITIALIZED:
				strcpy(errString, "CUBLAS_STATUS_NOT_INITIALIZED");
				break;
			case CUBLAS_STATUS_ALLOC_FAILED:
				strcpy(errString, "CUBLAS_STATUS_ALLOC_FAILED");
				break;
			case CUBLAS_STATUS_INVALID_VALUE:
				strcpy(errString, "CUBLAS_STATUS_INVALID_VALUE");
				break;
			case CUBLAS_STATUS_ARCH_MISMATCH:
				strcpy(errString, "CUBLAS_STATUS_ARCH_MISMATCH");
				break;
			case CUBLAS_STATUS_MAPPING_ERROR:
				strcpy(errString, "CUBLAS_STATUS_MAPPING_ERROR");
				break;
			case CUBLAS_STATUS_EXECUTION_FAILED:
				strcpy(errString, "CUBLAS_STATUS_EXECUTION_FAILED");
				break;
			case CUBLAS_STATUS_INTERNAL_ERROR:
				strcpy(errString, "CUBLAS_STATUS_INTERNAL_ERROR");
				break;
			case CUBLAS_STATUS_NOT_SUPPORTED:
				strcpy(errString, "CUBLAS_STATUS_NOT_SUPPORTED");
				break;
			case CUBLAS_STATUS_LICENSE_ERROR:
				strcpy(errString, "CUBLAS_STATUS_LICENSE_ERROR");
				break;
			default:
				strcpy(errString, "UNRECOGNIZED_ERROR");
		}
		fprintf(stderr, "cublasSafeCall() failed at %s: %i: %s.\n", file, line, errString);
		
		if (errString) {
			free(errString);
			errString = NULL;
		}

		exit(-1);
	}
#endif
	return;
}

inline void __cusparseSafeCall(cusparseStatus_t cudaStatus, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	if (cudaStatus != CUSPARSE_STATUS_SUCCESS) {
		char *errString = (char *)malloc(60);
		assert(errString != NULL);

		switch (cudaStatus) {
			case CUSPARSE_STATUS_NOT_INITIALIZED:
				strcpy(errString, "CUSPARSE_STATUS_NOT_INITIALIZED");
				break;
			case CUSPARSE_STATUS_ALLOC_FAILED:
				strcpy(errString, "CUSPARSE_STATUS_ALLOC_FAILED");
				break;
			case CUSPARSE_STATUS_INVALID_VALUE:
				strcpy(errString, "CUSPARSE_STATUS_INVALID_VALUE");
				break;
			case CUSPARSE_STATUS_ARCH_MISMATCH:
				strcpy(errString, "CUSPARSE_STATUS_ARCH_MISMATCH");
				break;
			case CUSPARSE_STATUS_MAPPING_ERROR:
				strcpy(errString, "CUSPARSE_STATUS_MAPPING_ERROR");
				break;
			case CUSPARSE_STATUS_EXECUTION_FAILED:
				strcpy(errString, "CUSPARSE_STATUS_EXECUTION_FAILED");
				break;
			case CUSPARSE_STATUS_INTERNAL_ERROR:
				strcpy(errString, "CUSPARSE_STATUS_INTERNAL_ERROR");
				break;
			case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
				strcpy(errString, "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED");
				break;
			default:
				strcpy(errString, "UNRECOGNIZED_ERROR");
		}
		fprintf(stderr, "cusparseSafeCall() failed at %s: %i: %s.\n", file, line, errString);

		if (errString) {
			free(errString);
			errString = NULL;
		}

		exit(-1);
	}
#endif
	return;
}

inline void __cusolverSafeCall(cusolverStatus_t cudaStatus, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	if (cudaStatus != CUSOLVER_STATUS_SUCCESS) {
		char *errString = (char *)malloc(60);
		assert(NULL != errString);

		switch (cudaStatus) {
			case CUSOLVER_STATUS_NOT_INITIALIZED:
				strcpy(errString, "CUSOLVER_STATUS_NOT_INITIALIZED");
				break;
			case CUSOLVER_STATUS_ALLOC_FAILED:
				strcpy(errString, "CUSOLVER_STATUS_ALLOC_FAILED");
				break;
			case CUSOLVER_STATUS_INVALID_VALUE:
				strcpy(errString, "CUSOLVER_STATUS_INVALID_VALUE");
				break;
			case CUSOLVER_STATUS_ARCH_MISMATCH:
				strcpy(errString, "CUSOVLER_STATUS_ARCH_MISMATCH");
				break;
			case CUSOLVER_STATUS_EXECUTION_FAILED:
				strcpy(errString, "CUSOLVER_STATUS_EXECUTION_FAILED");
				break;
			case CUSOLVER_STATUS_INTERNAL_ERROR:
				strcpy(errString, "CUSOLVER_STATUS_INTERNAL_ERROR");
				break;
			case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
				strcpy(errString, "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED");
				break;
			default:
				strcpy(errString, "UNRECOGNIZED_ERROR");
		}
		fprintf(stderr, "cusolverSafeCall() failed at %s: %i: %s.\n", file, line, errString);
		
		if (errString) {
			free(errString);
			errString = NULL;
		}

		exit(-1);
	}
#endif
	return;
}
