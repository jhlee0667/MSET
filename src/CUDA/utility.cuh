#ifndef UTILITY_CUH
#define UTILITY_CUH
#include <iostream>
#include <array>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda.h>
#include <cufft.h>
#include <memory>
#include <vector>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <new> 


__global__ void CuPointwiseAdd(cufftComplex *a, cufftComplex *b, size_t size);
__global__ void CuPointwiseAdd(cufftComplex a, cufftComplex *b, size_t size);
__global__ void CuPointwiseAdd(float *a, float *b, size_t size);
__global__ void CuPointwiseAdd(float a, float *b, size_t size);
__global__ void CuPointwiseAdd(cufftDoubleComplex *a, cufftDoubleComplex *b, size_t size);
__global__ void CuPointwiseAdd(cufftDoubleComplex a, cufftDoubleComplex *b, size_t size);
__global__ void CuPointwiseAdd(double *a, double *b, size_t size);
__global__ void CuPointwiseAdd(double a, double *b, size_t size);

__global__ void CuPointwiseSub(cufftComplex *a, cufftComplex *b, size_t size);
__global__ void CuPointwiseSub(cufftComplex a, cufftComplex *b, size_t size);
__global__ void CuPointwiseSub(cufftComplex *a, cufftComplex b, size_t size);
__global__ void CuPointwiseSub(cufftDoubleComplex *a, cufftDoubleComplex *b, size_t size);
__global__ void CuPointwiseSub(cufftDoubleComplex a, cufftDoubleComplex *b, size_t size);
__global__ void CuPointwiseSub(cufftDoubleComplex *a, cufftDoubleComplex b, size_t size);
__global__ void CuPointwiseSub(float *a, float *b, size_t size);
__global__ void CuPointwiseSub(float a, float *b, size_t size);
__global__ void CuPointwiseSub(float *a, float b, size_t size);
__global__ void CuPointwiseSub(double *a, double *b, size_t size);
__global__ void CuPointwiseSub(double a, double *b, size_t size);
__global__ void CuPointwiseSub(double *a, double b, size_t size);

__global__ void CuPointwiseMul(cufftComplex *a, cufftComplex *b, size_t size);
__global__ void CuPointwiseMul(cufftComplex a, cufftComplex *b, size_t size);
__global__ void CuPointwiseMul(float a, float *b, size_t size);
__global__ void CuPointwiseMul(cufftComplex *a, cufftComplex *b, size_t size, size_t start_p);
__global__ void CuPointwiseMul(cufftDoubleComplex *a, cufftDoubleComplex *b, size_t size);
__global__ void CuPointwiseMul(cufftDoubleComplex a, cufftDoubleComplex *b, size_t size);
__global__ void CuPointwiseMul(double a, double *b, size_t size);
__global__ void CuPointwiseMul(cufftDoubleComplex *a, cufftDoubleComplex *b, size_t size, size_t start_p);

__global__ void CuPointwiseDiv(cufftComplex *a, cufftComplex *b, size_t size);
__global__ void CuPointwiseDiv(float *a, float *b, size_t size);
__global__ void CuPointwiseDiv(cufftDoubleComplex *a, cufftDoubleComplex *b, size_t size);
__global__ void CuPointwiseDiv(double *a, double *b, size_t size);

__global__ void CuPointwiseExp(cufftComplex *a, size_t size);
__global__ void CuPointwiseExp(cufftDoubleComplex *a, size_t size);

__global__ void CuPointwiseAbsSquare(float *a, size_t size);
__global__ void CuPointwiseAbsSquare(double *a, size_t size);
__global__ void CuPointwiseAbsSquare(cufftComplex *a, float *b, size_t size);
__global__ void CuPointwiseAbsSquare(cufftDoubleComplex *a, double *b, size_t size);

__global__ void CuPointwiseAbs(cufftComplex *a, size_t size);
__global__ void CuPointwiseAbs(float *a, size_t size);
__global__ void CuPointwiseAbs(cufftDoubleComplex *a, size_t size);
__global__ void CuPointwiseAbs(double *a, size_t size);

__global__ void CuPointwisePow(float *a, float b, size_t size);
__global__ void CuPointwisePow(double *a, double b, size_t size);

__global__ void CuPointwiseConj(cufftComplex *a, size_t size);
__global__ void CuPointwiseConj(cufftDoubleComplex *a, size_t size);

__global__ void CuPositivity(float *a, size_t size);
__global__ void CuPositivity(double *a, size_t size);

__global__ void datatransfer_F2C(float2 *f2, float *f, size_t size);
__global__ void datatransfer_F2C(double2 *f2, double *f, size_t size);
__global__ void datatransfer_C2F(float *f, float2 *f2, size_t size);
__global__ void datatransfer_C2F(double *f, double2 *f2, size_t size);

__global__ void circular_shift2D_dev(float2 *f1, float2 *f2, int dims1, int dims2, int N1_shift, int N2_shift);
__global__ void circular_shift2D_dev(float *f1, float *f2, int dims1, int dims2, int N1_shift, int N2_shift);
__global__ void circular_shift2D_dev(double2 *f1, double2 *f2, int dims1, int dims2, int N1_shift, int N2_shift);
__global__ void circular_shift2D_dev(double *f1, double *f2, int dims1, int dims2, int N1_shift, int N2_shift);

void circular_shift2D(float2 *f1, float2 *f2, int dims1, int dims2, int N1_shift, int N2_shift);
void circular_shift2D(float *f1, float *f2, int dims1, int dims2, int N1_shift, int N2_shift);
void circular_shift2D(double2 *f1, double2 *f2, int dims1, int dims2, int N1_shift, int N2_shift);
void circular_shift2D(double *f1, double *f2, int dims1, int dims2, int N1_shift, int N2_shift);

__global__ void fourier_shift_factor_dev(cufftComplex *f, int dims1, int dims2, float N1_shift, float N2_shift);
__global__ void fourier_shift_factor_dev(cufftDoubleComplex *f, int dims1, int dims2, float N1_shift, float N2_shift);

__global__ void gradient_2D_dev(cufftComplex *f1, cufftComplex *f2, int dims1, int dims2, int option);
__global__ void gradient_2D_dev(cufftDoubleComplex *f1, cufftDoubleComplex *f2, int dims1, int dims2, int option);

float Cusum_real(cufftComplex *f, size_t size);

void print_array_dev(float *a, size_t dims1, size_t dims2);
void print_array_dev(cufftComplex *a, size_t dims1, size_t dims2);
void print_array_dev(double *a, size_t dims1, size_t dims2);
void print_array_dev(cufftDoubleComplex *a, size_t dims1, size_t dims2);
void print_array(float *a, size_t dims1, size_t dims2);
void print_array(float2 *a, size_t dims1, size_t dims2);
void print_array(double *a, size_t dims1, size_t dims2);
void print_array(double2 *a, size_t dims1, size_t dims2);


#endif