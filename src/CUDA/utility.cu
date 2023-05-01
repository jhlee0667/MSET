#include "utility.cuh"

// Utility routine to perform complex pointwise calculations (add, subs, mult, conj ...)

__global__ void CuPointwiseAdd(cufftComplex *a, cufftComplex *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = cuCaddf(a[i], b[i]);
    }
    return;
}

__global__ void CuPointwiseAdd(cufftComplex a, cufftComplex *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = cuCaddf(a, b[i]);
    }
    return;
}

__global__ void CuPointwiseAdd(float a, float *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = a + b[i];
    }
    return;
}

__global__ void CuPointwiseAdd(float *a, float *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = a[i] + b[i];
    }
    return;
}

__global__ void CuPointwiseAdd(cufftDoubleComplex *a, cufftDoubleComplex *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = cuCadd(a[i], b[i]);
    }
    return;
}

__global__ void CuPointwiseAdd(cufftDoubleComplex a, cufftDoubleComplex *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = cuCadd(a, b[i]);
    }
    return;
}

__global__ void CuPointwiseAdd(double a, double *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = a + b[i];
    }
    return;
}

__global__ void CuPointwiseAdd(double *a, double *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = a[i] + b[i];
    }
    return;
}


__global__ void CuPointwiseSub(cufftComplex *a, cufftComplex *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = cuCsubf(a[i], b[i]);
    }
    return;
}

__global__ void CuPointwiseSub(cufftComplex a, cufftComplex *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = cuCsubf(a, b[i]);
    }
    return;
}

__global__ void CuPointwiseSub(cufftComplex *a, cufftComplex b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        a[i] = cuCsubf(a[i], b);
    }
    return;
}

__global__ void CuPointwiseSub(cufftDoubleComplex *a, cufftDoubleComplex *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = cuCsub(a[i], b[i]);
    }
    return;
}

__global__ void CuPointwiseSub(cufftDoubleComplex a, cufftDoubleComplex *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = cuCsub(a, b[i]);
    }
    return;
}

__global__ void CuPointwiseSub(cufftDoubleComplex *a, cufftDoubleComplex b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        a[i] = cuCsub(a[i], b);
    }
    return;
}


__global__ void CuPointwiseSub(float *a, float *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = a[i]- b[i];
    }
    return;
}

__global__ void CuPointwiseSub(float a, float *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = a - b[i];
    }
    return;
}

__global__ void CuPointwiseSub(float *a, float b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        a[i] = a[i] - b;
    }
    return;
}

__global__ void CuPointwiseSub(double *a, double *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = a[i] - b[i];
    }
    return;
}

__global__ void CuPointwiseSub(double a, double *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = a - b[i];
    }
    return;
}

__global__ void CuPointwiseSub(double *a, double b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        a[i] = a[i]- b;
    }
    return;
}


__global__ void CuPointwiseMul(cufftComplex *a, cufftComplex *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    cufftComplex c;
    for (int i = threadID; i < size; i += numThreads)
    {
        c = cuCmulf(a[i], b[i]);
        b[i] = make_cuFloatComplex(cuCrealf(c), cuCimagf(c));
    }
    return;
}

__global__ void CuPointwiseMul(cufftComplex *a, cufftComplex *b, size_t size, size_t start_p)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    cufftComplex c;
    for (int i = threadID; i < size; i += numThreads)
    {
        c = cuCmulf(a[i+start_p*size], b[i]);
        b[i] = make_cuFloatComplex(cuCrealf(c), cuCimagf(c));
    }
    return;
}

__global__ void CuPointwiseMul(cufftComplex a, cufftComplex *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    cufftComplex c;
    for (int i = threadID; i < size; i += numThreads)
    {
        c = cuCmulf(a, b[i]);
        b[i] = make_cuFloatComplex(cuCrealf(c), cuCimagf(c));
    }
    return;
}

__global__ void CuPointwiseMul(float a, float *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = a * b[i];
    }
    return;
}

__global__ void CuPointwiseMul(cufftDoubleComplex *a, cufftDoubleComplex *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = cuCmul(a[i], b[i]);
    }
    return;
}

__global__ void CuPointwiseMul(cufftDoubleComplex *a, cufftDoubleComplex *b, size_t size, size_t start_p)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = cuCmul(a[i+start_p*size], b[i]);
    }
    return;
}

__global__ void CuPointwiseMul(cufftDoubleComplex a, cufftDoubleComplex *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = cuCmul(a, b[i]);
    }
    return;
}

__global__ void CuPointwiseMul(double a, double *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = a * b[i];
    }
    return;
}


__global__ void CuPointwiseDiv(cufftComplex *a, cufftComplex *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = cuCdivf(a[i], b[i]);
    }
    return;
}

__global__ void CuPointwiseDiv(float *a, float *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = a[i]/b[i];
    }
    return;
}


__global__ void CuPointwiseDiv(cufftDoubleComplex *a, cufftDoubleComplex *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = cuCdiv(a[i], b[i]);
    }
    return;
}

__global__ void CuPointwiseDiv(double *a, double *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        b[i] = a[i]/b[i];
    }
    return;
}


__global__ void CuPointwiseExp(cufftComplex *a, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    cufftComplex res;
    float s, c, e;

    for (int i = threadID; i < size; i += numThreads)
    {
        e = expf(a[i].x);
        sincosf(a[i].y, &s, &c);
        res.x = c * e;
        res.y = s * e;
        a[i] = res;
    }
    return;
}

__global__ void CuPointwiseExp(cufftDoubleComplex *a, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    cufftDoubleComplex res;
    float s, c, e;

    for (int i = threadID; i < size; i += numThreads)
    {
        e = exp(a[i].x);
        sincos(a[i].y, &s, &c);
        res.x = c * e;
        res.y = s * e;
        a[i] = res;
    }
    return;
}

__global__ void CuPointwiseAbsSquare(float *a, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        a[i] = a[i]*a[i];
    }
}

__global__ void CuPointwiseAbsSquare(double *a, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        a[i] = a[i]*a[i];
    }
}


__global__ void CuPointwiseAbsSquare(cufftComplex *a, float *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp;
    for (int i = threadID; i < size; i += numThreads)
    {
        tmp = cuCabsf(a[i]); 
        b[i] = tmp*tmp;
    }
}

__global__ void CuPointwiseAbsSquare(cufftDoubleComplex *a, double *b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    double tmp;
    for (int i = threadID; i < size; i += numThreads)
    {
        tmp = cuCabs(a[i]); 
        b[i] = tmp*tmp;
    }
}


__global__ void CuPointwiseAbs(cufftComplex *a, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        a[i] = make_cuFloatComplex(cuCabsf(a[i]), 0.0f);
    }
}

__global__ void CuPointwiseAbs(cufftDoubleComplex *a, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        a[i] = make_cuDoubleComplex(cuCabs(a[i]), 0.0f);
    }
}


__global__ void CuPointwiseAbs(float *a, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        a[i] = fabsf(a[i]);
    }
}

__global__ void CuPointwiseAbs(double *a, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        a[i] = fabs(a[i]);
    }
}

__global__ void CuPointwisePow(float *a, float b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        a[i] = powf(a[i], b);
    }
}

__global__ void CuPointwisePow(double *a, double b, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        a[i] = pow(a[i], b);
    }
}


__global__ void CuPointwiseConj(cufftComplex *a, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        a[i] = cuConjf(a[i]);
    }
}

__global__ void CuPointwiseConj(cufftDoubleComplex *a, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        a[i] = cuConj(a[i]);
    }
}

__global__ void CuPositivity(float *a, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        if (a[i] < 0){
            a[i] = 0.0f;
        }   
    }
}

__global__ void CuPositivity(double *a, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        if (a[i] < 0){
            a[i] = 0.0;
        }   
    }
}

// ---------------------------------------------------
__global__ void datatransfer_F2C(float2 *f2, float *f, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        f2[i].x =  f[i];
        f2[i].y = 0;
    }
    return;
}

__global__ void datatransfer_F2C(double2 *f2, double *f, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        f2[i].x =  f[i];
        f2[i].y = 0;
    }
    return;
}

__global__ void datatransfer_C2F(float *f, float2 *f2, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        f[i] = f2[i].x;
    }
    return;
}

__global__ void datatransfer_C2F(double *f, double2 *f2, size_t size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        f[i] = f2[i].x;
    }
    return;
}

__global__ void circular_shift2D_dev(float2 *f1, float2 *f2, int dims1, int dims2, int N1_shift, int N2_shift)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int n1, n2; 
    for (int i = threadID; i < dims1*dims2; i += numThreads)
    {
        n1 = (i/dims2 + N1_shift)%dims1;
        n2 = (i%dims2 + N2_shift)%dims2; 
        if (n1 < 0){
            n1 = n1 + dims1;     
        }
        if (n2 < 0){
            n2 = n2 + dims2;
        }

        f2[n1*dims2 + n2].x = f1[i].x;
        f2[n1*dims2 + n2].y = f1[i].y;
    }
    return;

}

__global__ void circular_shift2D_dev(float *f1, float *f2, int dims1, int dims2, int N1_shift, int N2_shift)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int n1, n2;
    for (int i = threadID; i < dims1*dims2; i += numThreads)
    {
        n1 = (i/dims2 + N1_shift)%dims1;
        n2 = (i%dims2 + N2_shift)%dims2; 
        if (n1 < 0){
            n1 = n1 + dims1;     
        }
        if (n2 < 0){
            n2 = n2 + dims2;
        }  
    
        f2[n1*dims2 + n2] = f1[i];
    }
    return;

}

__global__ void circular_shift2D_dev(double2 *f1, double2 *f2, int dims1, int dims2, int N1_shift, int N2_shift)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int n1, n2;
    for (int i = threadID; i < dims1*dims2; i += numThreads)
    {
        n1 = (i/dims2 + N1_shift)%dims1;
        n2 = (i%dims2 + N2_shift)%dims2; 
        if (n1 < 0){
            n1 = n1 + dims1;     
        }
        if (n2 < 0){
            n2 = n2 + dims2;
        }

        f2[n1*dims2 + n2].x = f1[i].x;
        f2[n1*dims2 + n2].y = f1[i].y;
    }
    return;

}

__global__ void circular_shift2D_dev(double *f1, double *f2, int dims1, int dims2, int N1_shift, int N2_shift)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int n1, n2;
    for (int i = threadID; i < dims1*dims2; i += numThreads)
    {
        n1 = (i/dims2 + N1_shift)%dims1;
        n2 = (i%dims2 + N2_shift)%dims2; 
        if (n1 < 0){
            n1 = n1 + dims1;     
        }
        if (n2 < 0){
            n2 = n2 + dims2;
        }  
    
        f2[n1*dims2 + n2] = f1[i];
    }
    return;

}



void circular_shift2D(float2 *f1, float2 *f2, int dims1, int dims2, int N1_shift, int N2_shift)
{
    int n1, n2;
    for (int i = 0; i < dims1*dims2; ++i){
        n1 = (i/dims2 + N1_shift)%dims1;
        n2 = (i%dims2 + N2_shift)%dims2; 
        if (n1 < 0){
            n1 = n1 + dims1;     
        }
        if (n2 < 0){
            n2 = n2 + dims2;
        }

        f2[n1*dims2 + n2].x = f1[i].x;
        f2[n1*dims2 + n2].y = f1[i].y;
    }
    return;
}

void circular_shift2D(float *f1, float *f2, int dims1, int dims2, int N1_shift, int N2_shift)
{
    int n1, n2;
    for (int i = 0; i < dims1*dims2; ++i){
        n1 = (i/dims2 + N1_shift)%dims1;
        n2 = (i%dims2 + N2_shift)%dims2; 
        if (n1 < 0){
            n1 = n1 + dims1;     
        }
        if (n2 < 0){
            n2 = n2 + dims2;
        }

        f2[n1*dims2 + n2] = f1[i];
    }
    return;
}

void circular_shift2D(double2 *f1, double2 *f2, int dims1, int dims2, int N1_shift, int N2_shift)
{
    int n1, n2;
    for (int i = 0; i < dims1*dims2; ++i){
        n1 = (i/dims2 + N1_shift)%dims1;
        n2 = (i%dims2 + N2_shift)%dims2; 
        if (n1 < 0){
            n1 = n1 + dims1;     
        }
        if (n2 < 0){
            n2 = n2 + dims2;
        }

        f2[n1*dims2 + n2].x = f1[i].x;
        f2[n1*dims2 + n2].y = f1[i].y;
    }
    return;
}

void circular_shift2D(double *f1, double *f2, int dims1, int dims2, int N1_shift, int N2_shift)
{
    int n1, n2;
    for (int i = 0; i < dims1*dims2; ++i){
        n1 = (i/dims2 + N1_shift)%dims1;
        n2 = (i%dims2 + N2_shift)%dims2; 
        if (n1 < 0){
            n1 = n1 + dims1;     
        }
        if (n2 < 0){
            n2 = n2 + dims2;
        }

        f2[n1*dims2 + n2] = f1[i];
    }
    return;
}

// Fourier shift factor
__global__ void fourier_shift_factor_dev(cufftComplex *f, int dims1, int dims2, float N1_shift, float N2_shift)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    float n1, n2;
    cufftComplex res;
    float s, c, e;
    
    for (int i = threadID; i < dims1*dims2; i += numThreads)
    {   
        n1 = (float) (i/dims2) - (float) dims1/2.0f;
        n2 = (float) (i%dims2) - (float) dims2/2.0f;

        f[i].x = 0;
        f[i].y = 2.0f * 3.1415926f * (N1_shift * (float) n1/ (float) dims1 + N2_shift * (float) n2/ (float) dims2);

        e = expf(f[i].x);
        sincosf(f[i].y, &s, &c);
        res.x = c * e;
        res.y = s * e;
        f[i] = res;        
    }
    return;

}

__global__ void fourier_shift_factor_dev(cufftDoubleComplex *f, int dims1, int dims2, double N1_shift, double N2_shift)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    double n1, n2;
    cufftDoubleComplex res;
    double s, c, e;
    
    for (int i = threadID; i < dims1*dims2; i += numThreads)
    {   
        n1 = (double) (i/dims2) - (double) dims1/2.0;
        n2 = (double) (i%dims2) - (double) dims2/2.0;

        f[i].x = 0;
        f[i].y = 2.0 * 3.1415926 * (N1_shift * (double) n1/ (double) dims1 + N2_shift * (double) n2/ (double) dims2);

        e = exp(f[i].x);
        sincos(f[i].y, &s, &c);
        res.x = c * e;
        res.y = s * e;
        f[i] = res;        
    }
    return;

}

// gradient 2D array -> 2D array
__global__ void gradient_2D_dev(cufftComplex *f1, cufftComplex *f2, int dims1, int dims2, int option)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int n1, n2; 

    if (option == 1){
        for (int i = threadID; i < dims1*dims2; i += numThreads)
        {
            n1 = i/dims2;
            n2 = i%dims2;
            if (n1 == 0){
                f2[i].x = f1[i+dims2].x - f1[i].x;
                f2[i].y = f1[i+dims2].y - f1[i].y;
            }            
            else if (n1 == dims1-1){
                f2[i].x = f1[i].x - f1[i-dims2].x;
                f2[i].y = f1[i].y - f1[i-dims2].y;
            }
            else{
                f2[i].x = (f1[i+dims2].x - f1[i-dims2].x)/2.0f;
                f2[i].y = (f1[i+dims2].y - f1[i-dims2].y)/2.0f;
            }
        }    

    }
    else if(option == 2){
        for (int i = threadID; i < dims1*dims2; i += numThreads)
        {
            n1 = i/dims2;
            n2 = i%dims2;
            if (n2 == 0){
                f2[i].x = f1[i+1].x - f1[i].x;
                f2[i].y = f1[i+1].y - f1[i].y;
            }            
            else if (n2 == dims2-1){
                f2[i].x = f1[i].x - f1[i-1].x;
                f2[i].y = f1[i].y - f1[i-1].y;
            }
            else{
                f2[i].x = (f1[i+1].x - f1[i-1].x)/2.0f;
                f2[i].y = (f1[i+1].y - f1[i-1].y)/2.0f;
            }
        }
    }     
    return;

}

__global__ void gradient_2D_dev(cufftDoubleComplex *f1, cufftDoubleComplex *f2, int dims1, int dims2, int option)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int n1, n2; 

    if (option == 1){
        for (int i = threadID; i < dims1*dims2; i += numThreads)
        {
            n1 = i/dims2;
            n2 = i%dims2;
            if (n1 == 0){
                f2[i].x = f1[i+dims2].x - f1[i].x;
                f2[i].y = f1[i+dims2].y - f1[i].y;
            }            
            else if (n1 == dims1-1){
                f2[i].x = f1[i].x - f1[i-dims2].x;
                f2[i].y = f1[i].y - f1[i-dims2].y;
            }
            else{
                f2[i].x = (f1[i+dims2].x - f1[i-dims2].x)/2.0;
                f2[i].y = (f1[i+dims2].y - f1[i-dims2].y)/2.0;
            }
        }    

    }
    else if(option == 2){
        for (int i = threadID; i < dims1*dims2; i += numThreads)
        {
            n1 = i/dims2;
            n2 = i%dims2;
            if (n2 == 0){
                f2[i].x = f1[i+1].x - f1[i].x;
                f2[i].y = f1[i+1].y - f1[i].y;
            }            
            else if (n2 == dims2-1){
                f2[i].x = f1[i].x - f1[i-1].x;
                f2[i].y = f1[i].y - f1[i-1].y;
            }
            else{
                f2[i].x = (f1[i+1].x - f1[i-1].x)/2.0;
                f2[i].y = (f1[i+1].y - f1[i-1].y)/2.0;
            }
        }
    }     
    return;

}


float Cusum_real(cufftComplex *f, size_t size)
{
    float output = 0.0f;
    float2 *tmp_result1;
    tmp_result1 = new float2[size]; 

    cudaMemcpy(tmp_result1, f, sizeof(float2)*size, cudaMemcpyDeviceToHost);
    for(size_t i=0; i<size; i++){
        output += tmp_result1[i].x; 
    }
    //std::cout << output << std::endl;

    delete [] tmp_result1; 
    return output;
}


// print array for debugging

void print_array_dev(float *a, size_t dims1, size_t dims2)
{
    
    float *tmp_result1;
    tmp_result1 = new float[dims1*dims2]; 

    cudaMemcpy(tmp_result1, a, sizeof(float)*dims1*dims2, cudaMemcpyDeviceToHost);

    for(size_t x=0; x<dims2; x++){
        for(size_t y=0; y<dims1; y++){
            std::cout.precision(3);
            std::cout <<"[" <<std::scientific<< tmp_result1[(y*dims2)+x] << "] ";
        }
        std::cout<<std::endl;
    }

    delete [] tmp_result1; 
}

void print_array_dev(cufftComplex *a, size_t dims1, size_t dims2)
{
    
    float2 *tmp_result1;
    tmp_result1 = new float2[dims1*dims2]; 

    cudaMemcpy(tmp_result1, a, sizeof(float2)*dims1*dims2, cudaMemcpyDeviceToHost);

    for(size_t x=0; x<dims2; x++){
        for(size_t y=0; y<dims1; y++){
            std::cout.precision(3);
            std::cout <<"[" <<std::scientific<< tmp_result1[(y*dims2)+x].x << ", " << tmp_result1[(y*dims2)+x].y << "] ";
        }
        std::cout<<std::endl;
    }

    delete [] tmp_result1; 
}

void print_array_dev(double *a, size_t dims1, size_t dims2)
{
    
    double *tmp_result1;
    tmp_result1 = new double[dims1*dims2]; 

    cudaMemcpy(tmp_result1, a, sizeof(double)*dims1*dims2, cudaMemcpyDeviceToHost);

    for(size_t x=0; x<dims2; x++){
        for(size_t y=0; y<dims1; y++){
            std::cout.precision(3);
            std::cout <<"[" <<std::scientific<< tmp_result1[(y*dims2)+x] << "] ";
        }
        std::cout<<std::endl;
    }

    delete [] tmp_result1; 
}


void print_array_dev(cufftDoubleComplex *a, size_t dims1, size_t dims2)
{
    
    double2 *tmp_result1;
    tmp_result1 = new double2[dims1*dims2]; 

    cudaMemcpy(tmp_result1, a, sizeof(double2)*dims1*dims2, cudaMemcpyDeviceToHost);

    for(size_t x=0; x<dims2; x++){
        for(size_t y=0; y<dims1; y++){
            std::cout.precision(3);
            std::cout <<"[" <<std::scientific<< tmp_result1[(y*dims2)+x].x << ", " << tmp_result1[(y*dims2)+x].y << "] ";
        }
        std::cout<<std::endl;
    }

    delete [] tmp_result1; 
}


void print_array(float *a, size_t dims1, size_t dims2)
{

     for(size_t x=0; x<dims2; x++){
        for(size_t y=0; y<dims1; y++){
            std::cout.precision(3);
            std::cout <<"[" <<std::scientific<< a[(y*dims2)+x] << "] ";
        }
        std::cout<<std::endl;
    }

}

void print_array(float2 *a, size_t dims1, size_t dims2)
{
    for(size_t x=0; x<dims2; x++){
        for(size_t y=0; y<dims1; y++){
            std::cout.precision(3);
            std::cout <<"[" <<std::scientific<< a[(y*dims2)+x].x << ", " << a[(y*dims2)+x].y << "] ";
        }
        std::cout<<std::endl;
    }

}

void print_array(double *a, size_t dims1, size_t dims2)
{
        for(size_t x=0; x<dims2; x++){
        for(size_t y=0; y<dims1; y++){
            std::cout.precision(3);
            std::cout <<"[" <<std::scientific<< a[(y*dims2)+x] << "] ";
        }
        std::cout<<std::endl;
    }

}


void print_array(double2 *a, size_t dims1, size_t dims2)
{
        for(size_t x=0; x<dims2; x++){
        for(size_t y=0; y<dims1; y++){
            std::cout.precision(3);
            std::cout <<"[" <<std::scientific<< a[(y*dims2)+x].x << ", " << a[(y*dims2)+x].y << "] ";
        }
        std::cout<<std::endl;
    }

}




