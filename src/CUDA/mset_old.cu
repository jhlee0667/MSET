#include "mset.cuh"

// constructor
mset::paras::paras(std::array<size_t,3> rec_size) : rec_size(rec_size){

    RVol = new float[rec_size[0]*rec_size[1]*rec_size[2]];
    init_wave2D = new float2[rec_size[1]*rec_size[2]];
    prop2D = new float2[rec_size[1]*rec_size[2]];
    ifftshift_prop2D = new float2[rec_size[1]*rec_size[2]];
    error_array = new float[rec_size[1]*rec_size[2]];

    cudaMalloc(&dev_RVol, sizeof(float) * rec_size[0]*rec_size[1]*rec_size[2]);
    cudaMalloc(&dev_trans_fun, sizeof(float2) * rec_size[0]*rec_size[1]*rec_size[2]);
    cudaMalloc(&dev_conj_trans_fun, sizeof(float2) * rec_size[0]*rec_size[1]*rec_size[2]);
    cudaMalloc(&dev_ifftshift_prop2D, sizeof(float2) * rec_size[1]*rec_size[2]);
    cudaMalloc(&dev_ifftshift_back_prop2D, sizeof(float2) * rec_size[1]*rec_size[2]);
    cudaMalloc(&dev_init_wave2D, sizeof(float2) * rec_size[1]*rec_size[2]);
    cudaMalloc(&dev_wave2D, sizeof(float2) * rec_size[1]*rec_size[2]);
    cudaMalloc(&dev_save_wave3D, sizeof(float2) * rec_size[0]*rec_size[1]*rec_size[2]);
    cudaMalloc(&dev_tmp_4Dcell_array, sizeof(float) * rec_size[1]*rec_size[2]);
    cudaMalloc(&dev_tmp_ifftshift_4Dcell_array, sizeof(float) * rec_size[1]*rec_size[2]);
    cudaMalloc(&dev_fourier_inten, sizeof(float) * rec_size[1]*rec_size[2]);
    cudaMalloc(&dev_residual_vector, sizeof(float2) * rec_size[1]*rec_size[2]);
    cudaMalloc(&dev_grad_complex, sizeof(float2) * rec_size[0]*rec_size[1]*rec_size[2]);
    cudaMalloc(&dev_grad, sizeof(float) * rec_size[0]*rec_size[1]*rec_size[2]);
    cudaMalloc(&dev_grad2d, sizeof(float2) * rec_size[0]*rec_size[1]*rec_size[2]);
    cudaMalloc(&dev_tmp, sizeof(float2) * rec_size[0]*rec_size[1]*rec_size[2]);
    cudaMalloc(&dev_error_array, sizeof(float) * rec_size[1]*rec_size[2]);
    cudaMalloc(&dev_tmp_error_array, sizeof(float) * rec_size[1]*rec_size[2]);

    // init.
    cudaMemset((void*) dev_error_array, 0, sizeof(float) * rec_size[1]*rec_size[2]);
}

// destructor
mset::paras::~paras(){
    //std::cout << "Call destructor" <<std::endl;

    cudaFree (dev_RVol);
    cudaFree (dev_trans_fun); 
    cudaFree (dev_conj_trans_fun); 
    cudaFree (dev_ifftshift_prop2D);
    cudaFree (dev_ifftshift_back_prop2D);
    cudaFree (dev_init_wave2D);
    cudaFree (dev_wave2D);
    cudaFree (dev_save_wave3D);
    cudaFree (dev_tmp_4Dcell_array);
    cudaFree (dev_tmp_ifftshift_4Dcell_array);
    cudaFree (dev_fourier_inten);
    cudaFree (dev_residual_vector);
    cudaFree (dev_grad_complex);
    cudaFree (dev_grad);
    cudaFree (dev_grad2d);
    cudaFree (dev_tmp);
    cudaFree (dev_error_array);
    cudaFree (dev_tmp_error_array);
 
    delete [] RVol;
    delete [] init_wave2D;
    delete [] prop2D;  
    delete [] ifftshift_prop2D;
    delete [] error_array;

}






void mset::v0::run_main_mset(std::shared_ptr<mset::paras> stem_paras){

    unsigned long dims[3] = {stem_paras->rec_size[0], stem_paras->rec_size[1], stem_paras->rec_size[2]};
    size_t pot_array_centers[2] = {(size_t) round((stem_paras->rec_size[1])/2), (size_t) round((stem_paras->rec_size[1])/2)};
    float po_pr_ratio = stem_paras->potential_pixelsize[0]/stem_paras->probe_step_size[0];
    size_t scan_size[2] = {(size_t) stem_paras->N_scan_y[0], (size_t) stem_paras->N_scan_x[0]};
    size_t scan_array_centers[2] = {(size_t) round(( (float) scan_size[0]+1.0f)/2.0f), (size_t) round(((float) scan_size[1]+1.0f)/2.0f)};


    // --------------------------------------------------------
    const int BATCH = 1;
    int dims_2d[2] = {(int) dims[1], (int) dims[2]};
    cufftComplex ifft_scale_factor = make_cuFloatComplex ((float) 1/((float) dims[1]* (float) dims[2]), 0.0f);
    cufftComplex complex_sigma = make_cuFloatComplex ((float) stem_paras->sigma[0], 0.0f);
    cufftComplex complex_i = make_cuFloatComplex (0.0f, 1.0f);
    cufftComplex complex_pot_size = make_cuFloatComplex ((float) stem_paras->potential_pixelsize[0], 0.0f); 
    //std::cout<< "(" << complex_sigma.x << ", "<< complex_sigma.y << ")" << std::endl; 

    // Create a 2D FFT plan. 
    cufftHandle plan;
    if (cufftPlanMany(&plan, 2, (int *) dims_2d,
                        NULL, 1, 0, // *inembed, istride, idist 
                        NULL, 1, 0, // *onembed, ostride, odist
                        CUFFT_C2C, BATCH) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: Plan creation failed");
        return;	
    }	
    // --------------------------------------------------------

    /* -------------------  main part  --------------------- */
    for (size_t scan_x = 1; scan_x <= scan_size[1]; ++scan_x){
        for (size_t scan_y = 1; scan_y <= scan_size[0]; ++scan_y){

            // init. wave2D
            //cudaMemcpy(stem_paras->dev_wave2D, stem_paras->dev_init_wave2D, sizeof(float2)*dims[1]*dims[2], cudaMemcpyDeviceToDevice);
            circular_shift2D_dev<<<256, 256>>>(stem_paras->dev_init_wave2D, stem_paras->dev_wave2D, (int) dims[1], (int) dims[2], (int) (((float) scan_y- (float) scan_array_centers[0])/po_pr_ratio), (int) (((float) scan_x- (float) scan_array_centers[1])/po_pr_ratio));
            
                     
            // ------------ transmision function ----------------------
            // copy RVol -> trans
            datatransfer_F2C<<<256, 256>>>(stem_paras->dev_trans_fun, stem_paras->dev_RVol, (size_t) dims[0]*dims[1]*dims[2]);


            // calculate trans function
            CuPointwiseMul<<<1024, 256>>>((cufftComplex) complex_pot_size, (cufftComplex *) stem_paras->dev_trans_fun, dims[0]*dims[1]*dims[2]);
            CuPointwiseMul<<<1024, 256>>>((cufftComplex) complex_sigma, (cufftComplex *) stem_paras->dev_trans_fun, dims[0]*dims[1]*dims[2]);
            CuPointwiseMul<<<1024, 256>>>((cufftComplex) complex_i, (cufftComplex *) stem_paras->dev_trans_fun, dims[0]*dims[1]*dims[2]);
            CuPointwiseExp<<<1024, 256>>>((cufftComplex *) stem_paras->dev_trans_fun, dims[0]*dims[1]*dims[2]);
            cudaMemcpy(stem_paras->dev_conj_trans_fun, stem_paras->dev_trans_fun, sizeof(float2)*dims[0]*dims[1]*dims[2], cudaMemcpyDeviceToDevice);
            CuPointwiseConj<<<256, 256>>>((cufftComplex *) stem_paras->dev_conj_trans_fun, dims[0]*dims[1]*dims[2]);
            // -------------------------------------------------------- 

            
            // FORWARD PROPAGATION calculation
            for (size_t i = 0; i< dims[0]; ++i){

                // trans_function * wave_2D
                CuPointwiseMul<<<1024, 256>>>((cufftComplex *) stem_paras->dev_trans_fun, (cufftComplex *) stem_paras->dev_wave2D, dims[1]*dims[2], i);

                // CUFFT FORWARD
                if (cufftExecC2C(plan, (cufftComplex *) stem_paras->dev_wave2D, (cufftComplex *) stem_paras->dev_wave2D, CUFFT_FORWARD) != CUFFT_SUCCESS){
                    fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
                    return;	
                }

                // ifftshift_prop2D * wave_2D
                CuPointwiseMul<<<1024, 256>>>((cufftComplex *) stem_paras->dev_ifftshift_prop2D, (cufftComplex *) stem_paras->dev_wave2D, dims[1]*dims[2]); 

                // CUFFT INVERSE
                if (cufftExecC2C(plan, (cufftComplex *) stem_paras->dev_wave2D, (cufftComplex *) stem_paras->dev_wave2D, CUFFT_INVERSE) != CUFFT_SUCCESS){
                    fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
                    return;	
                }
                // scaling
                CuPointwiseMul<<<1024, 256>>>((cufftComplex) ifft_scale_factor, (cufftComplex *) stem_paras->dev_wave2D, dims[1]*dims[2]);
                
                // save the 2D wave funtion into 3D volume
                cudaMemcpy(stem_paras->dev_save_wave3D + i*dims[1]*dims[2], stem_paras->dev_wave2D, sizeof(float2)*dims[1]*dims[2], cudaMemcpyDeviceToDevice); 

            }
            // -------------------------------------------------------- 
            
            // calculate residual vector
            // CUFFT FORWARD            
            if (cufftExecC2C(plan, (cufftComplex *) stem_paras->dev_wave2D, (cufftComplex *) stem_paras->dev_wave2D, CUFFT_FORWARD) != CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
                return;	
            }

            CuPointwiseAbsSquare<<<1024, 256>>>((cufftComplex *) stem_paras->dev_wave2D, (float *) stem_paras->dev_fourier_inten, dims[1]*dims[2]);

            CuPointwiseAdd<<<1024, 256>>>((float) pow(10,-30), (float *) stem_paras->dev_fourier_inten, dims[1]*dims[2]); // preventing divergence 
            // square root
            CuPointwisePow<<<1024, 256>>>((float *) stem_paras->dev_fourier_inten, 0.5f, dims[1]*dims[2]);
            

            // get measured 4D STEM data
            stem_paras->pMxCell_4DSTEM_element = mxGetCell(stem_paras->pMxCell_4DSTEM, (scan_y-1)*scan_size[1]+(scan_x-1));
            stem_paras->tmp_4Dcell_array = mxGetSingles(stem_paras->pMxCell_4DSTEM_element);

            
            // ifftshift 4D STEM data
            cudaMemcpy(stem_paras->dev_tmp_4Dcell_array, stem_paras->tmp_4Dcell_array, sizeof(float)*dims[1]*dims[2], cudaMemcpyHostToDevice);
           
            circular_shift2D_dev<<<256, 256>>>((float *) stem_paras->dev_tmp_4Dcell_array, (float *) stem_paras->dev_tmp_ifftshift_4Dcell_array, (int) dims[1], (int) dims[2], (int) ceil(((float)dims[1])/2.0f),(int) ceil(((float)dims[2])/2.0f));
            // square root
            CuPointwisePow<<<1024, 256>>>((float *) stem_paras->dev_tmp_ifftshift_4Dcell_array, 0.5f, dims[1]*dims[2]);
            

            // calculate error
            cudaMemcpy(stem_paras->dev_tmp_error_array, stem_paras->dev_tmp_ifftshift_4Dcell_array, sizeof(float)*dims[1]*dims[2], cudaMemcpyDeviceToDevice); 
            CuPointwiseSub<<<1024, 256>>>((float *) stem_paras->dev_fourier_inten, (float *) stem_paras->dev_tmp_error_array, dims[1]*dims[2]); 
            CuPointwiseAbsSquare<<<1024, 256>>>((float *) stem_paras->dev_tmp_error_array, dims[1]*dims[2]);
            CuPointwiseAdd<<<1024, 256>>>((float *) stem_paras->dev_tmp_error_array, (float *) stem_paras->dev_error_array, dims[1]*dims[2]);

            // divide estimated Fourier intensity by measured Fourier intensity (result: second argument)
            CuPointwiseDiv<<<1024, 256>>>((float *) stem_paras->dev_tmp_ifftshift_4Dcell_array, (float *) stem_paras->dev_fourier_inten, dims[1]*dims[2]); 
            

            // copy the above result (dev_fourier_inten) -> dev_residual_vector
            datatransfer_F2C<<<256, 256>>>(stem_paras->dev_residual_vector, stem_paras->dev_fourier_inten, (size_t) dims[1]*dims[2]);
            CuPointwiseSub<<<1024, 256>>>(make_cuFloatComplex (1.0f, 0.0f), stem_paras->dev_residual_vector, (size_t) dims[1]*dims[2]);
             
            CuPointwiseMul<<<1024, 256>>>((cufftComplex *) stem_paras->dev_residual_vector, (cufftComplex *) stem_paras->dev_wave2D, dims[1]*dims[2]);

            // -------------------------------------------------------- 

            
            // BACK PROPAGATION calculation
            for (int i = dims[0]-1; i >= 0; --i){

                // ifftshift_back_prop2D * wave_2D 
                CuPointwiseMul<<<1024, 256>>>((cufftComplex *) stem_paras->dev_ifftshift_back_prop2D, (cufftComplex *) stem_paras->dev_wave2D, dims[1]*dims[2]);

                // CUFFT INVERSE
                if (cufftExecC2C(plan, (cufftComplex *) stem_paras->dev_wave2D, (cufftComplex *) stem_paras->dev_wave2D, CUFFT_INVERSE) != CUFFT_SUCCESS){
                    fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
                    return;	
                }
                // scaling
                CuPointwiseMul<<<1024, 256>>>((cufftComplex) ifft_scale_factor, (cufftComplex *) stem_paras->dev_wave2D, dims[1]*dims[2]);

                // save 3D grad
                cudaMemcpy(stem_paras->dev_grad2d, stem_paras->dev_wave2D, sizeof(float2)*dims[1]*dims[2], cudaMemcpyDeviceToDevice);
                cudaMemcpy(stem_paras->dev_tmp, stem_paras->dev_save_wave3D + i*dims[1]*dims[2], sizeof(float2)*dims[1]*dims[2], cudaMemcpyDeviceToDevice);
                CuPointwiseConj<<<256, 256>>>((cufftComplex *) stem_paras->dev_tmp, dims[1]*dims[2]);
                CuPointwiseMul<<<1024, 256>>>((cufftComplex *) stem_paras->dev_tmp, (cufftComplex *) stem_paras->dev_grad2d, dims[1]*dims[2]); 
                CuPointwiseMul<<<1024, 256>>>((cufftComplex *) stem_paras->dev_conj_trans_fun, (cufftComplex *) stem_paras->dev_grad2d, dims[1]*dims[2], i);
                CuPointwiseMul<<<1024, 256>>>((cufftComplex) complex_i, (cufftComplex *) stem_paras->dev_grad2d, dims[1]*dims[2]);
                cudaMemcpy(stem_paras->dev_grad_complex + i*dims[1]*dims[2], stem_paras->dev_grad2d, sizeof(float2)*dims[1]*dims[2], cudaMemcpyDeviceToDevice); 

                // conjugate_trans_function * wave_2D
                CuPointwiseMul<<<1024, 256>>>((cufftComplex *) stem_paras->dev_conj_trans_fun, (cufftComplex *) stem_paras->dev_wave2D, dims[1]*dims[2], i);

                // CUFFT FORWARD
                if (cufftExecC2C(plan, (cufftComplex *) stem_paras->dev_wave2D, (cufftComplex *) stem_paras->dev_wave2D, CUFFT_FORWARD) != CUFFT_SUCCESS){
                    fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
                    return;	
                }

            } 


            // update
            datatransfer_C2F<<<256, 256>>>(stem_paras->dev_grad, stem_paras->dev_grad_complex, (size_t) dims[0]*dims[1]*dims[2]);
            CuPointwiseMul<<<1024, 256>>>((float) stem_paras->step_size[0], (float *) stem_paras->dev_grad, dims[0]*dims[1]*dims[2]); 
            CuPointwiseAdd<<<1024, 256>>>((float *) stem_paras->dev_grad, (float *) stem_paras->dev_RVol, dims[0]*dims[1]*dims[2]); 
            // positivity
            //CuPositivity<<<256, 256>>>((float *) stem_paras->dev_RVol, dims[0]*dims[1]*dims[2]);

        }
    }
}










void mset::v0::run_main_sset(std::shared_ptr<mset::paras> stem_paras){

    unsigned long dims[3] = {stem_paras->rec_size[0], stem_paras->rec_size[1], stem_paras->rec_size[2]};
    size_t pot_array_centers[2] = {(size_t) round((stem_paras->rec_size[1])/2), (size_t) round((stem_paras->rec_size[1])/2)};
    float po_pr_ratio = stem_paras->potential_pixelsize[0]/stem_paras->probe_step_size[0];
    size_t scan_size[2] = {(size_t) stem_paras->N_scan_y[0], (size_t) stem_paras->N_scan_x[0]};
    size_t scan_array_centers[2] = {(size_t) round(( (float) scan_size[0]+1.0f)/2.0f), (size_t) round(((float) scan_size[1]+1.0f)/2.0f)};


    // --------------------------------------------------------
    const int BATCH = 1;
    int dims_2d[2] = {(int) dims[1], (int) dims[2]};
    cufftComplex ifft_scale_factor = make_cuFloatComplex ((float) 1/((float) dims[1]* (float) dims[2]), 0.0f);
    cufftComplex complex_sigma = make_cuFloatComplex ((float) stem_paras->sigma[0], 0.0f);
    cufftComplex complex_i = make_cuFloatComplex (0.0f, 1.0f);
    cufftComplex complex_pot_size = make_cuFloatComplex ((float) stem_paras->potential_pixelsize[0], 0.0f); 
    //std::cout<< "(" << complex_sigma.x << ", "<< complex_sigma.y << ")" << std::endl; 

    // Create a 2D FFT plan. 
    cufftHandle plan;
    if (cufftPlanMany(&plan, 2, (int *) dims_2d,
                        NULL, 1, 0, // *inembed, istride, idist 
                        NULL, 1, 0, // *onembed, ostride, odist
                        CUFFT_C2C, BATCH) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: Plan creation failed");
        return;	
    }	
    // --------------------------------------------------------

    /* -------------------  main part  --------------------- */
    for (size_t scan_x = 1; scan_x <= scan_size[1]; ++scan_x){
        for (size_t scan_y = 1; scan_y <= scan_size[0]; ++scan_y){

            // init. wave2D
            //cudaMemcpy(stem_paras->dev_wave2D, stem_paras->dev_init_wave2D, sizeof(float2)*dims[1]*dims[2], cudaMemcpyDeviceToDevice);
            circular_shift2D_dev<<<256, 256>>>(stem_paras->dev_init_wave2D, stem_paras->dev_wave2D, (int) dims[1], (int) dims[2], (int) (((float) scan_y- (float) scan_array_centers[0])/po_pr_ratio), (int) (((float) scan_x- (float) scan_array_centers[1])/po_pr_ratio));
            
                     
            // ------------ transmision function ----------------------
            // copy RVol -> trans
            datatransfer_F2C<<<256, 256>>>(stem_paras->dev_trans_fun, stem_paras->dev_RVol, (size_t) dims[0]*dims[1]*dims[2]);


            // calculate trans function
            for (size_t i = 0+1; i< dims[0]; ++i){
                CuPointwiseAdd<<<1024, 256>>>((cufftComplex *) stem_paras->dev_trans_fun+ i*dims[1]*dims[2],(cufftComplex *) stem_paras->dev_trans_fun, dims[1]*dims[2]);  
            }
            CuPointwiseMul<<<1024, 256>>>((cufftComplex) complex_pot_size, (cufftComplex *) stem_paras->dev_trans_fun, dims[1]*dims[2]);
            CuPointwiseMul<<<1024, 256>>>((cufftComplex) complex_sigma, (cufftComplex *) stem_paras->dev_trans_fun, dims[1]*dims[2]);
            CuPointwiseMul<<<1024, 256>>>((cufftComplex) complex_i, (cufftComplex *) stem_paras->dev_trans_fun, dims[1]*dims[2]);
            CuPointwiseExp<<<1024, 256>>>((cufftComplex *) stem_paras->dev_trans_fun, dims[1]*dims[2]);
            cudaMemcpy(stem_paras->dev_conj_trans_fun, stem_paras->dev_trans_fun, sizeof(float2)*dims[0]*dims[1]*dims[2], cudaMemcpyDeviceToDevice);
            CuPointwiseConj<<<256, 256>>>((cufftComplex *) stem_paras->dev_conj_trans_fun, dims[1]*dims[2]);
            // -------------------------------------------------------- 

            
            // FORWARD PROPAGATION calculation
            CuPointwiseMul<<<1024, 256>>>((cufftComplex *) stem_paras->dev_trans_fun, (cufftComplex *) stem_paras->dev_wave2D, dims[1]*dims[2]);
            // -------------------------------------------------------- 
            
            // calculate residual vector
            // CUFFT FORWARD            
            if (cufftExecC2C(plan, (cufftComplex *) stem_paras->dev_wave2D, (cufftComplex *) stem_paras->dev_wave2D, CUFFT_FORWARD) != CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
                return;	
            }

            CuPointwiseAbsSquare<<<1024, 256>>>((cufftComplex *) stem_paras->dev_wave2D, (float *) stem_paras->dev_fourier_inten, dims[1]*dims[2]);

            CuPointwiseAdd<<<1024, 256>>>((float) pow(10,-30), (float *) stem_paras->dev_fourier_inten, dims[1]*dims[2]); // preventing divergence
            // square root
            CuPointwisePow<<<1024, 256>>>((float *) stem_paras->dev_fourier_inten, 0.5f, dims[1]*dims[2]);
            

            // get measured 4D STEM data
            stem_paras->pMxCell_4DSTEM_element = mxGetCell(stem_paras->pMxCell_4DSTEM, (scan_y-1)*scan_size[1]+(scan_x-1));
            stem_paras->tmp_4Dcell_array = mxGetSingles(stem_paras->pMxCell_4DSTEM_element);

            
            // ifftshift 4D STEM data
            cudaMemcpy(stem_paras->dev_tmp_4Dcell_array, stem_paras->tmp_4Dcell_array, sizeof(float)*dims[1]*dims[2], cudaMemcpyHostToDevice);
           
            circular_shift2D_dev<<<256, 256>>>((float *) stem_paras->dev_tmp_4Dcell_array, (float *) stem_paras->dev_tmp_ifftshift_4Dcell_array, (int) dims[1], (int) dims[2], (int) ceil(((float)dims[1])/2.0f),(int) ceil(((float)dims[2])/2.0f));
            // square root
            CuPointwisePow<<<1024, 256>>>((float *) stem_paras->dev_tmp_ifftshift_4Dcell_array, 0.5f, dims[1]*dims[2]);
            

            // calculate error
            cudaMemcpy(stem_paras->dev_tmp_error_array, stem_paras->dev_tmp_ifftshift_4Dcell_array, sizeof(float)*dims[1]*dims[2], cudaMemcpyDeviceToDevice); 
            CuPointwiseSub<<<1024, 256>>>((float *) stem_paras->dev_fourier_inten, (float *) stem_paras->dev_tmp_error_array, dims[1]*dims[2]); 
            CuPointwiseAbsSquare<<<1024, 256>>>((float *) stem_paras->dev_tmp_error_array, dims[1]*dims[2]);
            CuPointwiseAdd<<<1024, 256>>>((float *) stem_paras->dev_tmp_error_array, (float *) stem_paras->dev_error_array, dims[1]*dims[2]);

            // divide estimated Fourier intensity by measured Fourier intensity (result: second argument)
            CuPointwiseDiv<<<1024, 256>>>((float *) stem_paras->dev_tmp_ifftshift_4Dcell_array, (float *) stem_paras->dev_fourier_inten, dims[1]*dims[2]); 
            

            // copy the above result (dev_fourier_inten) -> dev_residual_vector
            datatransfer_F2C<<<256, 256>>>(stem_paras->dev_residual_vector, stem_paras->dev_fourier_inten, (size_t) dims[1]*dims[2]);
            CuPointwiseSub<<<1024, 256>>>(make_cuFloatComplex (1.0f, 0.0f), stem_paras->dev_residual_vector, (size_t) dims[1]*dims[2]);
             
            CuPointwiseMul<<<1024, 256>>>((cufftComplex *) stem_paras->dev_residual_vector, (cufftComplex *) stem_paras->dev_wave2D, dims[1]*dims[2]);
            // --------------------------------------------------------  

            

            // BACK PROPAGATION calculation
            // CUFFT INVERSE
            if (cufftExecC2C(plan, (cufftComplex *) stem_paras->dev_wave2D, (cufftComplex *) stem_paras->dev_wave2D, CUFFT_INVERSE) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
            return;	
            }
            // scaling
            CuPointwiseMul<<<1024, 256>>>((cufftComplex) ifft_scale_factor, (cufftComplex *) stem_paras->dev_wave2D, dims[1]*dims[2]);

            circular_shift2D_dev<<<256, 256>>>(stem_paras->dev_init_wave2D, stem_paras->dev_tmp, (int) dims[1], (int) dims[2], (int) (((float) scan_y- (float) scan_array_centers[0])/po_pr_ratio), (int) (((float) scan_x- (float) scan_array_centers[1])/po_pr_ratio));
            CuPointwiseConj<<<256, 256>>>((cufftComplex *) stem_paras->dev_tmp, dims[1]*dims[2]);
            CuPointwiseMul<<<1024, 256>>>((cufftComplex *) stem_paras->dev_tmp, (cufftComplex *) stem_paras->dev_wave2D, dims[1]*dims[2]);
            CuPointwiseMul<<<1024, 256>>>((cufftComplex *) stem_paras->dev_conj_trans_fun, (cufftComplex *) stem_paras->dev_wave2D, dims[1]*dims[2]);
            cudaMemcpy(stem_paras->dev_grad2d, stem_paras->dev_wave2D, sizeof(float2)*dims[1]*dims[2], cudaMemcpyDeviceToDevice);
            CuPointwiseMul<<<1024, 256>>>((cufftComplex) complex_i, (cufftComplex *) stem_paras->dev_grad2d, dims[1]*dims[2]);
            
            //back projection
            for (int i = dims[0]-1; i >= 0; --i){
                cudaMemcpy(stem_paras->dev_grad_complex + i*dims[1]*dims[2], stem_paras->dev_grad2d, sizeof(float2)*dims[1]*dims[2], cudaMemcpyDeviceToDevice);
            } 
            

            // update
            datatransfer_C2F<<<256, 256>>>(stem_paras->dev_grad, stem_paras->dev_grad_complex, (size_t) dims[0]*dims[1]*dims[2]);
            CuPointwiseMul<<<1024, 256>>>((float) stem_paras->step_size[0], (float *) stem_paras->dev_grad, dims[0]*dims[1]*dims[2]); 
            CuPointwiseAdd<<<1024, 256>>>((float *) stem_paras->dev_grad, (float *) stem_paras->dev_RVol, dims[0]*dims[1]*dims[2]); 
            // positivity
            //CuPositivity<<<256, 256>>>((float *) stem_paras->dev_RVol, dims[0]*dims[1]*dims[2]);

        }
    }
}










void mset::v0::run_error_mset(std::shared_ptr<mset::paras> stem_paras){

    unsigned long dims[3] = {stem_paras->rec_size[0], stem_paras->rec_size[1], stem_paras->rec_size[2]};
    size_t pot_array_centers[2] = {(size_t) round((stem_paras->rec_size[1])/2), (size_t) round((stem_paras->rec_size[1])/2)};
    float po_pr_ratio = stem_paras->potential_pixelsize[0]/stem_paras->probe_step_size[0];
    size_t scan_size[2] = {(size_t) stem_paras->N_scan_y[0], (size_t) stem_paras->N_scan_x[0]};
    size_t scan_array_centers[2] = {(size_t) round(( (float) scan_size[0]+1.0f)/2.0f), (size_t) round(((float) scan_size[1]+1.0f)/2.0f)};


    // --------------------------------------------------------
    const int BATCH = 1;
    int dims_2d[2] = {(int) dims[1], (int) dims[2]};
    cufftComplex ifft_scale_factor = make_cuFloatComplex ((float) 1/((float) dims[1]* (float) dims[2]), 0.0f);
    cufftComplex complex_sigma = make_cuFloatComplex ((float) stem_paras->sigma[0], 0.0f);
    cufftComplex complex_i = make_cuFloatComplex (0.0f, 1.0f);
    cufftComplex complex_pot_size = make_cuFloatComplex ((float) stem_paras->potential_pixelsize[0], 0.0f); 
    //std::cout<< "(" << complex_sigma.x << ", "<< complex_sigma.y << ")" << std::endl; 

    // Create a 2D FFT plan. 
    cufftHandle plan;
    if (cufftPlanMany(&plan, 2, (int *) dims_2d,
                        NULL, 1, 0, // *inembed, istride, idist 
                        NULL, 1, 0, // *onembed, ostride, odist
                        CUFFT_C2C, BATCH) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: Plan creation failed");
        return;	
    }	
    // --------------------------------------------------------

    /* -------------------  main part  --------------------- */
    for (size_t scan_x = 1; scan_x <= scan_size[1]; ++scan_x){
        for (size_t scan_y = 1; scan_y <= scan_size[0]; ++scan_y){

            // init. wave2D
            //cudaMemcpy(stem_paras->dev_wave2D, stem_paras->dev_init_wave2D, sizeof(float2)*dims[1]*dims[2], cudaMemcpyDeviceToDevice);
            circular_shift2D_dev<<<256, 256>>>(stem_paras->dev_init_wave2D, stem_paras->dev_wave2D, (int) dims[1], (int) dims[2], (int) (((float) scan_y- (float) scan_array_centers[0])/po_pr_ratio), (int) (((float) scan_x- (float) scan_array_centers[1])/po_pr_ratio));
            
                        
            // ------------ transmision function ----------------------
            // copy RVol -> trans
            datatransfer_F2C<<<256, 256>>>(stem_paras->dev_trans_fun, stem_paras->dev_RVol, (size_t) dims[0]*dims[1]*dims[2]);


            // calculate trans function
            CuPointwiseMul<<<1024, 256>>>((cufftComplex) complex_pot_size, (cufftComplex *) stem_paras->dev_trans_fun, dims[0]*dims[1]*dims[2]);
            CuPointwiseMul<<<1024, 256>>>((cufftComplex) complex_sigma, (cufftComplex *) stem_paras->dev_trans_fun, dims[0]*dims[1]*dims[2]);
            CuPointwiseMul<<<1024, 256>>>((cufftComplex) complex_i, (cufftComplex *) stem_paras->dev_trans_fun, dims[0]*dims[1]*dims[2]);
            CuPointwiseExp<<<1024, 256>>>((cufftComplex *) stem_paras->dev_trans_fun, dims[0]*dims[1]*dims[2]);
            cudaMemcpy(stem_paras->dev_conj_trans_fun, stem_paras->dev_trans_fun, sizeof(float2)*dims[0]*dims[1]*dims[2], cudaMemcpyDeviceToDevice);
            CuPointwiseConj<<<256, 256>>>((cufftComplex *) stem_paras->dev_conj_trans_fun, dims[0]*dims[1]*dims[2]);
            // -------------------------------------------------------- 

            
            // FORWARD calculation
            for (size_t i = 0; i< dims[0]; ++i){

                // trans_function * wave_2D
                CuPointwiseMul<<<1024, 256>>>((cufftComplex *) stem_paras->dev_trans_fun, (cufftComplex *) stem_paras->dev_wave2D, dims[1]*dims[2], i);

                // CUFFT FORWARD
                if (cufftExecC2C(plan, (cufftComplex *) stem_paras->dev_wave2D, (cufftComplex *) stem_paras->dev_wave2D, CUFFT_FORWARD) != CUFFT_SUCCESS){
                    fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
                    return;	
                }

                // ifftshift_prop2D * wave_2D
                CuPointwiseMul<<<1024, 256>>>((cufftComplex *) stem_paras->dev_ifftshift_prop2D, (cufftComplex *) stem_paras->dev_wave2D, dims[1]*dims[2]); 

                // CUFFT INVERSE
                if (cufftExecC2C(plan, (cufftComplex *) stem_paras->dev_wave2D, (cufftComplex *) stem_paras->dev_wave2D, CUFFT_INVERSE) != CUFFT_SUCCESS){
                    fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
                    return;	
                }
                // scaling
                CuPointwiseMul<<<1024, 256>>>((cufftComplex) ifft_scale_factor, (cufftComplex *) stem_paras->dev_wave2D, dims[1]*dims[2]);
                
                // save the 2D wave funtion into 3D volume
                cudaMemcpy(stem_paras->dev_save_wave3D + i*dims[1]*dims[2], stem_paras->dev_wave2D, sizeof(float2)*dims[1]*dims[2], cudaMemcpyDeviceToDevice); 

            }
            // -------------------------------------------------------- 
            
            // calculate residual vector
            // CUFFT FORWARD            
            if (cufftExecC2C(plan, (cufftComplex *) stem_paras->dev_wave2D, (cufftComplex *) stem_paras->dev_wave2D, CUFFT_FORWARD) != CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
                return;	
            }

            CuPointwiseAbsSquare<<<1024, 256>>>((cufftComplex *) stem_paras->dev_wave2D, (float *) stem_paras->dev_fourier_inten, dims[1]*dims[2]);

            CuPointwiseAdd<<<1024, 256>>>((float) pow(10,-30), (float *) stem_paras->dev_fourier_inten, dims[1]*dims[2]); 
            // square root
            CuPointwisePow<<<1024, 256>>>((float *) stem_paras->dev_fourier_inten, 0.5f, dims[1]*dims[2]);
            

            // get measured 4D STEM data
            stem_paras->pMxCell_4DSTEM_element = mxGetCell(stem_paras->pMxCell_4DSTEM, (scan_y-1)*scan_size[1]+(scan_x-1));
            stem_paras->tmp_4Dcell_array = mxGetSingles(stem_paras->pMxCell_4DSTEM_element);

            
            // ifftshift 4D STEM data
            cudaMemcpy(stem_paras->dev_tmp_4Dcell_array, stem_paras->tmp_4Dcell_array, sizeof(float)*dims[1]*dims[2], cudaMemcpyHostToDevice);
            
            circular_shift2D_dev<<<256, 256>>>((float *) stem_paras->dev_tmp_4Dcell_array, (float *) stem_paras->dev_tmp_ifftshift_4Dcell_array, (int) dims[1], (int) dims[2], (int) ceil(((float)dims[1])/2.0f),(int) ceil(((float)dims[2])/2.0f));
            // square root
            CuPointwisePow<<<1024, 256>>>((float *) stem_paras->dev_tmp_ifftshift_4Dcell_array, 0.5f, dims[1]*dims[2]);
            

            // calculate error
            cudaMemcpy(stem_paras->dev_tmp_error_array, stem_paras->dev_tmp_ifftshift_4Dcell_array, sizeof(float)*dims[1]*dims[2], cudaMemcpyDeviceToDevice); 
            CuPointwiseSub<<<1024, 256>>>((float *) stem_paras->dev_fourier_inten, (float *) stem_paras->dev_tmp_error_array, dims[1]*dims[2]); 
            CuPointwiseAbsSquare<<<1024, 256>>>((float *) stem_paras->dev_tmp_error_array, dims[1]*dims[2]);
            CuPointwiseAdd<<<1024, 256>>>((float *) stem_paras->dev_tmp_error_array, (float *) stem_paras->dev_error_array, dims[1]*dims[2]);

            // divide estimated Fourier intensity by measured Fourier intensity (result: second argument)
            CuPointwiseDiv<<<1024, 256>>>((float *) stem_paras->dev_tmp_ifftshift_4Dcell_array, (float *) stem_paras->dev_fourier_inten, dims[1]*dims[2]); 
            
            // --------------------------------------------------------  

        }
    }

}



void mset::v0::run_error_sset(std::shared_ptr<mset::paras> stem_paras){

    unsigned long dims[3] = {stem_paras->rec_size[0], stem_paras->rec_size[1], stem_paras->rec_size[2]};
    size_t pot_array_centers[2] = {(size_t) round((stem_paras->rec_size[1])/2), (size_t) round((stem_paras->rec_size[1])/2)};
    float po_pr_ratio = stem_paras->potential_pixelsize[0]/stem_paras->probe_step_size[0];
    size_t scan_size[2] = {(size_t) stem_paras->N_scan_y[0], (size_t) stem_paras->N_scan_x[0]};
    size_t scan_array_centers[2] = {(size_t) round(( (float) scan_size[0]+1.0f)/2.0f), (size_t) round(((float) scan_size[1]+1.0f)/2.0f)};


    // --------------------------------------------------------
    const int BATCH = 1;
    int dims_2d[2] = {(int) dims[1], (int) dims[2]};
    cufftComplex ifft_scale_factor = make_cuFloatComplex ((float) 1/((float) dims[1]* (float) dims[2]), 0.0f);
    cufftComplex complex_sigma = make_cuFloatComplex ((float) stem_paras->sigma[0], 0.0f);
    cufftComplex complex_i = make_cuFloatComplex (0.0f, 1.0f);
    cufftComplex complex_pot_size = make_cuFloatComplex ((float) stem_paras->potential_pixelsize[0], 0.0f); 
    //std::cout<< "(" << complex_sigma.x << ", "<< complex_sigma.y << ")" << std::endl; 

    // Create a 2D FFT plan. 
    cufftHandle plan;
    if (cufftPlanMany(&plan, 2, (int *) dims_2d,
                        NULL, 1, 0, // *inembed, istride, idist 
                        NULL, 1, 0, // *onembed, ostride, odist
                        CUFFT_C2C, BATCH) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: Plan creation failed");
        return;	
    }	
    // --------------------------------------------------------

    /* -------------------  main part  --------------------- */
    for (size_t scan_x = 1; scan_x <= scan_size[1]; ++scan_x){
        for (size_t scan_y = 1; scan_y <= scan_size[0]; ++scan_y){

            // init. wave2D
            //cudaMemcpy(stem_paras->dev_wave2D, stem_paras->dev_init_wave2D, sizeof(float2)*dims[1]*dims[2], cudaMemcpyDeviceToDevice);
            circular_shift2D_dev<<<256, 256>>>(stem_paras->dev_init_wave2D, stem_paras->dev_wave2D, (int) dims[1], (int) dims[2], (int) (((float) scan_y- (float) scan_array_centers[0])/po_pr_ratio), (int) (((float) scan_x- (float) scan_array_centers[1])/po_pr_ratio));
            
                        
            // ------------ transmision function ----------------------
            // copy RVol -> trans
            datatransfer_F2C<<<256, 256>>>(stem_paras->dev_trans_fun, stem_paras->dev_RVol, (size_t) dims[0]*dims[1]*dims[2]);


            // calculate trans function
            for (size_t i = 0+1; i< dims[0]; ++i){
                CuPointwiseAdd<<<1024, 256>>>((cufftComplex *) stem_paras->dev_trans_fun+ i*dims[1]*dims[2],(cufftComplex *) stem_paras->dev_trans_fun, dims[1]*dims[2]);  
            }
            CuPointwiseMul<<<1024, 256>>>((cufftComplex) complex_pot_size, (cufftComplex *) stem_paras->dev_trans_fun, dims[1]*dims[2]);
            CuPointwiseMul<<<1024, 256>>>((cufftComplex) complex_sigma, (cufftComplex *) stem_paras->dev_trans_fun, dims[1]*dims[2]);
            CuPointwiseMul<<<1024, 256>>>((cufftComplex) complex_i, (cufftComplex *) stem_paras->dev_trans_fun, dims[1]*dims[2]);
            CuPointwiseExp<<<1024, 256>>>((cufftComplex *) stem_paras->dev_trans_fun, dims[1]*dims[2]);
            cudaMemcpy(stem_paras->dev_conj_trans_fun, stem_paras->dev_trans_fun, sizeof(float2)*dims[0]*dims[1]*dims[2], cudaMemcpyDeviceToDevice);
            CuPointwiseConj<<<256, 256>>>((cufftComplex *) stem_paras->dev_conj_trans_fun, dims[1]*dims[2]);
            // -------------------------------------------------------- 

            
            // FORWARD calculation
            CuPointwiseMul<<<1024, 256>>>((cufftComplex *) stem_paras->dev_trans_fun, (cufftComplex *) stem_paras->dev_wave2D, dims[1]*dims[2]);
            // -------------------------------------------------------- 
            
            // calculate residual vector
            // CUFFT FORWARD            
            if (cufftExecC2C(plan, (cufftComplex *) stem_paras->dev_wave2D, (cufftComplex *) stem_paras->dev_wave2D, CUFFT_FORWARD) != CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
                return;	
            }

            CuPointwiseAbsSquare<<<1024, 256>>>((cufftComplex *) stem_paras->dev_wave2D, (float *) stem_paras->dev_fourier_inten, dims[1]*dims[2]);

            CuPointwiseAdd<<<1024, 256>>>((float) pow(10,-30), (float *) stem_paras->dev_fourier_inten, dims[1]*dims[2]); 
            // square root
            CuPointwisePow<<<1024, 256>>>((float *) stem_paras->dev_fourier_inten, 0.5f, dims[1]*dims[2]);
            

            // get measured 4D STEM data
            stem_paras->pMxCell_4DSTEM_element = mxGetCell(stem_paras->pMxCell_4DSTEM, (scan_y-1)*scan_size[1]+(scan_x-1));
            stem_paras->tmp_4Dcell_array = mxGetSingles(stem_paras->pMxCell_4DSTEM_element);

            
            // ifftshift 4D STEM data
            cudaMemcpy(stem_paras->dev_tmp_4Dcell_array, stem_paras->tmp_4Dcell_array, sizeof(float)*dims[1]*dims[2], cudaMemcpyHostToDevice);
            
            circular_shift2D_dev<<<256, 256>>>((float *) stem_paras->dev_tmp_4Dcell_array, (float *) stem_paras->dev_tmp_ifftshift_4Dcell_array, (int) dims[1], (int) dims[2], (int) ceil(((float)dims[1])/2.0f),(int) ceil(((float)dims[2])/2.0f));
            // square root
            CuPointwisePow<<<1024, 256>>>((float *) stem_paras->dev_tmp_ifftshift_4Dcell_array, 0.5f, dims[1]*dims[2]);
            

            // calculate error
            cudaMemcpy(stem_paras->dev_tmp_error_array, stem_paras->dev_tmp_ifftshift_4Dcell_array, sizeof(float)*dims[1]*dims[2], cudaMemcpyDeviceToDevice); 
            CuPointwiseSub<<<1024, 256>>>((float *) stem_paras->dev_fourier_inten, (float *) stem_paras->dev_tmp_error_array, dims[1]*dims[2]); 
            CuPointwiseAbsSquare<<<1024, 256>>>((float *) stem_paras->dev_tmp_error_array, dims[1]*dims[2]);
            CuPointwiseAdd<<<1024, 256>>>((float *) stem_paras->dev_tmp_error_array, (float *) stem_paras->dev_error_array, dims[1]*dims[2]);

            // divide estimated Fourier intensity by measured Fourier intensity (result: second argument)
            CuPointwiseDiv<<<1024, 256>>>((float *) stem_paras->dev_tmp_ifftshift_4Dcell_array, (float *) stem_paras->dev_fourier_inten, dims[1]*dims[2]); 
            
            // --------------------------------------------------------  

        }
    }

}


