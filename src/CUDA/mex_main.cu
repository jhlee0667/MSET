#include <iostream>
#include <atomic>
#include <vector>
#include <map>
#include <cmath>
#include "mex.h"

#include "mex_tools.hpp"
#include "mset.cuh"

using namespace std;

static std::shared_ptr<mset::paras> stem_paras;

void upload(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void run(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], int opti);
void error(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], int opti);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

    std::string Run_mode = "";
    Run_mode = get_MxString(prhs[0]);

	// Choose mode
	if (Run_mode == std::string("upload")) { 
		upload(nlhs, plhs, nrhs, prhs);}
    else if (Run_mode == std::string("run_mset")){	
		run(nlhs, plhs, nrhs, prhs, 1);}
    else if (Run_mode == std::string("run_sset")){	
        run(nlhs, plhs, nrhs, prhs, 2);}
    else if (Run_mode == std::string("error_mset")){	
        error(nlhs, plhs, nrhs, prhs, 1);}
    else if (Run_mode == std::string("error_sset")){	
        error(nlhs, plhs, nrhs, prhs, 2);}
    else {
        mexErrMsgTxt("The running mode is not correct (choose 'upload' or 'run_mset' or 'run_sset' or 'error_mset' or 'error_sset')");}

    //std::cout<< Run_mode << std::endl;
}


void upload(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

    std::vector<const mxArray*> STEM_data(prhs+1, prhs + nrhs);

    /* get a struct data (second argument) */
    // --------------------------------------------------------
    std::map<std::string, const mxArray*> map_input;
    MxStruct2vector(STEM_data[0], map_input);

    // get dimensions of init. 3D volume
    std::array<size_t,3> dims;
    get_Struct_arraysizeInfo(map_input,"RVol", dims);
    
    // create class 
    stem_paras = std::make_shared<mset::paras>(dims);


    // get Mxdata (array)
    get_Mxdata(map_input, "RVol", stem_paras->RVol, dims[0]*dims[1]*dims[2]);
    get_Mxdata(map_input, "init_wave2D", stem_paras->init_wave2D, dims[1]*dims[2]); 
    get_Mxdata(map_input, "prop2D", stem_paras->prop2D, dims[1]*dims[2]); 

    // get Mxdata (parameters)
    get_Mxdata(map_input, "alpha", stem_paras->alpha); 
    get_Mxdata(map_input, "sigma", stem_paras->sigma);
    get_Mxdata(map_input, "E0", stem_paras->E0);
    get_Mxdata(map_input, "potential_pixelsize", stem_paras->potential_pixelsize);
    get_Mxdata(map_input, "N_scan_x", stem_paras->N_scan_x);
    get_Mxdata(map_input, "N_scan_y", stem_paras->N_scan_y);
    get_Mxdata(map_input, "scan_xlist", stem_paras->scan_xlist);
    get_Mxdata(map_input, "scan_ylist", stem_paras->scan_ylist);
    get_Mxdata(map_input, "num_scan_pos", stem_paras->num_scan_pos);
    get_Mxdata(map_input, "step_size", stem_paras->step_size);
    get_Mxdata(map_input, "slice_binning", stem_paras->slice_binning);

    // send CPU data to GPU data
    cudaMemcpy(stem_paras->dev_RVol, stem_paras->RVol, sizeof(float)*dims[0]*dims[1]*dims[2], cudaMemcpyHostToDevice);
    cudaMemcpy(stem_paras->dev_init_wave2D, stem_paras->init_wave2D, sizeof(float2)*dims[1]*dims[2], cudaMemcpyHostToDevice);
    circular_shift2D((float2 *) stem_paras->prop2D, (float2 *) stem_paras->ifftshift_prop2D, (int) dims[1], (int) dims[2],(int) ceil(((float)dims[1])/2.0f),(int) ceil(((float)dims[2])/2.0f));
    cudaMemcpy(stem_paras->dev_ifftshift_prop2D, stem_paras->ifftshift_prop2D, sizeof(float2)*dims[1]*dims[2], cudaMemcpyHostToDevice);
    cudaMemcpy(stem_paras->dev_ifftshift_back_prop2D, stem_paras->dev_ifftshift_prop2D, sizeof(float2)*dims[1]*dims[2], cudaMemcpyDeviceToDevice);
    CuPointwiseConj<<<256, 256>>>((cufftComplex *) stem_paras->dev_ifftshift_back_prop2D, dims[1]*dims[2]);
    // --------------------------------------------------------

    /* Get measured full 4D-STEM data */
    stem_paras->pMxCell_4DSTEM = STEM_data[1];
    stem_paras->total_num_of_4DCell = mxGetNumberOfElements(stem_paras->pMxCell_4DSTEM);
    stem_paras->cell_dims = mxGetDimensions(stem_paras->pMxCell_4DSTEM);
    // --------------------------------------------------------
   
    // Check variables for debugging
    //stem_paras->print();

}


void run(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], int opti){

    /* main calculation */
    if (opti == 1){
        mset::run_main_mset(stem_paras);}
    else if (opti == 2){
        mset::run_main_sset(stem_paras);}


    /* extract reconstructed 3D volume */
    float *output_rec, *output_scan_xlist, *output_scan_ylist, *output_error;
    mxComplexSingle *output_probe;
    unsigned long output_rec_dims[3] = {stem_paras->rec_size[2], stem_paras->rec_size[1], stem_paras->rec_size[0]}; 
    unsigned long output_probe_dims[2] = {stem_paras->rec_size[2], stem_paras->rec_size[1]};
    unsigned long output_scanpos_dims[1] = {(unsigned long) stem_paras->num_scan_pos[0]}; 
    unsigned long output_error_dims[1] = {1}; 
    cudaMemcpy(stem_paras->RVol, stem_paras->dev_RVol, sizeof(float)*output_rec_dims[0]*output_rec_dims[1]*output_rec_dims[2], cudaMemcpyDeviceToHost); 
    cudaMemcpy(stem_paras->init_wave2D, stem_paras->dev_init_wave2D, sizeof(float2)*output_probe_dims[0]*output_probe_dims[1], cudaMemcpyDeviceToHost); 
    cudaMemcpy(stem_paras->error_array, stem_paras->dev_error_array, sizeof(float)*output_rec_dims[0]*output_rec_dims[1], cudaMemcpyDeviceToHost); 
    
    // object output
    plhs[0] = mxCreateNumericArray(3, output_rec_dims, mxSINGLE_CLASS, mxREAL);
    output_rec = mxGetSingles(plhs[0]);
    for (int i = 0; i < stem_paras->rec_size[0]*stem_paras->rec_size[1]*stem_paras->rec_size[2]; ++i){ 
        output_rec[i] = stem_paras->RVol[i];
    }
    // probe wave output
    plhs[1] = mxCreateNumericArray(2, output_probe_dims, mxSINGLE_CLASS, mxCOMPLEX);
    output_probe = mxGetComplexSingles(plhs[1]);
    for (int i = 0; i < stem_paras->rec_size[1]*stem_paras->rec_size[2]; ++i){ 
        output_probe[i].real = stem_paras->init_wave2D[i].x;
        output_probe[i].imag = stem_paras->init_wave2D[i].y;
    }
    // scan position output
    plhs[2] = mxCreateNumericArray(1, output_scanpos_dims, mxSINGLE_CLASS, mxREAL);
    plhs[3] = mxCreateNumericArray(1, output_scanpos_dims, mxSINGLE_CLASS, mxREAL); 
    output_scan_xlist = mxGetSingles(plhs[2]);
    output_scan_ylist = mxGetSingles(plhs[3]);
    for (int i = 0; i < stem_paras->num_scan_pos[0]; ++i){ 
        output_scan_xlist[i] = stem_paras->scan_xlist[i];
        output_scan_ylist[i] = stem_paras->scan_ylist[i];
    }

    // error output
    if (nlhs == 5) {
        plhs[4] = mxCreateNumericArray(1, output_error_dims, mxSINGLE_CLASS, mxREAL);
        output_error = mxGetSingles(plhs[4]);

        output_error[0] = 0.0f;
        for (int i = 0; i < stem_paras->rec_size[1]*stem_paras->rec_size[2]; ++i){ 
            output_error[0] += stem_paras->error_array[i];
        }
        output_error[0] = output_error[0] / ((float) stem_paras->N_scan_x[0]* (float) stem_paras->N_scan_y[0])/ ((float) stem_paras->rec_size[1]* (float) stem_paras->rec_size[2]);
    }
    // ---------------------------------------------------------


}

void error(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], int opti){

    /* error calculation */
    if (opti == 1){
        mset::run_error_mset(stem_paras);}
    else if (opti == 2){
        mset::run_error_sset(stem_paras);} 

    /* extract reconstructed 3D volume */
    float *output_error;
    unsigned long output_dims[3] = {stem_paras->rec_size[2], stem_paras->rec_size[1], stem_paras->rec_size[0]}; 
    unsigned long output_error_dims[1] = {1};
    cudaMemcpy(stem_paras->error_array, stem_paras->dev_error_array, sizeof(float)*output_dims[0]*output_dims[1], cudaMemcpyDeviceToHost); 

  
    plhs[0] = mxCreateNumericArray(1, output_error_dims, mxSINGLE_CLASS,  mxREAL);
    output_error = mxGetSingles(plhs[0]);

    output_error[0] = 0.0f;
    for (int i = 0; i < stem_paras->rec_size[1]*stem_paras->rec_size[2]; ++i){ 
        output_error[0] += stem_paras->error_array[i];
    }
    output_error[0] = output_error[0] / ((float) stem_paras->N_scan_x[0]* (float) stem_paras->N_scan_y[0])/ ((float) stem_paras->rec_size[1]* (float) stem_paras->rec_size[2]);
    
    // ---------------------------------------------------------


}