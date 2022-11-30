#ifndef MSET_CUH
#define MSET_CUH

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
#include "mex.h"
#include "utility.cuh"


using namespace std;

namespace mset{
    class paras{

    private:

    public:
        //mset_paras(const mset_paras&) = delete;
        //void operator=(const mset_paras&) = delete;
        
        /* Declare dynamic arrays. (for 3D, 2D data) */
        std::array<size_t,3> rec_size;
        float* RVol;
        float2* init_wave2D;
        float2* prop2D;
        float2* ifftshift_prop2D;
        float* error_array;

        float* dev_RVol;
        float2* dev_trans_fun;
        float2* dev_conj_trans_fun;
        float2* dev_ifftshift_prop2D;
        float2* dev_ifftshift_back_prop2D;
        float2* dev_init_wave2D;
        float2* dev_fsfactor;
        float2* dev_shift_fsfactor;
        float2* dev_wave2D;
        float2* dev_save_wave3D;
        float2* dev_grad_complex;
        float* dev_grad;
        float2* dev_grad2d;
        float2* dev_tmp;
        
        float* dev_error_array;
        float* dev_tmp_error_array;

        float* dev_tmp_4Dcell_array;
        float* dev_tmp_ifftshift_4Dcell_array;
        float* dev_fourier_inten;

        float2* dev_residual_vector;

        // constructor
        paras(std::array<size_t,3> rec_size);
        // destructor
        ~paras();

        /* Declare variables. (STEM_data) */
        std::vector<float> alpha;
        std::vector<float> sigma;
        std::vector<float> inner_angle;
        std::vector<float> outer_angle;
        std::vector<float> E0;
        std::vector<float> probe_step_size;
        std::vector<float> potential_pixelsize;
        std::vector<float> N_scan_x;
        std::vector<float> N_scan_y;
        std::vector<float> scan_xlist;
        std::vector<float> scan_ylist;
        std::vector<float> num_scan_pos;
        std::vector<float> step_size;
        std::vector<float> slice_binning;
        

        /* Declare variables. (Measured 4D full STEM_data) */
        const mxArray *pMxCell_4DSTEM;
        const mxArray *pMxCell_4DSTEM_element;
        size_t total_num_of_4DCell;
        const unsigned long * cell_dims;
        float* tmp_4Dcell_array;
        float* tmp_ifftshift_4Dcell_array;

        // for checking
        void print(){
            std::cout << "alpha: " << alpha[0] << std::endl;
            std::cout << "sigma: " << sigma[0] << std::endl;
            //std::cout << "inner_angle: " << inner_angle[0] << std::endl;
            //std::cout << "outer_angle: " << outer_angle[0] << std::endl;
            std::cout << "E0: " << E0[0] << std::endl;
            std::cout << "probe_step_size: " << probe_step_size[0] << std::endl;
            std::cout << "potential_pixelsize: " << potential_pixelsize[0] << std::endl;
            std::cout << "rec_size1(z): " << rec_size[0] << "  rec_size2(y): " << rec_size[1] << "  rec_size3(x): " << rec_size[2] <<std::endl;
            std::cout << "step_size: " << step_size[0] <<std::endl;
            
            std::cout << "total num of cells: " << total_num_of_4DCell <<std::endl;
            std::cout << "Dimension of cells: (" << cell_dims[0] << ", " << cell_dims[1] << ")" <<std::endl;  

            std::cout << "number of scan positions:" << num_scan_pos[0] <<std::endl;  
            std::cout << "scan_pos[0]= (" << scan_xlist[0] << ", " << scan_ylist[0] << ")" <<std::endl;  
            std::cout << "scan_pos[end] = (" << scan_xlist[num_scan_pos[0]-1] << ", " << scan_ylist[num_scan_pos[0]-1] << ")" <<std::endl;
            std::cout << "slice_binning: " << slice_binning[0] <<std::endl;   

        }

    };


    
    inline namespace v0 {
       void run_main_mset(std::shared_ptr<paras> stem_data);
       void run_main_sset(std::shared_ptr<paras> stem_data);

       void run_error_mset(std::shared_ptr<paras> stem_data);
       void run_error_sset(std::shared_ptr<paras> stem_data);
    }

}


#endif