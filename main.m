%%
% Author: J.Lee, KAIST (Korea), 2022.
% Y.Yang, Multi-Dimensional Atomic Imaging Lab
% 4D-STEM based multislice electron tomography (MSET/SSET main code)
% phase retrival reconstruction method 

%%
clear all
clc

%% input parameters
%%% input data %%%
STEM_data.input_filepath = './examples/CuAu_24size_dataset'; % 4D-STEM input data folder path
STEM_data.input_filename_list = ["CuAu_s24_p72_Np_8_4DSTEM_data_1", ...
                                 "CuAu_s24_p72_Np_8_4DSTEM_data_2", ...
                                 "CuAu_s24_p72_Np_8_4DSTEM_data_3", ...
                                 "CuAu_s24_p72_Np_8_4DSTEM_data_4", ...
                                 "CuAu_s24_p72_Np_8_4DSTEM_data_5", ...
                                 "CuAu_s24_p72_Np_8_4DSTEM_data_6", ...
                                 "CuAu_s24_p72_Np_8_4DSTEM_data_7", ...
                                 "CuAu_s24_p72_Np_8_4DSTEM_data_8"]; % 4D-STEM input data file name list
STEM_data.output_filepath = STEM_data.input_filepath; % output (reconstruction file) folder path
STEM_data.output_filename = 'MSET_recon'; % output (reconstruction file) file name
STEM_data.tilt_angles = importdata('./examples/CuAu_24size_dataset/tilt_angles.mat'); % tilt angles (Nx3 array) (deg)
STEM_data.vec1 = [0 0 1]; % first angles rotation direction
STEM_data.vec2 = [0 1 0]; % second angles rotation direction
STEM_data.vec3 = [1 0 0]; % third angles rotation direction
STEM_data.scan_pos = importdata('./examples/CuAu_24size_dataset/scan_pos.mat'); % probe scan position (Nx2 array) (pixels)
%STEM_data.probe_step_size = 0.4; % scan step size (A)

%%% reconstruction conditions %%%
STEM_data.method = 0; % 0 for MSET, 1 for SSET
STEM_data.alpha = 21; % Probe forming aperture (mrad)
STEM_data.E0 = 300; % Probe accelerating voltage (kV)
STEM_data.potential_pixelsize = 0.4;% potential resolution (A);
STEM_data.rec = zeros(24,24,24,'single'); % initial volume for 3D reconstruction
STEM_data.slice_binning =  1; % slice-binning along beam direction
STEM_data.probeDefocus = -200; % C1 defocus (A)
STEM_data.C3 = 0; % C3 aberration (A) 
STEM_data.C5 = 0; % C5 aberration (A) 
STEM_data.use_gpu = 0; % computation modes (0: cpu, 1:matlab gpu, 2: cuda)
STEM_data.step_size = [1*10^(2), 0, 0]; % step size 1x3 input, [objection rec. step size, probe shape optimization step size, scan position step size]
STEM_data.N_iter = 5; % Number of iterations

%%% regularization parameters %%%
STEM_data.bls_parameter = 0.1; % backtracking line search parameter
STEM_data.use_positivity = 1; % positivity on(1)/off(0)
STEM_data.use_TV = 0; % TV regularization on(1)/off(0)
STEM_data.TV_lambda = 0.005; % TV lambda parameter


%% Run
[STEM_data] = Run_MSET_recon(STEM_data);
