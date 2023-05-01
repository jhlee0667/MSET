%%
% Author: J.Lee, KAIST (Korea), 2022.
% Y.Yang, Multi-Dimensional Atomic Imaging Lab
% Multislice electron tomography

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% input parameters  %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% input data %%%
% STEM_data.input_filepath: 4D-STEM input data folder path
% STEM_data.input_filename: 4D-STEM input data file name
% STEM_data.output_filepath: output (reconstruction file) folder path
% STEM_data.output_filename: output (reconstruction file) file name
% STEM_data.tilt_angles: tilt angles (Nx3 array) (deg)
% STEM_data.scan_pos: probe scan position (Nx2 array) (A)
% STEM_data.probe_step_size: scan step size (A) (if having specific scan position, this value is ignored. if you have any specific probe scan positions, this option creates scan position grid)
% STEM_data.vec1 = [0 0 1]; first angles rotation direction
% STEM_data.vec2 = [0 1 0]; second angles rotation direction
% STEM_data.vec3 = [1 0 0]; third angles rotation direction

%%% reconstruction conditions %%%
% STEM_data.method: 0 for MSET, 1 for SSET
% STEM_data.alpha: Probe forming aperture (mrad)
% STEM_data.E0: Probe accelerating voltage (kV)
% STEM_data.potential_pixelsize: potential resolution (A);
% STEM_data.rec: initial volume for 3D reconstruction
% STEM_data.slice_binning: slice-binning along beam direction
% STEM_data.probeDefocus: C1 defocus (A)
% STEM_data.C3: C3 aberration (A) 
% STEM_data.C5: C5 aberration (A) 
% STEM_data.use_gpu: computation modes (0: cpu, 1:matlab gpu, 2: cuda)
% STEM_data.step_size: step size 1x3 input, [objection rec. step size, probe shape optimization step size, scan position step size]
% STEM_data.N_iter: Number of iterations

%%% regulization parameters %%%
% STEM_data.bls_parameter: backtracking line search parameter
% STEM_data.use_positivity: positivity on(1)/off(0)
% STEM_data.use_TV: TV regularization on(1)/off(0)
% STEM_data.TV_lambda: TV lambda parameter

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% output parameters %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% STEM_data: meta data for reconstruction
% mat_save.rec: reconstruction file for each iteration 
% mat_save.error: mean error list for each iteration 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%     data form     %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 4D-STEM data: (Rx, Ry, Qx, Qy) each cell (Rx, Ry) has 2D array (Qx, Qy) 
% probe scan position: (Rx, Ry)
% 3D object reconstruction: (Qx, Qy, Qx)


function [STEM_data] = Run_MSET_recon(STEM_data)

    %%% Initialize 
    [STEM_data] = STEP00_INIT(STEM_data);

    % make folder & files
    rec_save = zeros(size(STEM_data.rec,1), size(STEM_data.rec,2), size(STEM_data.rec,3), STEM_data.N_iter);
    mean_error = zeros(1, STEM_data.N_iter);
    total_mean_error = zeros(1, STEM_data.N_iter);
    
    fprintf("output path: %s/%s.mat \n\n", STEM_data.output_filepath, STEM_data.output_filename);
    save(sprintf("%s/%s.mat",STEM_data.output_filepath,STEM_data.output_filename), "STEM_data", "rec_save", "mean_error", "total_mean_error", '-v7.3');
    mat_save = matfile(sprintf("%s/%s.mat", STEM_data.output_filepath, STEM_data.output_filename), 'Writable', true);
    clear rec_save mean_error total_mean_error
    
    N_angle = size(STEM_data.tilt_angles,1);
    STEM_data.measured_4D_data = {};
    old_mean_error = 10^31;
    step_size = STEM_data.step_size;
    
    %%% new Reconstruction
    breakflag = 0;
    tStart = tic;
    for i = 1:STEM_data.N_iter
        
        old_rec = STEM_data.rec;
        old_probe = STEM_data.probe_wfn;
        old_scan_pos = STEM_data.scan_pos;
    
        tic
        while true
            for j = 1:N_angle 
                %%% STEP01 - inverse Rotation (W=R'*U)
                STEM_data.W = [];
                STEM_data.Nth_angle = j;
                [STEM_data] = STEP01_ROTATION(STEM_data);
                
                STEM_data.measured_4D_data = importdata(sprintf("%s/%s_%d.mat",STEM_data.input_filepath,STEM_data.input_filename,j));
                STEM_data.measured_4D_data = cellfun(@single,STEM_data.measured_4D_data,'un',0);

                STEM_data.N_scan_x = size(STEM_data.measured_4D_data,1);
                STEM_data.N_scan_y = size(STEM_data.measured_4D_data,2);
                STEM_data.scan_xlist = squeeze(STEM_data.scan_pos(:,1,j));
                STEM_data.scan_ylist = squeeze(STEM_data.scan_pos(:,2,j));
                STEM_data.init_wave2D = single(STEM_data.probe_wfn(:,:,j));
    
                if STEM_data.use_gpu == 0 || STEM_data.use_gpu == 1
                    for k = 1:size(STEM_data.scan_pos,1)
                        
                        STEM_data.k = k;
                        STEM_data.row = STEM_data.scan_xlist(k);
                        STEM_data.col = STEM_data.scan_ylist(k);
    
                        if STEM_data.method == 1
                            % STEP02 - forward propagation & calculate residual vector
                            [STEM_data] = STEP02_FORWARD_4D_SSET(STEM_data);      
                
                            % STEP03 - back propagation
                            [STEM_data] = STEP03_BACK_SSET(STEM_data); 
                        else
                            % STEP02 - forward propagation & calculate residual vector
                            [STEM_data] = STEP02_FORWARD_4D(STEM_data);      
                
                            % STEP03 - back propagation
                            [STEM_data] = STEP03_BACK(STEM_data); 
                        end
    
                        % STEP04 - update
                        STEM_data.RVol = STEM_data.RVol + STEM_data.step_size(1) * real(STEM_data.grad);
                        STEM_data.scan_pos(k,1,j) = STEM_data.row;
                        STEM_data.scan_pos(k,2,j) = STEM_data.col;
                    end
                elseif STEM_data.use_gpu==2
        
                    % STEP02-04
                    mset_cuda_kernel('upload', STEM_data, STEM_data.measured_4D_data);
    
                    if STEM_data.method == 1
                        [tmp_RVol, tmp_probe, tmp_scan_xlist, tmp_scan_ylist] = mset_cuda_kernel('run_sset',STEM_data, STEM_data.measured_4D_data);               
                    else
                        [tmp_RVol, tmp_probe, tmp_scan_xlist, tmp_scan_ylist] = mset_cuda_kernel('run_mset',STEM_data, STEM_data.measured_4D_data);            
                    end
                    
                    STEM_data.RVol = tmp_RVol;
                    STEM_data.probe_wfn(:,:,STEM_data.Nth_angle) = tmp_probe;
                    STEM_data.scan_pos(:,1,j) = tmp_scan_xlist;
                    STEM_data.scan_pos(:,2,j) = tmp_scan_ylist;
                    clear mset_cuda_kernel
    
                end
    
                %%% STEP05 - Rotation & constraint
                STEM_data.rec = Func_rot_3Dvol_FourierShear(STEM_data.RVol, STEM_data.vec1, STEM_data.vec2, STEM_data.vec3,...
                                STEM_data.tilt_angles(STEM_data.Nth_angle,1), STEM_data.tilt_angles(STEM_data.Nth_angle,2), STEM_data.tilt_angles(STEM_data.Nth_angle,3));
                
                % positivity
                if STEM_data.use_positivity == 1
                    STEM_data.rec(STEM_data.rec<0)=0;
                end

            end
            
            %%% error calculation
            STEM_data.error = zeros(1,size(STEM_data.tilt_angles,1));
    
            for j = 1:N_angle 
                %%% STEP01 - inverse Rotation (W=R'*U)
                STEM_data.W = [];
                STEM_data.Nth_angle = j;
                [STEM_data] = STEP01_ROTATION(STEM_data);
        
                STEM_data.measured_4D_data = importdata(sprintf("%s/%s_%d.mat",STEM_data.input_filepath,STEM_data.input_filename,j));
                STEM_data.measured_4D_data = cellfun(@single,STEM_data.measured_4D_data,'un',0);

                STEM_data.N_scan_x = size(STEM_data.measured_4D_data,1);
                STEM_data.N_scan_y = size(STEM_data.measured_4D_data,2);

                STEM_data.scan_xlist = squeeze(STEM_data.scan_pos(:,1,j));
                STEM_data.scan_ylist = squeeze(STEM_data.scan_pos(:,2,j));
                STEM_data.init_wave2D = single(STEM_data.probe_wfn(:,:,j));

                if STEM_data.use_gpu == 0 || STEM_data.use_gpu == 1
                    for k = 1:size(STEM_data.scan_pos,1)
                        
                        STEM_data.k = k;
                        STEM_data.row = STEM_data.scan_xlist(k);
                        STEM_data.col = STEM_data.scan_ylist(k);
    
                        if STEM_data.method == 1
                            % STEP02 - forward propagation & calculate residual vector
                            [STEM_data] = STEP02_FORWARD_4D_SSET(STEM_data); 
                        else
                            % STEP02 - forward propagation & calculate residual vector
                            [STEM_data] = STEP02_FORWARD_4D(STEM_data);  
                        end        
                        
                    end
                elseif STEM_data.use_gpu == 2
        
                    % STEP02-04
                    mset_cuda_kernel('upload',STEM_data, STEM_data.measured_4D_data);
    
                    if STEM_data.method == 1
                        [tmp_error] = mset_cuda_kernel('error_sset',STEM_data, STEM_data.measured_4D_data);
                    else
                        [tmp_error] = mset_cuda_kernel('error_mset',STEM_data, STEM_data.measured_4D_data);
                    end
                    
                    STEM_data.error(STEM_data.Nth_angle) = tmp_error;
                    clear mset_cuda_kernel
                    
                end
    
            end
    
            if old_mean_error > mean(STEM_data.error)
                break;
            else
                fprintf('--//bls: mean error: %.11f, object step_size: %d -> %d \n', mean(STEM_data.error), STEM_data.step_size(1), STEM_data.step_size(1)*STEM_data.bls_parameter);
                STEM_data.rec = old_rec;
                STEM_data.probe_wfn = old_probe;
                STEM_data.scan_pos = old_scan_pos;
    
                STEM_data.step_size = STEM_data.bls_parameter .* STEM_data.step_size;
            end
            
            if STEM_data.step_size(1) < (STEM_data.bls_parameter)^5 * step_size(1)
                breakflag = 1;
                break;
            end
    
        end
        
        if STEM_data.step_size(1) < (STEM_data.bls_parameter)^5 * step_size(1)
            fprintf('break: step_size: [%d, %d, %d] \n', STEM_data.step_size(1),STEM_data.step_size(2),STEM_data.step_size(3));
            break;
        end
        toc
        
        mean_error = mean(STEM_data.error);
    
        if STEM_data.use_TV ==1
            sf_max = max(STEM_data.rec(:));
            [STEM_data.rec, TV_error] = TV_method(STEM_data.rec./sf_max, STEM_data.TV_lambda, 300);
            STEM_data.rec = sf_max .* STEM_data.rec;
    
            total_mean_error = mean_error + TV_error/N_angle/size(STEM_data.rec,1)/size(STEM_data.rec,2);
            fprintf('#%d: mean error: %.11f, total mean error: %.11f \n', i, mean_error, total_mean_error);
    
            mat_save.total_mean_error(1, i) = total_mean_error;
        else
            fprintf('#%d: mean error: %.11f \n', i, mean_error);
        end
        
        % save the reconstruction & error
        if STEM_data.N_iter == 1
            mat_save.rec_save = gather(STEM_data.rec);
        else
            mat_save.rec_save(:,:,:,i) = gather(STEM_data.rec);
        end
        mat_save.mean_error(1, i) = mean_error;
        STEM_data.errorlist(1, i) = mean_error;
        old_mean_error = mean_error;
    end
    tEnd = toc(tStart)
    
    mat_save.STEM_data = STEM_data;
    if breakflag == 1 && STEM_data.N_iter > 50
        mat_save.rec_save(:,:,:,STEM_data.N_iter-40:STEM_data.N_iter)=[];
        mat_save.mean_error(:,STEM_data.N_iter-40:STEM_data.N_iter)=[];
        mat_save.total_mean_error(:,STEM_data.N_iter-40:STEM_data.N_iter)=[];
    end

end
