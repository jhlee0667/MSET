% Author: J.Lee, KAIST (Korea), 2022.
% Y.Yang, Multi-Dimensional Atomic Imaging Lab

function [STEM_data] = STEP00_INIT(STEM_data)

    %%% parameter initialize %%%
   
    % check: input data (tilt angles)
    if ~any(ismember(fields(STEM_data),'tilt_angles'))
        error('input error: put tilt angles into "tilt_angles".');
        if size(STEM_data.tilt_angles,2) ~= 3
            error('input error: put proper format of tilt_angles (Number of tilt angles x 3).');
        end
    end

    % check & load: input data (a tilt series of 4D-STEM data)
    STEM_data.raw_5ddata = {};
    if any(ismember(fields(STEM_data),'input_filename_list'))
        STEM_data.input_filename_list = string(STEM_data.input_filename_list);
        if length(STEM_data.input_filename_list) ~= size(STEM_data.tilt_angles,1)
            error('input error: put the same number of 4D-STEM files with the number of tilt angles.');
        end

        for i1 = 1:size(STEM_data.tilt_angles,1)
            if ~isfile(sprintf('%s/%s.mat',STEM_data.input_filepath,STEM_data.input_filename_list(i1)))
                error('input error: no files (4D-STEM files)');
            else
                STEM_data.raw_5ddata{i1} = importdata(sprintf("%s/%s.mat",STEM_data.input_filepath,STEM_data.input_filename_list(i1)));
                STEM_data.raw_5ddata{i1} = cellfun(@single,STEM_data.raw_5ddata{i1},'un',0);
            end
        end

        STEM_data.input_filename = [];
    elseif any(ismember(fields(STEM_data),'input_filename'))
        for i1 = 1:size(STEM_data.tilt_angles,1)
            if ~isfile(sprintf('%s/%s_%d.mat',STEM_data.input_filepath,STEM_data.input_filename,i1))
                error('input error: no files (4D-STEM files)');
            else
                STEM_data.raw_5ddata{i1} = importdata(sprintf("%s/%s_%d.mat",STEM_data.input_filepath,STEM_data.input_filename,i1));
                STEM_data.raw_5ddata{i1} = cellfun(@single,STEM_data.raw_5ddata{i1},'un',0);
            end
        end
        STEM_data.input_filename_list = [];
    else
        error('input error: put proper PATH information of 4D-STEM files.');
    end
    

    % raw 4d stem data, rotation
    if ~any(ismember(fields(STEM_data),'diffraction_rotation'))
        STEM_data.diffraction_rotation = 0;
    end

    Rot = [cosd(STEM_data.diffraction_rotation), -sind(STEM_data.diffraction_rotation);
           sind(STEM_data.diffraction_rotation), cosd(STEM_data.diffraction_rotation)];

    % raw 4d stem data, transpose
    if ~any(ismember(fields(STEM_data),'diffraction_transpose'))
        STEM_data.diffraction_transpose = false;
    end
    
    % calculate mean diffraction intensity & rotation & transpose
    STEM_data.mean_intensity_list = zeros(1,size(STEM_data.tilt_angles,1));

    for i1 = 1:size(STEM_data.tilt_angles,1)
        for x1 = 1:size(STEM_data.raw_5ddata{i1},1)
            for y1 = 1:size(STEM_data.raw_5ddata{i1},2)
                if STEM_data.diffraction_rotation ~= 0
                    STEM_data.raw_5ddata{i1}{x1,y1} = single(Func_inv_rotating_2D(STEM_data.raw_5ddata{i1}{x1,y1}, Rot));
                end 
                if STEM_data.diffraction_transpose == true
                    STEM_data.raw_5ddata{i1}{x1,y1} = single(transpose(STEM_data.raw_5ddata{i1}{x1,y1}));
                end

                STEM_data.mean_intensity_list(i1) = STEM_data.mean_intensity_list(i1)+...
                    sum(STEM_data.raw_5ddata{i1}{x1,y1},[1,2])/size(STEM_data.raw_5ddata{i1},1)/size(STEM_data.raw_5ddata{i1},2);
            end
        end
    end

    % check: method
    if ~any(ismember(fields(STEM_data),'method'))
        STEM_data.method = "MSET";
    end
    if ~any(ismember(fields(STEM_data),'device'))
        STEM_data.device = 0;
    end

    % check: save
    if ~any(ismember(fields(STEM_data),'store_iterations'))
        STEM_data.store_iterations = false;
    end

    % check: metadata
    if ~any(ismember(fields(STEM_data),'E0'))
        error('input error: put acceleration voltage value (kV) into "E0".');
    end
    % Calculate electron wavelength and electron interaction parameter
    STEM_data.lambda = electron_wavelength(STEM_data.E0*1000); % Electronwavelength in A
    STEM_data.sigma = interaction_constant(STEM_data.E0*1000); % interaction constant rad/Volt/Angstrom

    if ~any(ismember(fields(STEM_data),'potential_pixelsize'))
        error('input error: put potential_pixelsize value into "potential_pixelsize".');
    end
    if ~any(ismember(fields(STEM_data),'N_iter'))
        error('input error: put number of iterations value into "N_iter".');
    end

    % check: rotation angle direction
    if ~any(ismember(fields(STEM_data),'vec1'))
        STEM_data.vec1 = [0 0 1];
    end
    if ~any(ismember(fields(STEM_data),'vec2'))
        STEM_data.vec2 = [0 1 0];
    end    
    if ~any(ismember(fields(STEM_data),'vec3'))
        STEM_data.vec3 = [1 0 0];
    end

    % check: slice binning parameter
    if ~any(ismember(fields(STEM_data),'slice_binning'))
        STEM_data.slice_binning = 1;
    end
    if mod(STEM_data.slice_binning,1) ~= 0
        error('input error: put a positive integer value into "slice_binning".');
    end
    if STEM_data.slice_binning > size(STEM_data.rec,3)
        STEM_data.slice_binning = size(STEM_data.rec,3);
    end

    % check: step size parameter
    if ~any(ismember(fields(STEM_data),'step_size'))
        error('input error: put step size into "step_size".');
    else
        if length(STEM_data.step_size) == 1
            STEM_data.step_size = [STEM_data.step_size, 0, 0];
        elseif size(STEM_data.step_size,2) == 3
        else
            error('input error: put properly format (1x3) of step size into "step_size".');
        end
    end

    % check: regularization parameter
    if ~any(ismember(fields(STEM_data),'bls_parameter'))
        STEM_data.bls_parameter = 0.1;
    end
    if ~any(ismember(fields(STEM_data),'use_positivity'))
        STEM_data.use_positivity = true;
    end
    if ~any(ismember(fields(STEM_data),'use_TV'))
        STEM_data.use_TV = false;
    end
    if ~any(ismember(fields(STEM_data),'TV_lambda'))
        STEM_data.TV_lambda = 0.005;
    end



    % Calculate scan positions
    if any(ismember(fields(STEM_data),'scan_pos'))
        if size(STEM_data.scan_pos,3) ~= size(STEM_data.tilt_angles,1) && size(STEM_data.scan_pos,3) == 1 
            STEM_data.scan_pos = repmat(STEM_data.scan_pos,1,1,size(STEM_data.tilt_angles,1));
        elseif size(STEM_data.scan_pos,3) == size(STEM_data.tilt_angles,1)
        else
            error('input error: match array size of "scan positions" (#scanposition, 2, #oftiltangle).');
        end
        STEM_data.scan_pos = single(STEM_data.scan_pos);
    elseif any(ismember(fields(STEM_data),'probe_step_size'))
        STEM_data.N_scan_x = size(STEM_data.raw_5ddata{1},1);
        STEM_data.N_scan_y = size(STEM_data.raw_5ddata{1},2);
        STEM_data.scan_size = [STEM_data.N_scan_x STEM_data.N_scan_y]; % scan size
        STEM_data.scan_pos = [];
        for y1 = 1:STEM_data.N_scan_y
            for x1 = 1:STEM_data.N_scan_x
                STEM_data.scan_pos = [STEM_data.scan_pos; x1, y1];
            end
        end
        STEM_data.scan_pos = single((STEM_data.scan_pos-1)*(STEM_data.probe_step_size/STEM_data.potential_pixelsize)+1);
        STEM_data.scan_pos = repmat(STEM_data.scan_pos,1,1,size(STEM_data.tilt_angles,1));
    else
        error('input error: put values into "probe_step_size" or "scan positions".');
    end

    STEM_data.num_scan_pos = single(size(STEM_data.scan_pos,1));
    STEM_data.numPlanes = ceil(size(STEM_data.rec,3)/STEM_data.slice_binning); % z size of pot3D
    STEM_data.pot_size = [size(STEM_data.rec,1) size(STEM_data.rec,2)]; % potential size (2D)


    % Calculate electron wavelength and electron interaction parameter
    STEM_data.lambda = electron_wavelength(STEM_data.E0*1000); % Electronwavelength in A
    STEM_data.sigma = interaction_constant(STEM_data.E0*1000); % interaction constant rad/Volt/Angstrom

    
    % Generate Fresnel freespace propagator & back propagator
    [STEM_data] = Func_generate_propagator(STEM_data);


    % Load or Generate probe wave function
    if any(ismember(fields(STEM_data),'probe_wave'))
        if size(STEM_data.probe_wave,3) ~= size(STEM_data.tilt_angles,1)
            if size(STEM_data.probe_wave,3) == 1
                for i1 = 1:size(STEM_data.tilt_angles,1)
                    STEM_data.probe_wave(:,:,i1) = STEM_data.probe_wave(:,:,1);
                end
            else
                error('input error: make 3rd dimension of probe_wave and # of tilt angle same');
            end
        end
    else
        % Initialize probe wave function 
        [STEM_data] = Func_generate_probe_wave(STEM_data);
    end

    % normalize probe wave function
    for p = 1:size(STEM_data.tilt_angles,1)
        STEM_data.probe_wave(:,:,p) = sqrt(STEM_data.mean_intensity_list(p)/sum(abs(fft2(STEM_data.probe_wave(:,:,p))).^2,[1 2])) ...
                   * STEM_data.probe_wave(:,:,p); 
    end

    % make error list
    STEM_data.error = zeros(1,size(STEM_data.tilt_angles,1));
    
    % make sure that rec must have single precision
    STEM_data.rec = single(STEM_data.rec);    

    %%% RAM array -> MATLAB VRAM
    if STEM_data.device == 1
        STEM_data.error  = gpuArray(STEM_data.error);
        STEM_data.prop = gpuArray(STEM_data.prop);
        STEM_data.back_prop = gpuArray(STEM_data.back_prop);
        STEM_data.probe_wave = gpuArray(STEM_data.probe_wave);
    end
    
    %%%
    fprintf('Initialize... done.\n\n');

end