% Author: J.Lee, KAIST (Korea), 2022.
% Y.Yang, Multi-Dimensional Atomic Imaging Lab

function [STEM_data] = STEP00_INIT(STEM_data)

    %%% parameter initialize
    STEM_data.rec = single(STEM_data.rec);
    STEM_data.error = zeros(1,size(STEM_data.tilt_angles,1));
    
    if STEM_data.use_gpu == 1
         STEM_data.error  = gpuArray(STEM_data.error);
         %STEM_data.tilt_series = gpuArray(STEM_data.tilt_series);
    end

    % check: folder location
    if ~isfolder(STEM_data.output_filepath)
        mkdir(STEM_data.output_filepath);
        fprintf('mkdir: %s \n',STEM_data.output_filepath);
    end
    
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
    
    STEM_data.mean_intensity_list = zeros(1,size(STEM_data.tilt_angles,1));
    for i1 = 1:size(STEM_data.tilt_angles,1)
        for x1 = 1:size(STEM_data.raw_5ddata{i1},1)
            for y1 = 1:size(STEM_data.raw_5ddata{i1},2)
                STEM_data.mean_intensity_list(i1) = STEM_data.mean_intensity_list(i1) ...
                    + sum(STEM_data.raw_5ddata{i1}{x1,y1},[1,2])/size(STEM_data.raw_5ddata{i1},1)/size(STEM_data.raw_5ddata{i1},2);
            end
        end 
    end


    % check: method
    if ~any(ismember(fields(STEM_data),'method'))
        STEM_data.method = 0;
    end
    if ~any(ismember(fields(STEM_data),'use_gpu'))
        STEM_data.use_gpu = 0;
    end

    % check: metadata
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

    % check: probe wavefunction parameter
    if ~any(ismember(fields(STEM_data),'alpha'))
        error('input error: put forming aperture value into "alpha".');
    end
    if ~any(ismember(fields(STEM_data),'probeDefocus'))
        error('input error: put probeDefocus value into "probeDefocus".');
    end
    if ~any(ismember(fields(STEM_data),'C3'))
        STEM_data.C3 = 0;
    end
    if ~any(ismember(fields(STEM_data),'C5'))
        STEM_data.C5 = 0;
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
        STEM_data.use_positivity = 1;
    end
    if ~any(ismember(fields(STEM_data),'use_TV'))
        STEM_data.use_TV = 0;
    end
    if ~any(ismember(fields(STEM_data),'TV_lambda'))
        STEM_data.TV_lambda = 0.005;
    end

    %%% Generate STEP01 variable
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
        STEM_data.N_scan_x = fix(size(STEM_data.rec,1)/(STEM_data.probe_step_size/STEM_data.potential_pixelsize));
        STEM_data.N_scan_y = fix(size(STEM_data.rec,2)/(STEM_data.probe_step_size/STEM_data.potential_pixelsize));
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


    %%% Generate STEP02 variable
    % Probe wave function parameter initialize
    if size(STEM_data.probeDefocus,2) == 1
        STEM_data.probeDefocus = ones(size(STEM_data.tilt_angles,1)).* STEM_data.probeDefocus;
        probeflag = 1;
    elseif size(STEM_data.probeDefocus,2) ~= size(STEM_data.tilt_angles,1)
        error('input error: make sizes of probeDefocus and tilt angle be same');
    else
        probeflag = 0;
    end
    if size(STEM_data.C3,2) == 1
        STEM_data.C3 = ones(size(STEM_data.tilt_angles,1)).* STEM_data.C3;
        probeflag = probeflag*1;
    elseif size(STEM_data.C3,2) ~= size(STEM_data.tilt_angles,1)
        error('input error: make sizes of C3 aberration and tilt angle be same');
    else
        probeflag = probeflag*0;
    end
    if size(STEM_data.C5,2) == 1
        STEM_data.C5 = ones(size(STEM_data.tilt_angles,1)).* STEM_data.C5;
        probeflag = probeflag*1;
    elseif size(STEM_data.C5,2) ~= size(STEM_data.tilt_angles,1)
        error('input error: make sizes of C5 aberration and tilt angle be same');
    else
        probeflag = probeflag*0;    
    end

    % Calculate electron wavelength and electron interaction parameter
    STEM_data.alphaBeamMax = STEM_data.alpha/1000; % in rads, for specifying maximum angle
    STEM_data.lambda = electron_wavelength(STEM_data.E0*1000); % Electronwavelength in A
    STEM_data.sigma = interaction_constant(STEM_data.E0*1000); % interaction constant rad/Volt/Angstrom

    % Make the Fourier grid
    kx = 1:STEM_data.pot_size(1);
    ky = 1:STEM_data.pot_size(2);

    MultF_X = 1/(length(kx)*STEM_data.potential_pixelsize);
    MultF_Y = 1/(length(ky)*STEM_data.potential_pixelsize);

    CentPos = round((STEM_data.pot_size+1)/2);
    STEM_data.CentPos = CentPos;
    
    [STEM_data.qxa, STEM_data.qya] = ndgrid((kx-CentPos(1))*MultF_X,(ky-CentPos(2))*MultF_Y);
    q2 = STEM_data.qxa.^2 + STEM_data.qya.^2;

    % propagators and mask
    STEM_data.qMax = min(max(abs(STEM_data.qxa)),max(abs(STEM_data.qya)))/2;
    
    qMask = false(STEM_data.pot_size);
    qMask(CentPos(1)-round(STEM_data.pot_size(1)/4)+1:CentPos(1)+round(STEM_data.pot_size(1)/4),...
        CentPos(1)-round(STEM_data.pot_size(1)/4)+1:CentPos(1)+round(STEM_data.pot_size(1)/4)) = true;
    STEM_data.qMask = qMask;
    
    % Generate Fresnel freespace propagator & back propagator
    STEM_data.prop = single(exp((-1i*pi*STEM_data.lambda*(STEM_data.potential_pixelsize*STEM_data.slice_binning))*q2));
    STEM_data.back_prop = single(exp((+1i*pi*STEM_data.lambda*(STEM_data.potential_pixelsize*STEM_data.slice_binning))*q2));
    STEM_data.prop2D = single(STEM_data.prop);
    
    if STEM_data.use_gpu == 1
         STEM_data.prop = gpuArray(STEM_data.prop);
         STEM_data.back_prop = gpuArray(STEM_data.back_prop);
    end
    
    % make beam mask
    beam_mask = q2 < (STEM_data.alphaBeamMax/STEM_data.lambda)^2;

    % Count number of beams
    STEM_data.numberBeams = sum(beam_mask(:));
    STEM_data.beams = zeros(STEM_data.pot_size);
    STEM_data.beams(beam_mask==true) = 1:STEM_data.numberBeams;
    STEM_data.beamsIndex = find(STEM_data.beams);    
    
    % set up probe scan grid 
    pot_size = STEM_data.pot_size;
    [xx,yy]  = ndgrid([1:pot_size(1)]-round((pot_size(1)+1)/2),...
                [1:pot_size(2)]-round((pot_size(2)+1)/2));
    
    xx = reshape(xx,[1,prod(pot_size)]);
    yy = reshape(yy,[1,prod(pot_size)]);
    
    % Generate defocus factor + aberration
    STEM_data.probeC1 = -1 .* STEM_data.probeDefocus;
    for p = 1:size(STEM_data.tilt_angles,1)
        if (probeflag == 0) || (probeflag == 1 && p == 1)
            chi = (pi*STEM_data.lambda*STEM_data.probeC1(p))*...
                (STEM_data.qxa(STEM_data.beamsIndex).^2+STEM_data.qya(STEM_data.beamsIndex).^2)...
                + (pi/2*STEM_data.lambda^3*STEM_data.C3(p))*...
                (STEM_data.qxa(STEM_data.beamsIndex).^2+STEM_data.qya(STEM_data.beamsIndex).^2).^2 ...
                + (pi/3*STEM_data.lambda^5*STEM_data.C5(p))*...
                (STEM_data.qxa(STEM_data.beamsIndex).^2+STEM_data.qya(STEM_data.beamsIndex).^2).^4;
            
            % Generate probe wave function
            if STEM_data.numberBeams > 100
                Memory_saving_N = 20;
            else
                Memory_saving_N = 1;
            end
            Nstep = fix(STEM_data.numberBeams/Memory_saving_N); 
            probe_wf =zeros(size(xx));
            for i = 1:Memory_saving_N
                if i<=Memory_saving_N-1
                    probe_wf = probe_wf + sum(exp(-1i*chi(Nstep*(i-1)+1:Nstep*i) ...
                                    -2i*pi ...
                                   *(STEM_data.qxa(STEM_data.beamsIndex(Nstep*(i-1)+1:Nstep*i)).*xx*STEM_data.potential_pixelsize ...
                                   + STEM_data.qya(STEM_data.beamsIndex(Nstep*(i-1)+1:Nstep*i)).*yy*STEM_data.potential_pixelsize))...
                                   ,1);
                else
                    probe_wf = probe_wf + sum(exp(-1i*chi(Nstep*(i-1)+1:end) ...
                                    -2i*pi ...
                                   *(STEM_data.qxa(STEM_data.beamsIndex(Nstep*(i-1)+1:end)).*xx*STEM_data.potential_pixelsize ...
                                   + STEM_data.qya(STEM_data.beamsIndex(Nstep*(i-1)+1:end)).*yy*STEM_data.potential_pixelsize))...
                                   ,1);
                end
        
            end

            probe_wf = reshape(probe_wf,[pot_size(1),pot_size(2)]);   
            probe_wf = sqrt(STEM_data.mean_intensity_list(p)/sum(abs(fft2(probe_wf)).^2,[1 2])) ...
                * probe_wf; %normalize
            
        end
        
        STEM_data.probe_wfn(:,:,p) = single(probe_wf);
    end

    if STEM_data.use_gpu == 1
         STEM_data.probe_wfn = gpuArray(STEM_data.probe_wfn);
    end
    
    %%%
    fprintf('Initialize... done.\n');

end