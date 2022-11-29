% Author: J.Lee, KAIST (Korea), 2021.
% Y.Yang, Multi-Dimensional Atomic Imaging Lab

function [STEM_data] = STEP00_INIT(STEM_data)


    %%% STEP00
    STEM_data.error = zeros(1,size(STEM_data.tilt_angles,1));
    
    if STEM_data.use_gpu == 1
         STEM_data.error  = gpuArray(STEM_data.error);
         %STEM_data.tilt_series = gpuArray(STEM_data.tilt_series);
    end

    % parameter initialize
    if ~any(ismember(fields(STEM_data),'C3'))
        STEM_data.C3 = 0;
    end
    if ~any(ismember(fields(STEM_data),'C5'))
        STEM_data.C5 = 0;
    end    
    if ~any(ismember(fields(STEM_data),'z_bin'))
        STEM_data.z_bin = 1;
    end
    if mod(STEM_data.z_bin,1) ~= 0
        error('input error: put a positive integer value into "z_bin".');
    end
    if STEM_data.z_bin > size(STEM_data.rec,3)
        STEM_data.z_bin = size(STEM_data.rec,3);
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
    STEM_data.numPlanes = ceil(size(STEM_data.rec,3)/STEM_data.z_bin); % z size of pot3D
    STEM_data.pot_size = [size(STEM_data.rec,1) size(STEM_data.rec,2)]; % potential size (2D)


    %%% Generate STEP02 variable
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
    STEM_data.prop = single(exp((-1i*pi*STEM_data.lambda*(STEM_data.potential_pixelsize*STEM_data.z_bin))*q2));
    STEM_data.back_prop = single(exp((+1i*pi*STEM_data.lambda*(STEM_data.potential_pixelsize*STEM_data.z_bin))*q2));
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
    STEM_data.probeC1 = -1 * STEM_data.probeDefocus;
    
    chi = (pi*STEM_data.lambda*STEM_data.probeC1)*...
        (STEM_data.qxa(STEM_data.beamsIndex).^2+STEM_data.qya(STEM_data.beamsIndex).^2)...
        + (pi/2*STEM_data.lambda^3*STEM_data.C3)*...
        (STEM_data.qxa(STEM_data.beamsIndex).^2+STEM_data.qya(STEM_data.beamsIndex).^2).^2 ...
        + (pi/3*STEM_data.lambda^5*STEM_data.C5)*...
        (STEM_data.qxa(STEM_data.beamsIndex).^2+STEM_data.qya(STEM_data.beamsIndex).^2).^4;
    
    % Generate probe wave function
    probe_wf = sum(exp(-1i*chi ...
                    -2i*pi ...
                   *(STEM_data.qxa(STEM_data.beamsIndex).*xx*STEM_data.potential_pixelsize ...
                   + STEM_data.qya(STEM_data.beamsIndex).*yy*STEM_data.potential_pixelsize))...
                   ,1);

    probe_wf = reshape(probe_wf,[pot_size(1),pot_size(2)]);          
    
    fft_probe_wf = fft2(probe_wf);
    Ap = max(abs(fft_probe_wf(:)));
    probe_wfn = 1/Ap * probe_wf; %normalize
    STEM_data.probe_wfn = single(probe_wfn);
    STEM_data.init_wave2D = single(probe_wfn);

    if STEM_data.use_gpu == 1
         STEM_data.probe_wfn = gpuArray(STEM_data.probe_wfn);
    end
    
    %%%
    fprintf('Initialize... done.\n');

end