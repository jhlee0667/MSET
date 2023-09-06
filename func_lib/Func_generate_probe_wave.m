% Author: Juhyeok Lee, 2023.

% Generate complex probe wave function

% input:  STEM_data.probeDefocus, STEM_data.aberration_paras.{C1, C3, C5},
%         STEM_data.alpha, STEM_data.lambda
%         STEM_data.tilt_angles
%         STEM_data.pot_size
%         STEM_data.mean_intensity_list
%         STEM_data.lambda
%         STEM_data.slice_binning

% output: STEM_data.probe_wfn

function [STEM_data] = Func_generate_probe_wave(STEM_data)
    
    % check: probe wavefunction parameters
    if ~any(ismember(fields(STEM_data),'alpha'))
        error('input error: put probe forming aperture value (convergence semi-angle) into "alpha".');
    end

    if ~any(ismember(fields(STEM_data),'aberration_paras'))
        STEM_data.aberration_paras.C1 = zeros(1,size(STEM_data.tilt_angles,1));
        STEM_data.aberration_paras.C3 = zeros(1,size(STEM_data.tilt_angles,1));
        STEM_data.aberration_paras.C5 = zeros(1,size(STEM_data.tilt_angles,1));
    else
        % C1
        if any(ismember(fields(STEM_data.aberration_paras),'C1'))
            if size(STEM_data.aberration_paras.C1, 2) == 1
                STEM_data.aberration_paras.C1 = ones(1,size(STEM_data.tilt_angles,1)).* STEM_data.aberration_paras.C1;
            elseif size(STEM_data.aberration_paras.C1,2) ~= size(STEM_data.tilt_angles,1)
                error('input error: make C1 list size and # of tilt angle be same');
            end
        else
            STEM_data.aberration_paras.C1 = zeros(1,size(STEM_data.tilt_angles,1));
        end
        % C3
        if any(ismember(fields(STEM_data.aberration_paras),'C3'))
            if size(STEM_data.aberration_paras.C3, 2) == 1
                STEM_data.aberration_paras.C3 = ones(1,size(STEM_data.tilt_angles,1)).* STEM_data.aberration_paras.C3;
            elseif size(STEM_data.aberration_paras.C3,2) ~= size(STEM_data.tilt_angles,1)
                error('input error: make C3 list size and # of tilt angle be same');
            end
        else
            STEM_data.aberration_paras.C3 = zeros(1,size(STEM_data.tilt_angles,1));
        end
        % C5
        if any(ismember(fields(STEM_data.aberration_paras),'C5'))
            if size(STEM_data.aberration_paras.C5, 2) == 1
                STEM_data.aberration_paras.C5 = ones(1,size(STEM_data.tilt_angles,1)).* STEM_data.aberration_paras.C5;
            elseif size(STEM_data.aberration_paras.C5,2) ~= size(STEM_data.tilt_angles,1)
                error('input error: make C5 list size and # of tilt angle be same');
            end
        else
            STEM_data.aberration_paras.C5 = zeros(1,size(STEM_data.tilt_angles,1));
        end
    end

    if any(ismember(fields(STEM_data),'probeDefocus'))
        if size(STEM_data.probeDefocus, 2) == 1
            STEM_data.aberration_paras.C1 = -1*ones(1,size(STEM_data.tilt_angles,1)).* STEM_data.probeDefocus;
        elseif size(STEM_data.probeDefocus,2) ~= size(STEM_data.tilt_angles,1)
            error('input error: make probeDefocus list size and # of tilt angle be same');
        end
    end

    
    % Make the Fourier grid
    kx = 1:STEM_data.pot_size(1);
    ky = 1:STEM_data.pot_size(2);

    MultF_X = 1/(length(kx)*STEM_data.potential_pixelsize);
    MultF_Y = 1/(length(ky)*STEM_data.potential_pixelsize);

    CentPos = round((STEM_data.pot_size+1)/2);
    
    [qxa, qya] = ndgrid((kx-CentPos(1))*MultF_X,(ky-CentPos(2))*MultF_Y);
    q2 = qxa.^2 + qya.^2;


    % make beam mask
    alphaBeamMax = STEM_data.alpha/1000; % in rads, for specifying maximum angle
    beam_mask = q2 < (alphaBeamMax/STEM_data.lambda)^2;


    % Generate defocus factor + aberration
    for p = 1:size(STEM_data.tilt_angles,1)
        
        % calculate chi
        chi = (pi*STEM_data.lambda*STEM_data.aberration_paras.C1(p)) * q2...
            + (pi/2*STEM_data.lambda^3*STEM_data.aberration_paras.C3(p)) * q2.^2 ...
            + (pi/3*STEM_data.lambda^5*STEM_data.aberration_paras.C5(p)) * q2.^4;

        chi = exp(-1i*chi);

        % Generate probe wave function
        probe_wf = fftshift(fft2((ifftshift(beam_mask.*chi))));
        probe_wf = sqrt(STEM_data.mean_intensity_list(p)/sum(abs(fft2(probe_wf)).^2,[1 2])) ...
                   * probe_wf; %normalize

        STEM_data.probe_wfn(:,:,p) = single(probe_wf);

    end
    
end