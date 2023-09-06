% Author: Juhyeok Lee, 2023.

% Generate complex probe wave function

% input:  STEM_data.probeDefocus
%         STEM_data.aberration_paras
%         %% aberration parameters %%
%         {C10 -> -1*probe defocus
%          C12 -> two fold astigmatism (A1)
%          C21 -> axial comma (B2)
%          C30 -> 3rd spherical aberration
%          C50 -> 5rd spherical aberration}
%         %%%%%
%         STEM_data.alpha
%         STEM_data.lambda
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
        STEM_data.aberration_paras.C10 = zeros(1,size(STEM_data.tilt_angles,1));
        STEM_data.aberration_paras.C30 = zeros(1,size(STEM_data.tilt_angles,1));
        STEM_data.aberration_paras.C50 = zeros(1,size(STEM_data.tilt_angles,1));
    else
        % C10
        if any(ismember(fields(STEM_data.aberration_paras),'C10'))
            if size(STEM_data.aberration_paras.C10, 2) == 1
                STEM_data.aberration_paras.C10 = ones(1,size(STEM_data.tilt_angles,1)).* STEM_data.aberration_paras.C10;
            elseif size(STEM_data.aberration_paras.C10,2) ~= size(STEM_data.tilt_angles,1)
                error('input error: make C10 list size and # of tilt angle be same');
            end
        else
            STEM_data.aberration_paras.C10 = zeros(1,size(STEM_data.tilt_angles,1));
        end
        % C30
        if any(ismember(fields(STEM_data.aberration_paras),'C30'))
            if size(STEM_data.aberration_paras.C30, 2) == 1
                STEM_data.aberration_paras.C30 = ones(1,size(STEM_data.tilt_angles,1)).* STEM_data.aberration_paras.C30;
            elseif size(STEM_data.aberration_paras.C30,2) ~= size(STEM_data.tilt_angles,1)
                error('input error: make C30 list size and # of tilt angle be same');
            end
        else
            STEM_data.aberration_paras.C30 = zeros(1,size(STEM_data.tilt_angles,1));
        end
        % C50
        if any(ismember(fields(STEM_data.aberration_paras),'C50'))
            if size(STEM_data.aberration_paras.C50, 2) == 1
                STEM_data.aberration_paras.C50 = ones(1,size(STEM_data.tilt_angles,1)).* STEM_data.aberration_paras.C50;
            elseif size(STEM_data.aberration_paras.C50,2) ~= size(STEM_data.tilt_angles,1)
                error('input error: make C50 list size and # of tilt angle be same');
            end
        else
            STEM_data.aberration_paras.C50 = zeros(1,size(STEM_data.tilt_angles,1));
        end
    end

    if any(ismember(fields(STEM_data),'probeDefocus'))
        if size(STEM_data.probeDefocus, 2) == 1
            STEM_data.aberration_paras.C10 = -1*ones(1,size(STEM_data.tilt_angles,1)).* STEM_data.probeDefocus;
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
    q2_alpha = STEM_data.lambda*q2;

    % make beam mask
    alphaBeamMax = STEM_data.alpha/1000; % in rads, for specifying maximum angle
    beam_mask = q2 < (alphaBeamMax/STEM_data.lambda)^2;


    % Generate defocus factor + aberration
    for p = 1:size(STEM_data.tilt_angles,1)
        
        % calculate chi
        chi = (1/2*STEM_data.aberration_paras.C10(p)) * q2_alpha...
            + (1/4*STEM_data.aberration_paras.C30(p)) * q2_alpha.^2 ...
            + (1/6*STEM_data.aberration_paras.C50(p)) * q2_alpha.^4;


        chi = 2*pi/STEM_data.lambda*chi;
        chi = exp(-1i*chi);

        % Generate probe wave function
        probe_wf = fftshift(fft2((ifftshift(beam_mask.*chi))));
        probe_wf = sqrt(STEM_data.mean_intensity_list(p)/sum(abs(fft2(probe_wf)).^2,[1 2])) ...
                   * probe_wf; %normalize

        STEM_data.probe_wfn(:,:,p) = single(probe_wf);

    end
    
end