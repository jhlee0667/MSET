% Author: Juhyeok Lee, 2023.

% Generate complex probe wave function

% input:  STEM_data.aberration_paras
%         STEM_data.alpha
%         STEM_data.lambda
%         STEM_data.tilt_angles
%         STEM_data.pot_size
%         STEM_data.lambda

% output: STEM_data.probe_wfn

function [STEM_data] = Func_generate_probe_wave(STEM_data)
    
    % check: probe wavefunction parameters
    [STEM_data] = Func_check_probe_wave_parameters(STEM_data);
    
    % Make the Fourier grid
    kx = 1:STEM_data.pot_size(1);
    ky = 1:STEM_data.pot_size(2);

    MultF_X = 1/(length(kx)*STEM_data.potential_pixelsize);
    MultF_Y = 1/(length(ky)*STEM_data.potential_pixelsize);

    CentPos = round((STEM_data.pot_size+1)/2);
    
    [qxa, qya] = ndgrid((kx-CentPos(1))*MultF_X,(ky-CentPos(2))*MultF_Y);
    q2 = qxa.^2 + qya.^2;
    q_alpha = STEM_data.lambda*q2.^(0.5);
    Phi = atan2(qya,qxa); % in rad
    
    % make beam mask
    alphaBeamMax = STEM_data.alpha/1000; % in rads, for specifying maximum angle
    beam_mask = q2 < (alphaBeamMax/STEM_data.lambda)^2;
    

    % Generate defocus factor + aberration
    for p = 1:size(STEM_data.tilt_angles,1)
        
        % calculate chi
        chi = 1/2*q_alpha.^2 * (STEM_data.aberration_paras.C10(p))  ...
            + 1/4*q_alpha.^4 * (STEM_data.aberration_paras.C30(p))  ...
            + 1/6*q_alpha.^6 * (STEM_data.aberration_paras.C50(p))  ...
            + 1/2*q_alpha.^2 * (STEM_data.aberration_paras.C12(p)) .* cos(2*(Phi-STEM_data.aberration_paras.Phi12(p))) ... % in rad
            + 1/3*q_alpha.^3 * (STEM_data.aberration_paras.C21(p)) .* cos(1*(Phi-STEM_data.aberration_paras.Phi21(p))) ... % in rad
            + 1/3*q_alpha.^3 * (STEM_data.aberration_paras.C23(p)) .* cos(3*(Phi-STEM_data.aberration_paras.Phi23(p)));

        chi = 2*pi/STEM_data.lambda*chi;
        chi = exp(-1i*chi);

        % Generate probe wave function
        STEM_data.probe_wave(:,:,p) = fftshift(fft2((ifftshift(beam_mask.*chi))));

    end
    
end