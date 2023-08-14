% Author: J.Lee, KAIST (Korea), 2022.
% Y.Yang, Multi-Dimensional Atomic Imaging Lab
% Multislice electron tomography package


function [STEM_data] = STEP02_FORWARD_4D(STEM_data) 
    
    % make slices of 2D projected potentials
    RotVol = STEM_data.RVol;
    
    % Calculate transmission function
    for k=1:STEM_data.numPlanes
       if k < STEM_data.numPlanes
           STEM_data.potz(:,:,k) = sum(RotVol(:,:,STEM_data.slice_binning*(k-1)+1:STEM_data.slice_binning*k),3) .* STEM_data.potential_pixelsize;
       else
           STEM_data.potz(:,:,k) = sum(RotVol(:,:,STEM_data.slice_binning*(k-1)+1:size(RotVol,3)),3) .* STEM_data.potential_pixelsize;
       end
    end
    
    STEM_data.trans = single(exp(1i*STEM_data.sigma*STEM_data.potz));
    STEM_data.prop = single(STEM_data.prop);

    % index
    [x1,y1] = ind2sub([STEM_data.N_scan_x, STEM_data.N_scan_y], STEM_data.k);

    % Calculate incident beam wave function
    STEM_data.tmp_probe_wfn = single(Func_generate_probe_wave_scan_sub_pixel(STEM_data.probe_wfn(:,:,STEM_data.Nth_angle),STEM_data.row,STEM_data.col));
    wave_f = STEM_data.tmp_probe_wfn;
    
    % Forward propagation
    for a2 = 1:STEM_data.numPlanes
        wave_f = fftshift(fftn((ifftshift(wave_f.*STEM_data.trans(:,:,a2))))).*STEM_data.prop;   
        wave_f = fftshift((ifftn(ifftshift(wave_f))));
        STEM_data.save_wave_f(:,:,a2) = wave_f;
    end
    wave_f = fftshift(fftn((ifftshift(wave_f))));
    
    STEM_data.calculated_4D_data{x1,y1} = abs(wave_f).^2;
    
    % load measured 4D data
    measured_4D_data = STEM_data.measured_4D_data{x1,y1};
    
    % calculate residual vector
    STEM_data.resi_vec = wave_f.*(1-(measured_4D_data).^(0.5)./(STEM_data.calculated_4D_data{x1,y1}+10^(-30)).^(0.5));
    %STEM_data.resi_vec = exp(1i*angle(wave_f)).*(abs(wave_f)-(measured_4D_data).^(0.5));

    % estimate error
    tmp_error = (measured_4D_data.^(0.5)-STEM_data.calculated_4D_data{x1,y1}.^(0.5)).^2;
    STEM_data.error(STEM_data.Nth_angle) = STEM_data.error(STEM_data.Nth_angle) + mean(abs(tmp_error(:)))/STEM_data.N_scan_x/STEM_data.N_scan_y;

end



