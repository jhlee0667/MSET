% Author: J.Lee, KAIST (Korea), 2022.
% Y.Yang, Multi-Dimensional Atomic Imaging Lab
% Multislice electron tomography package


function [STEM_data] = STEP03_BACK_SSET(STEM_data) 
        
    % Calculate conjugate transmission function   
    back_wave_f = single(STEM_data.resi_vec);

    back_wave_f = fftshift((ifftn(ifftshift(back_wave_f))));

    STEM_data.grad = repmat(+1i*conj(STEM_data.proj_trans).*conj(STEM_data.save_wave_f)...
                   .*back_wave_f,1,1,size(STEM_data.rec,3));
    
    back_wave_f = back_wave_f.*conj(STEM_data.proj_trans);

    
    % update probe wave
    shift_pixels=2*round((size(STEM_data.tmp_probe_wave)+1)/2) - [STEM_data.row,STEM_data.col];
    STEM_data.probe_wave(:,:,STEM_data.Nth_angle) = STEM_data.probe_wave(:,:,STEM_data.Nth_angle) ...
        - STEM_data.step_size(2) .*single(Func_generate_probe_wave_scan_sub_pixel(back_wave_f,shift_pixels(1),shift_pixels(2)));
    
    % update scan position
    conj_tmp_probe_wave = conj(STEM_data.tmp_probe_wave);
    [de_probey, de_probex] = gradient(conj_tmp_probe_wave); % derivative for scan position optimization
    STEM_data.row = STEM_data.row + STEM_data.step_size(3) * real(sum(de_probex.* back_wave_f,[1 2]));
    STEM_data.col = STEM_data.col + STEM_data.step_size(3) * real(sum(de_probey.* back_wave_f,[1 2]));

end

