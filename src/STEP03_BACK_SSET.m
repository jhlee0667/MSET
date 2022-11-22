% Author: J.Lee, KAIST (Korea), 2022.
% Y.Yang, Multi-Dimensional Atomic Imaging Lab
% Multislice simulation package


function [STEM_data] = STEP03_BACK_SSET(STEM_data) 
        
    % Calculate conjugate transmission function   
    back_wave_f = single(STEM_data.resi_vec);

    back_wave_f = fftshift((ifftn(ifftshift(back_wave_f))));

    STEM_data.grad = repmat(+1i*conj(STEM_data.proj_trans).*conj(STEM_data.save_wave_f)...
                   .*back_wave_f,1,1,size(STEM_data.rec,3));
    
end

