% Author: J.Lee, KAIST (Korea), 2022.
% Y.Yang, Multi-Dimensional Atomic Imaging Lab
% Multislice simulation package


function [STEM_data] = STEP03_BACK(STEM_data) 
        
    % Calculate conjugate transmission function
    conj_trans = conj(STEM_data.trans);    
    back_wave_f = single(STEM_data.resi_vec);

    for a2 = STEM_data.numPlanes:-1:1
        back_wave_f = fftshift((ifftn(ifftshift(back_wave_f.*STEM_data.back_prop))));
        %STEM_data.grad(:,:,STEM_data.slice_binning*(a2-1)+1) = +1i*STEM_data.sigma*conj_trans(:,:,a2)...
            %    .*conj(STEM_data.save_wave_f(:,:,a2)).*back_wave_f;
        STEM_data.grad(:,:,STEM_data.slice_binning*(a2-1)+1) = +1i*conj_trans(:,:,a2)...
                .*conj(STEM_data.save_wave_f(:,:,a2)).*back_wave_f;

        if a2 < STEM_data.numPlanes
            STEM_data.grad(:,:,STEM_data.slice_binning*(a2-1)+1:STEM_data.slice_binning*a2) = repmat(STEM_data.grad(:,:,STEM_data.slice_binning*(a2-1)+1), [1, 1, STEM_data.slice_binning]);
        else
            STEM_data.grad(:,:,STEM_data.slice_binning*(a2-1)+1:size(STEM_data.rec,3)) = repmat(STEM_data.grad(:,:,STEM_data.slice_binning*(a2-1)+1), [1, 1, size(STEM_data.rec,3)-STEM_data.slice_binning*(a2-1)]);
        end
        
        back_wave_f = fftshift(fftn((ifftshift(back_wave_f.*conj_trans(:,:,a2)))));
    end

end