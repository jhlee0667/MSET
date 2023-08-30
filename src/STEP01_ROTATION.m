% Author: J.Lee, KAIST (Korea), 2022.
% Y.Yang, Multi-Dimensional Atomic Imaging Lab


function [STEM_data] = STEP01_ROTATION(STEM_data) 

    Vol = STEM_data.rec;
    
    if STEM_data.device ==1
        Vol = gather(Vol);
    end

    STEM_data.RVol = single(Func_inv_rot_3Dvol_FourierShear(Vol, STEM_data.vec1, STEM_data.vec2, STEM_data.vec3,...
        STEM_data.tilt_angles(STEM_data.Nth_angle,1), STEM_data.tilt_angles(STEM_data.Nth_angle,2), STEM_data.tilt_angles(STEM_data.Nth_angle,3)));

    if STEM_data.device ==1
        STEM_data.RVol = gpuArray(STEM_data.RVol);
    end


end

