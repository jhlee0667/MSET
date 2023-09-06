% Author: Juhyeok Lee, 2023.

% Generate Fresnel freespace propagator & back propagator

% input:  STEM_data.pot_size
%         STEM_data.potential_pixelsize
%         STEM_data.lambda
%         STEM_data.slice_binning

% output: STEM_data.prop
%         STEM_data.back_prop
%         STEM_data.prop2D (for cuda, same as STEM_data.prop)

function [STEM_data] = Func_generate_propagator(STEM_data)

    % Make the Fourier grid
    kx = 1:STEM_data.pot_size(1);
    ky = 1:STEM_data.pot_size(2);

    MultF_X = 1/(length(kx)*STEM_data.potential_pixelsize);
    MultF_Y = 1/(length(ky)*STEM_data.potential_pixelsize);

    CentPos = round((STEM_data.pot_size+1)/2);
    
    [qxa, qya] = ndgrid((kx-CentPos(1))*MultF_X,(ky-CentPos(2))*MultF_Y);
    q2 = qxa.^2 + qya.^2;

    % propagators and mask
    %STEM_data.qMax = min(max(abs(qxa)),max(abs(qya)))/2;
    %qMask = false(STEM_data.pot_size);
    %qMask(CentPos(1)-round(STEM_data.pot_size(1)/4)+1:CentPos(1)+round(STEM_data.pot_size(1)/4),...
    %      CentPos(1)-round(STEM_data.pot_size(1)/4)+1:CentPos(1)+round(STEM_data.pot_size(1)/4)) = true;
    %STEM_data.qMask = qMask;
    
    % Generate Fresnel freespace propagator & back propagator
    STEM_data.prop = single(exp((-1i*pi*STEM_data.lambda*(STEM_data.potential_pixelsize*STEM_data.slice_binning))*q2));
    STEM_data.back_prop = single(exp((+1i*pi*STEM_data.lambda*(STEM_data.potential_pixelsize*STEM_data.slice_binning))*q2));
    STEM_data.prop2D = single(STEM_data.prop);
    
end