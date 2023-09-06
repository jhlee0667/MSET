% Author: Juhyeok Lee, 2023.

% Generate Fresnel freespace propagator & back propagator

% input:  STEM_data.pot_size
%         STEM_data.potential_pixelsize
%         STEM_data.lambda
%         STEM_data.slice_binning

% output: STEM_data.q2, STEM_data.qxa, STEM_data.qya            
%         STEM_data.prop
%         STEM_data.back_prop
%         STEM_data.prop2D (for cuda, same as STEM_data.prop)

function [STEM_data] = Func_generate_propagator(STEM_data)

    % Make the Fourier grid
    kx = 1:STEM_data.pot_size(1);
    ky = 1:STEM_data.pot_size(2);

    MultF_X = 1/(length(kx)*STEM_data.potential_pixelsize);
    MultF_Y = 1/(length(ky)*STEM_data.potential_pixelsize);

    CentPos = round((STEM_data.pot_size+1)/2);
    
    [STEM_data.qxa, STEM_data.qya] = ndgrid((kx-CentPos(1))*MultF_X,(ky-CentPos(2))*MultF_Y);
    STEM_data.q2 = STEM_data.qxa.^2 + STEM_data.qya.^2;

    % propagators and mask
    %STEM_data.qMax = min(max(abs(STEM_data.qxa)),max(abs(STEM_data.qya)))/2;
    %qMask = false(STEM_data.pot_size);
    %qMask(STEM_data.CentPos(1)-round(STEM_data.pot_size(1)/4)+1:STEM_data.CentPos(1)+round(STEM_data.pot_size(1)/4),...
    %    STEM_data.CentPos(1)-round(STEM_data.pot_size(1)/4)+1:STEM_data.CentPos(1)+round(STEM_data.pot_size(1)/4)) = true;
    %STEM_data.qMask = qMask;
    
    % Generate Fresnel freespace propagator & back propagator
    STEM_data.prop = single(exp((-1i*pi*STEM_data.lambda*(STEM_data.potential_pixelsize*STEM_data.slice_binning))*STEM_data.q2));
    STEM_data.back_prop = single(exp((+1i*pi*STEM_data.lambda*(STEM_data.potential_pixelsize*STEM_data.slice_binning))*STEM_data.q2));
    STEM_data.prop2D = single(STEM_data.prop);
    
end