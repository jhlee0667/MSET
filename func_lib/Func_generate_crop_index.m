% Author: J.Lee, 2023

% generate index for cropping area in reconstruction volume
% input: xp (x center position, positive integer)
%        yp (y center position, positive integer)
%        pot_size (potential size)
%        rec_size (reconstruction vol size)

% output: xindx (x index list)
%         yindx (y index list)

function [xindx, yindx]=Func_generate_crop_index(xp, yp, pot_size, rec_size)
    
    xp = mod(xp, rec_size(1));
    yp = mod(yp, rec_size(2));

    % xindex
    if xp+fix((pot_size(1)-1)/2) > rec_size(1)
        xindx = [(xp-fix(pot_size(1)/2)):rec_size(1), 1:(xp+fix((pot_size(1)-1)/2)-rec_size(1))];
    elseif xp-fix(pot_size(1)/2) < 1
        xindx = [(rec_size(1)+xp-fix(pot_size(1)/2)):rec_size(1), 1:(xp+fix((pot_size(1)-1)/2))];
    else
        xindx = (xp-fix(pot_size(1)/2)):(xp+fix((pot_size(1)-1)/2));
    end

    % yindex
    if yp+fix((pot_size(2)-1)/2) > rec_size(2)
        yindx = [(yp-fix(pot_size(2)/2)):rec_size(2), 1:(yp+fix((pot_size(2)-1)/2)-rec_size(2))];
    elseif yp-fix(pot_size(2)/2) < 1
        yindx = [(rec_size(2)+yp-fix(pot_size(2)/2)):rec_size(2), 1:(yp+fix((pot_size(2)-1)/2))];
    else
        yindx = (yp-fix(pot_size(2)/2)):(yp+fix((pot_size(2)-1)/2));
    end

end