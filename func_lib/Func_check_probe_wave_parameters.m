% Author: Juhyeok Lee, 2023.

% Check probe wave parameters including aberration_parameters

% input:  STEM_data.probeDefocus
%         STEM_data.aberration_paras
%         %% aberration parameters %%
%         C10 -> defocus, -1*probe defocus
%         C30 -> 3rd spherical aberration
%         C50 -> 5rd spherical aberration
%
%         C12, Phi12 -> two fold astigmatism (A1)
%         C23, Phi23 -> Three fold astigmatism (A2) 
%         C21, Phi21 -> axial comma (B2)
%         
%         %%%%%

% output: STEM_data.aberration_paras

function [STEM_data] = Func_check_probe_wave_parameters(STEM_data)

    if ~any(ismember(fields(STEM_data),'alpha'))
        error('input error: put probe forming aperture value (convergence semi-angle) into "alpha".');
    end
    
    if ~any(ismember(fields(STEM_data),'aberration_paras'))
        STEM_data.aberration_paras.C10 = zeros(1,size(STEM_data.tilt_angles,1));
        STEM_data.aberration_paras.C30 = zeros(1,size(STEM_data.tilt_angles,1));
        STEM_data.aberration_paras.C50 = zeros(1,size(STEM_data.tilt_angles,1));
    
        STEM_data.aberration_paras.C12 = zeros(1,size(STEM_data.tilt_angles,1));
        STEM_data.aberration_paras.Phi12 = zeros(1,size(STEM_data.tilt_angles,1));
        STEM_data.aberration_paras.C23 = zeros(1,size(STEM_data.tilt_angles,1));
        STEM_data.aberration_paras.Phi23 = zeros(1,size(STEM_data.tilt_angles,1));

        STEM_data.aberration_paras.C21 = zeros(1,size(STEM_data.tilt_angles,1));
        STEM_data.aberration_paras.Phi21 = zeros(1,size(STEM_data.tilt_angles,1));
    
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
    
        % C12
        if any(ismember(fields(STEM_data.aberration_paras),'C12'))
            if size(STEM_data.aberration_paras.C12, 2) == 1
                STEM_data.aberration_paras.C12 = ones(1,size(STEM_data.tilt_angles,1)).* STEM_data.aberration_paras.C12;
            elseif size(STEM_data.aberration_paras.C12,2) ~= size(STEM_data.tilt_angles,1)
                error('input error: make C12 list size and # of tilt angle be same');
            end
        else
            STEM_data.aberration_paras.C12 = zeros(1,size(STEM_data.tilt_angles,1));
        end        
        % Phi12
        if any(ismember(fields(STEM_data.aberration_paras),'Phi12'))
            if size(STEM_data.aberration_paras.Phi12, 2) == 1
                STEM_data.aberration_paras.Phi12 = ones(1,size(STEM_data.tilt_angles,1)).* STEM_data.aberration_paras.Phi12;
            elseif size(STEM_data.aberration_paras.Phi12,2) ~= size(STEM_data.tilt_angles,1)
                error('input error: make Phi12 list size and # of tilt angle be same');
            end
        else
            STEM_data.aberration_paras.Phi12 = zeros(1,size(STEM_data.tilt_angles,1));
        end    

        % C23
        if any(ismember(fields(STEM_data.aberration_paras),'C23'))
            if size(STEM_data.aberration_paras.C23, 2) == 1
                STEM_data.aberration_paras.C23 = ones(1,size(STEM_data.tilt_angles,1)).* STEM_data.aberration_paras.C23;
            elseif size(STEM_data.aberration_paras.C23,2) ~= size(STEM_data.tilt_angles,1)
                error('input error: make C23 list size and # of tilt angle be same');
            end
        else
            STEM_data.aberration_paras.C23 = zeros(1,size(STEM_data.tilt_angles,1));
        end        
        % Phi23
        if any(ismember(fields(STEM_data.aberration_paras),'Phi23'))
            if size(STEM_data.aberration_paras.Phi23, 2) == 1
                STEM_data.aberration_paras.Phi23 = ones(1,size(STEM_data.tilt_angles,1)).* STEM_data.aberration_paras.Phi23;
            elseif size(STEM_data.aberration_paras.Phi23,2) ~= size(STEM_data.tilt_angles,1)
                error('input error: make Phi23 list size and # of tilt angle be same');
            end
        else
            STEM_data.aberration_paras.Phi23 = zeros(1,size(STEM_data.tilt_angles,1));
        end   

        % C21
        if any(ismember(fields(STEM_data.aberration_paras),'C21'))
            if size(STEM_data.aberration_paras.C21, 2) == 1
                STEM_data.aberration_paras.C21 = ones(1,size(STEM_data.tilt_angles,1)).* STEM_data.aberration_paras.C21;
            elseif size(STEM_data.aberration_paras.C21,2) ~= size(STEM_data.tilt_angles,1)
                error('input error: make C21 list size and # of tilt angle be same');
            end
        else
            STEM_data.aberration_paras.C21 = zeros(1,size(STEM_data.tilt_angles,1));
        end
        % Phi21
        if any(ismember(fields(STEM_data.aberration_paras),'Phi21'))
            if size(STEM_data.aberration_paras.Phi12, 2) == 1
                STEM_data.aberration_paras.Phi21 = ones(1,size(STEM_data.tilt_angles,1)).* STEM_data.aberration_paras.Phi21;
            elseif size(STEM_data.aberration_paras.Phi21,2) ~= size(STEM_data.tilt_angles,1)
                error('input error: make Phi21 list size and # of tilt angle be same');
            end
        else
            STEM_data.aberration_paras.Phi21 = zeros(1,size(STEM_data.tilt_angles,1));
        end   
    
    end
    
    if any(ismember(fields(STEM_data),'probeDefocus'))
        if size(STEM_data.probeDefocus, 2) == 1
            STEM_data.aberration_paras.C10 = -1*ones(1,size(STEM_data.tilt_angles,1)).* STEM_data.probeDefocus;
        elseif size(STEM_data.probeDefocus,2) ~= size(STEM_data.tilt_angles,1)
            error('input error: make probeDefocus list size and # of tilt angle be same');
        end
    end


end