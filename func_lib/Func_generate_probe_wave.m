% Author: Juhyeok Lee, 2023.

% Generate complex probe wave function

% input:  STEM_data.probeDefocus, STEM_data.C3, STEM_data.C5
%         STEM_data.alphaBeamMax, STEM_data.lambda
%         STEM_data.tilt_angles
%         STEM_data.pot_size
%         STEM_data.q2, STEM_data.qxa, STEM_data.qya    
%         STEM_data.mean_intensity_list
%         STEM_data.lambda
%         STEM_data.slice_binning

% output: STEM_data.probe_wfn

function [STEM_data] = Func_generate_probe_wave(STEM_data)
        
        % Probe wave function parameter initialize
        if size(STEM_data.probeDefocus,2) == 1
            STEM_data.probeDefocus = ones(1,size(STEM_data.tilt_angles,1)).* STEM_data.probeDefocus;
            probeflag = 1;
        elseif size(STEM_data.probeDefocus,2) ~= size(STEM_data.tilt_angles,1)
            error('input error: make sizes of probeDefocus and tilt angle be same');
        else
            probeflag = 0;
        end
        if size(STEM_data.C3,2) == 1
            STEM_data.C3 = ones(1,size(STEM_data.tilt_angles,1)).* STEM_data.C3;
            probeflag = probeflag*1;
        elseif size(STEM_data.C3,2) ~= size(STEM_data.tilt_angles,1)
            error('input error: make sizes of C3 aberration and tilt angle be same');
        else
            probeflag = probeflag*0;
        end
        if size(STEM_data.C5,2) == 1
            STEM_data.C5 = ones(1,size(STEM_data.tilt_angles,1)).* STEM_data.C5;
            probeflag = probeflag*1;
        elseif size(STEM_data.C5,2) ~= size(STEM_data.tilt_angles,1)
            error('input error: make sizes of C5 aberration and tilt angle be same');
        else
            probeflag = probeflag*0;    
        end
        
        % make beam mask
        alphaBeamMax = STEM_data.alpha/1000; % in rads, for specifying maximum angle
        beam_mask = STEM_data.q2 < (alphaBeamMax/STEM_data.lambda)^2;
    
        % Count number of beams
        numberBeams = sum(beam_mask(:));
        beams = zeros(STEM_data.pot_size);
        beams(beam_mask==true) = 1:numberBeams;
        beamsIndex = find(beams);    
        
        % generate real space grid
        [xx,yy]  = ndgrid((1:STEM_data.pot_size(1))-round((STEM_data.pot_size(1)+1)/2),...
                          (1:STEM_data.pot_size(2))-round((STEM_data.pot_size(2)+1)/2));
        
        xx = reshape(xx,[1,prod(STEM_data.pot_size)]);
        yy = reshape(yy,[1,prod(STEM_data.pot_size)]);

    
        % Generate defocus factor + aberration
        STEM_data.probeC1 = -1 .* STEM_data.probeDefocus;
        for p = 1:size(STEM_data.tilt_angles,1)
            if (probeflag == 0) || (probeflag == 1 && p == 1)
                chi = (pi*STEM_data.lambda*STEM_data.probeC1(p))*...
                    (STEM_data.qxa(beamsIndex).^2+STEM_data.qya(beamsIndex).^2)...
                    + (pi/2*STEM_data.lambda^3*STEM_data.C3(p))*...
                    (STEM_data.qxa(beamsIndex).^2+STEM_data.qya(beamsIndex).^2).^2 ...
                    + (pi/3*STEM_data.lambda^5*STEM_data.C5(p))*...
                    (STEM_data.qxa(beamsIndex).^2+STEM_data.qya(beamsIndex).^2).^4;
                
                % Generate probe wave function
                if numberBeams > 100
                    Memory_saving_N = 100;
                else
                    Memory_saving_N = 1;
                end
                Nstep = fix(numberBeams/Memory_saving_N); 
                probe_wf =zeros(size(xx));
                for i = 1:Memory_saving_N
                    if i<=Memory_saving_N-1
                        probe_wf = probe_wf + sum(exp(-1i*chi(Nstep*(i-1)+1:Nstep*i) ...
                                        -2i*pi ...
                                       *(STEM_data.qxa(beamsIndex(Nstep*(i-1)+1:Nstep*i)).*xx*STEM_data.potential_pixelsize ...
                                       + STEM_data.qya(beamsIndex(Nstep*(i-1)+1:Nstep*i)).*yy*STEM_data.potential_pixelsize))...
                                       ,1);
                    else
                        probe_wf = probe_wf + sum(exp(-1i*chi(Nstep*(i-1)+1:end) ...
                                        -2i*pi ...
                                       *(STEM_data.qxa(beamsIndex(Nstep*(i-1)+1:end)).*xx*STEM_data.potential_pixelsize ...
                                       + STEM_data.qya(beamsIndex(Nstep*(i-1)+1:end)).*yy*STEM_data.potential_pixelsize))...
                                       ,1);
                    end
            
                end
    
                probe_wf = reshape(probe_wf,[STEM_data.pot_size(1),STEM_data.pot_size(2)]);   
                probe_wf = sqrt(STEM_data.mean_intensity_list(p)/sum(abs(fft2(probe_wf)).^2,[1 2])) ...
                    * probe_wf; %normalize
                
            end
            
            STEM_data.probe_wfn(:,:,p) = single(probe_wf);
        end
    
end